from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

import tempfile
import os
import time
import traceback
import math

from backend.road_service import get_highway_info


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "Read the Road API"
APP_VERSION = "2.0.0"

ROAD_MODEL_PATH = "yolo26n-sem.pt"

DEPTH_MODEL_NAME = (
    "depth-anything/"
    "Depth-Anything-V2-Metric-Outdoor-Small-hf"
)

MIN_IMAGE_WIDTH = 320
MIN_IMAGE_HEIGHT = 240

MAX_IMAGE_WIDTH = 1920
MAX_IMAGE_HEIGHT = 1920

DEFAULT_HFOV = 70.0
MIN_HFOV = 40.0
MAX_HFOV = 110.0


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="""
    AI-powered road width measurement system using semantic
    segmentation, metric depth estimation and 3D geometry.
    """,
    version=APP_VERSION
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# GLOBAL MODELS
# ============================================================

road_model = None
depth_model = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():

    global road_model
    global depth_model

    print()
    print("=" * 60)
    print("LOADING ROAD SEGMENTATION MODEL")
    print("=" * 60)

    road_model = YOLO(
        ROAD_MODEL_PATH
    )

    print(
        "Road segmentation model loaded successfully."
    )

    print()
    print("=" * 60)
    print("LOADING METRIC DEPTH MODEL")
    print("=" * 60)

    depth_model = pipeline(
        "depth-estimation",
        model=DEPTH_MODEL_NAME
    )

    print(
        "Metric depth model loaded successfully."
    )

    print()
    print("=" * 60)
    print("ALL MODELS READY")
    print("=" * 60)
    print()


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    load_models()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "application": APP_NAME,
        "version": APP_VERSION,
        "message": "Read the Road API is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "road_model_loaded": road_model is not None,
        "depth_model_loaded": depth_model is not None
    }


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image):

    if image is None:

        raise ValueError(
            "The uploaded file could not be decoded as an image."
        )

    if len(image.shape) < 2:

        raise ValueError(
            "Invalid image dimensions."
        )

    height, width = image.shape[:2]

    if width < MIN_IMAGE_WIDTH:

        raise ValueError(
            f"Image is too narrow. "
            f"Minimum width is {MIN_IMAGE_WIDTH}px."
        )

    if height < MIN_IMAGE_HEIGHT:

        raise ValueError(
            f"Image is too short. "
            f"Minimum height is {MIN_IMAGE_HEIGHT}px."
        )

    return True


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    height, width = image.shape[:2]

    if (
        width <= MAX_IMAGE_WIDTH
        and
        height <= MAX_IMAGE_HEIGHT
    ):

        return image

    scale = min(
        MAX_IMAGE_WIDTH / width,
        MAX_IMAGE_HEIGHT / height
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# FOV VALIDATION
# ============================================================

def validate_hfov(hfov):

    try:

        hfov = float(hfov)

    except Exception:

        return DEFAULT_HFOV

    if not math.isfinite(hfov):

        return DEFAULT_HFOV

    return max(
        MIN_HFOV,
        min(
            MAX_HFOV,
            hfov
        )
    )


# ============================================================
# ROAD SEGMENTATION
# ============================================================

def get_road_mask(image):

    if road_model is None:

        raise RuntimeError(
            "Road segmentation model is not loaded."
        )

    print(
        "Running semantic road segmentation..."
    )

    results = road_model.predict(
        image,
        verbose=False,
        imgsz=640
    )

    if not results:

        raise RuntimeError(
            "Segmentation model returned no results."
        )

    result = results[0]

    if result.semantic_mask is None:

        raise RuntimeError(
            "Semantic segmentation did not produce a mask."
        )

    semantic_data = (
        result.semantic_mask.data
    )

    mask = (
        semantic_data[0]
        .cpu()
        .numpy()
    )

    height, width = image.shape[:2]

    mask = cv2.resize(
        mask.astype(np.uint8),
        (
            width,
            height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    road_mask = (
        mask == 0
    ).astype(np.uint8)

    return road_mask


# ============================================================
# MASK CLEANING
# ============================================================

def clean_road_mask(mask):

    binary = (
        mask > 0
    ).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel
    )

    binary = cv2.dilate(
        binary,
        np.ones(
            (3, 3),
            dtype=np.uint8
        ),
        iterations=1
    )

    return binary


# ============================================================
# ROAD COVERAGE
# ============================================================

def calculate_road_coverage(mask):

    total_pixels = mask.size

    road_pixels = int(
        np.sum(mask > 0)
    )

    if total_pixels == 0:

        return 0.0

    return (
        road_pixels /
        total_pixels
    ) * 100.0


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

def analyze_components(mask):

    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        number_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        width = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        height = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
        )

    return components, labels


# ============================================================
# MAIN ROAD COMPONENT
# ============================================================

def get_main_road_component(mask):

    height, width = mask.shape

    components, labels = (
        analyze_components(mask)
    )

    if not components:

        return mask

    image_center = (
        width / 2.0
    )

    scored = []

    for component in components:

        area = component["area"]

        x = component["x"]

        y = component["y"]

        component_width = (
            component["width"]
        )

        component_height = (
            component["height"]
        )

        center_x = (
            x +
            component_width / 2.0
        )

        bottom = (
            y +
            component_height
        )

        area_score = (
            area /
            (height * width)
        )

        bottom_score = min(
            bottom / height,
            1.0
        )

        center_distance = abs(
            center_x -
            image_center
        )

        center_score = (
            1.0 -
            min(
                center_distance /
                (width / 2.0),
                1.0
            )
        )

        vertical_score = min(
            component_height /
            height,
            1.0
        )

        score = (

            area_score * 2.0

            +

            bottom_score * 3.0

            +

            center_score * 2.0

            +

            vertical_score * 1.5
        )

        scored.append(
            (
                score,
                component["label"]
            )
        )

    scored.sort(
        reverse=True
    )

    best_label = scored[0][1]

    return (
        labels == best_label
    ).astype(np.uint8)


# ============================================================
# FIND ROAD INTERVALS
# ============================================================

def find_row_intervals(row):

    xs = np.where(
        row > 0
    )[0]

    if len(xs) == 0:

        return []

    intervals = []

    start = int(
        xs[0]
    )

    previous = int(
        xs[0]
    )

    for value in xs[1:]:

        value = int(value)

        if (
            value -
            previous
            >
            6
        ):

            intervals.append(
                (
                    start,
                    previous
                )
            )

            start = value

        previous = value

    intervals.append(
        (
            start,
            previous
        )
    )

    return intervals


# ============================================================
# SELECT BEST ROAD INTERVAL
# ============================================================

def choose_best_interval(
    intervals,
    image_width
):

    if not intervals:

        return None

    center = (
        image_width /
        2.0
    )

    candidates = []

    for left, right in intervals:

        width = (
            right -
            left
        )

        if width < (
            image_width *
            0.05
        ):

            continue

        if width > (
            image_width *
            0.98
        ):

            continue

        interval_center = (
            left +
            right
        ) / 2.0

        center_distance = abs(
            interval_center -
            center
        )

        center_score = (
            1.0 -
            min(
                center_distance /
                center,
                1.0
            )
        )

        width_score = min(
            width /
            image_width,
            1.0
        )

        score = (
            center_score *
            0.65
            +
            width_score *
            0.35
        )

        candidates.append(
            (
                score,
                left,
                right
            )
        )

    if not candidates:

        return None

    candidates.sort(
        reverse=True
    )

    return (
        candidates[0][1],
        candidates[0][2]
    )
def extract_raw_boundaries(image, mask):

    height, width = mask.shape

    left_points = []
    right_points = []

    # --------------------------------------------------------
    # Use rows from the middle-lower part of the image.
    # --------------------------------------------------------

    y_values = np.linspace(
        int(height * 0.40),
        int(height * 0.96),
        100
    ).astype(int)

    # --------------------------------------------------------
    # First try the AI road mask.
    # --------------------------------------------------------

    for y in y_values:

        xs = np.where(
            mask[y] > 0
        )[0]

        if len(xs) < 2:
            continue

        # Find the outermost road pixels.
        left_x = int(xs[0])
        right_x = int(xs[-1])

        road_width = right_x - left_x

        if road_width < width * 0.03:
            continue

        left_points.append(
            (
                float(left_x),
                float(y)
            )
        )

        right_points.append(
            (
                float(right_x),
                float(y)
            )
        )

    # --------------------------------------------------------
    # If the AI mask is weak, use image edges.
    # --------------------------------------------------------

    if (
        len(left_points) < 10
        or
        len(right_points) < 10
    ):

        print(
            "AI boundaries weak."
        )

        print(
            "Running image edge fallback..."
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            gray,
            40,
            120
        )

        # ----------------------------------------------------
        # Restrict detection to road area.
        # ----------------------------------------------------

        roi = np.zeros_like(
            edges
        )

        polygon = np.array(
            [[
                [0, height],
                [width, height],
                [int(width * 0.72), int(height * 0.32)],
                [int(width * 0.28), int(height * 0.32)]
            ]],
            dtype=np.int32
        )

        cv2.fillPoly(
            roi,
            polygon,
            255
        )

        edges = cv2.bitwise_and(
            edges,
            roi
        )

        # ----------------------------------------------------
        # Detect long road-edge lines.
        # ----------------------------------------------------

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=int(
                height * 0.12
            ),
            maxLineGap=60
        )

        left_candidates = []
        right_candidates = []

        if lines is not None:

            for line in lines:

                line = np.asarray(line).reshape(-1)

                if len(line) != 4:
                    continue

                x1 = int(line[0])
                y1 = int(line[1])
                x2 = int(line[2])
                y2 = int(line[3])

                dx = x2 - x1
                dy = y2 - y1

                if abs(dy) < 15:
                    continue

                slope = dx / dy

                # Ignore almost-horizontal lines.
                if abs(slope) < 0.05:
                    continue

                # Project line to bottom of image.
                bottom_y = height - 1

                x_bottom = (
                    x1
                    +
                    slope *
                    (
                        bottom_y -
                        y1
                    )
                )

                # LEFT boundary
                if slope < 0:

                    if (
                        -width * 0.20
                        <
                        x_bottom
                        <
                        width * 0.70
                    ):

                        left_candidates.append(
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            )
                        )

                # RIGHT boundary
                else:

                    if (
                        width * 0.30
                        <
                        x_bottom
                        <
                        width * 1.20
                    ):

                        right_candidates.append(
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            )
                        )

        # ----------------------------------------------------
        # Fit left edge.
        # ----------------------------------------------------

        if len(left_candidates) > 0:

            left_x_values = []
            left_y_values = []

            for (
                x1,
                y1,
                x2,
                y2
            ) in left_candidates:

                left_x_values.extend(
                    [
                        x1,
                        x2
                    ]
                )

                left_y_values.extend(
                    [
                        y1,
                        y2
                    ]
                )

            if len(left_x_values) >= 4:

                left_fit = np.polyfit(
                    left_y_values,
                    left_x_values,
                    1
                )

                left_points = []

                for y in y_values:

                    x = np.polyval(
                        left_fit,
                        y
                    )

                    if (
                        -width * 0.20
                        <
                        x
                        <
                        width * 0.80
                    ):

                        left_points.append(
                            (
                                float(x),
                                float(y)
                            )
                        )

        # ----------------------------------------------------
        # Fit right edge.
        # ----------------------------------------------------

        if len(right_candidates) > 0:

            right_x_values = []
            right_y_values = []

            for (
                x1,
                y1,
                x2,
                y2
            ) in right_candidates:

                right_x_values.extend(
                    [
                        x1,
                        x2
                    ]
                )

                right_y_values.extend(
                    [
                        y1,
                        y2
                    ]
                )

            if len(right_x_values) >= 4:

                right_fit = np.polyfit(
                    right_y_values,
                    right_x_values,
                    1
                )

                right_points = []

                for y in y_values:

                    x = np.polyval(
                        right_fit,
                        y
                    )

                    if (
                        width * 0.20
                        <
                        x
                        <
                        width * 1.20
                    ):

                        right_points.append(
                            (
                                float(x),
                                float(y)
                            )
                        )

    # --------------------------------------------------------
    # Final fallback.
    # --------------------------------------------------------

    if len(left_points) < 6:

        left_points = []

        for y in y_values:

            xs = np.where(
                mask[y] > 0
            )[0]

            if len(xs) >= 2:

                left_points.append(
                    (
                        float(xs[0]),
                        float(y)
                    )
                )

    if len(right_points) < 6:

        right_points = []

        for y in y_values:

            xs = np.where(
                mask[y] > 0
            )[0]

            if len(xs) >= 2:

                right_points.append(
                    (
                        float(xs[-1]),
                        float(y)
                    )
                )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    if len(left_points) < 6:

        raise RuntimeError(
            "Could not detect the left road boundary."
        )

    if len(right_points) < 6:

        raise RuntimeError(
            "Could not detect the right road boundary."
        )

    return (
        np.array(
            left_points,
            dtype=np.float32
        ),
        np.array(
            right_points,
            dtype=np.float32
        )
    )
# ============================================================
# ROBUST LINE FIT
# ============================================================

def robust_line_fit(points):

    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise RuntimeError("Not enough points for boundary fitting.")

    x = points[:, 0]
    y = points[:, 1]

    # Fit x as a function of y. This matches the road-boundary
    # representation used throughout the measurement pipeline.
    coefficients = np.polyfit(y, x, 1)

    # A small robust-refinement loop removes gross outliers without
    # requiring any additional dependency.
    for _ in range(3):

        predicted = np.polyval(coefficients, y)
        residuals = np.abs(x - predicted)

        median_residual = float(np.median(residuals))
        threshold = max(8.0, median_residual * 2.5)

        inliers = residuals <= threshold

        if np.sum(inliers) < 2:
            break

        new_coefficients = np.polyfit(
            y[inliers],
            x[inliers],
            1
        )

        if np.allclose(new_coefficients, coefficients, atol=1e-4):
            coefficients = new_coefficients
            break

        coefficients = new_coefficients

    return (
        float(coefficients[0]),
        float(coefficients[1])
    )


# ============================================================
# BOUNDARY QUALITY
# ============================================================

def calculate_boundary_quality(
    points,
    coefficients
):

    x = points[:, 0]

    y = points[:, 1]

    predicted = np.polyval(
        coefficients,
        y
    )

    residuals = np.abs(
        x -
        predicted
    )

    median_error = float(
        np.median(
            residuals
        )
    )

    mean_error = float(
        np.mean(
            residuals
        )
    )

    quality = (
        100.0 *
        math.exp(
            -median_error /
            20.0
        )
    )

    quality = max(
        0.0,
        min(
            100.0,
            quality
        )
    )

    return {
        "median_error_px":
            round(
                median_error,
                2
            ),

        "mean_error_px":
            round(
                mean_error,
                2
            ),

        "quality_percent":
            round(
                quality,
                1
            )
    }


# ============================================================
# VANISHING POINT
# ============================================================

def estimate_vanishing_point(
    left_line,
    right_line,
    image_shape
):

    height, width = (
        image_shape[:2]
    )

    left_a, left_b = left_line

    right_a, right_b = right_line

    denominator = (
        left_a -
        right_a
    )

    if abs(denominator) < 1e-6:

        return (
            width / 2.0,
            height * 0.35
        )

    y = (
        right_b -
        left_b
    ) / denominator

    x = (
        left_a *
        y
        +
        left_b
    )

    if not math.isfinite(x):

        x = width / 2.0

    if not math.isfinite(y):

        y = height * 0.35

    x = max(
        -width,
        min(
            2 * width,
            x
        )
    )

    y = max(
        -height,
        min(
            2 * height,
            y
        )
    )

    return (
        float(x),
        float(y)
    )


# ============================================================
# METRIC DEPTH
# ============================================================

def get_metric_depth(
    image_rgb
):

    if depth_model is None:

        raise RuntimeError(
            "Metric depth model is not loaded."
        )

    print(
        "Running metric depth estimation..."
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    result = depth_model(
        pil_image
    )

    if (
        "predicted_depth"
        not in result
    ):

        raise RuntimeError(
            "Depth model did not return predicted_depth."
        )

    predicted_depth = (
        result["predicted_depth"]
    )

    if hasattr(
        predicted_depth,
        "detach"
    ):

        predicted_depth = (
            predicted_depth
            .detach()
            .cpu()
            .numpy()
        )

    depth = np.squeeze(
        predicted_depth
    )

    height, width = (
        image_rgb.shape[:2]
    )

    depth = cv2.resize(
        depth.astype(
            np.float32
        ),
        (
            width,
            height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    depth = np.nan_to_num(
        depth,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    depth = np.maximum(
        depth,
        0.01
    )

    return depth


# ============================================================
# DEPTH QUALITY
# ============================================================

def calculate_depth_quality(
    depth
):

    valid = depth[
        np.isfinite(depth)
        &
        (depth > 0)
    ]

    if len(valid) == 0:

        return 0.0

    valid_ratio = (
        len(valid) /
        depth.size
    )

    spread = (
        np.percentile(
            valid,
            90
        )
        -
        np.percentile(
            valid,
            10
        )
    )

    if spread <= 0:

        spread_score = 20.0

    else:

        spread_score = min(
            100.0,
            spread * 2.0
        )

    quality = (
        valid_ratio *
        70.0
        +
        spread_score *
        0.30
    )

    return round(
        max(
            0.0,
            min(
                100.0,
                quality
            )
        ),
        1
    )


# ============================================================
# CAMERA INTRINSICS
# ============================================================

def calculate_camera_intrinsics(
    image_width,
    image_height,
    hfov
):

    hfov_rad = math.radians(
        hfov
    )

    fx = (
        image_width / 2.0
    ) / math.tan(
        hfov_rad / 2.0
    )

    fy = fx

    cx = (
        image_width /
        2.0
    )

    cy = (
        image_height /
        2.0
    )

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy)
    }


# ============================================================
# PIXEL TO 3D
# ============================================================

def pixel_to_3d(
    x,
    y,
    depth,
    intrinsics
):

    fx = intrinsics["fx"]

    fy = intrinsics["fy"]

    cx = intrinsics["cx"]

    cy = intrinsics["cy"]

    z = float(
        depth
    )

    X = (
        (x - cx)
        *
        z
        /
        fx
    )

    Y = (
        (y - cy)
        *
        z
        /
        fy
    )

    Z = z

    return np.array(
        [
            X,
            Y,
            Z
        ],
        dtype=np.float32
    )


# ============================================================
# DEPTH SAMPLING
# ============================================================

def sample_depth(
    depth,
    x,
    y,
    radius=4
):

    height, width = depth.shape

    x = int(
        round(x)
    )

    y = int(
        round(y)
    )

    x = max(
        0,
        min(
            width - 1,
            x
        )
    )

    y = max(
        0,
        min(
            height - 1,
            y
        )
    )

    x0 = max(
        0,
        x - radius
    )

    x1 = min(
        width,
        x + radius + 1
    )

    y0 = max(
        0,
        y - radius
    )

    y1 = min(
        height,
        y + radius + 1
    )

    region = depth[
        y0:y1,
        x0:x1
    ]

    valid = region[
        np.isfinite(region)
        &
        (region > 0)
    ]

    if len(valid) == 0:

        return None

    return float(
        np.median(valid)
    )


# ============================================================
# CROSS-SECTIONS
# ============================================================

def generate_cross_sections(
    image_shape
):

    height, width = (
        image_shape[:2]
    )

    y_values = np.linspace(
        int(height * 0.55),
        int(height * 0.92),
        10
    ).astype(int)

    return [
        int(y)
        for y in y_values
    ]


# ============================================================
# 3D WIDTH CALCULATION
# ============================================================

def calculate_3d_widths(
    left_line,
    right_line,
    depth,
    intrinsics,
    image_shape
):

    height, width = (
        image_shape[:2]
    )

    y_values = (
        generate_cross_sections(
            image_shape
        )
    )

    measurements = []

    for y in y_values:

        left_x = float(
            np.polyval(
                left_line,
                y
            )
        )

        right_x = float(
            np.polyval(
                right_line,
                y
            )
        )

        left_x = float(
            np.clip(
                left_x,
                0,
                width - 1
            )
        )

        right_x = float(
            np.clip(
                right_x,
                0,
                width - 1
            )
        )

        if right_x <= left_x:

            continue

        pixel_width = (
            right_x -
            left_x
        )

        left_depth = sample_depth(
            depth,
            left_x,
            y
        )

        right_depth = sample_depth(
            depth,
            right_x,
            y
        )

        if left_depth is None:

            continue

        if right_depth is None:

            continue

        if (
            left_depth <= 0
            or
            right_depth <= 0
        ):

            continue

        left_3d = pixel_to_3d(
            left_x,
            y,
            left_depth,
            intrinsics
        )

        right_3d = pixel_to_3d(
            right_x,
            y,
            right_depth,
            intrinsics
        )

        width_m = float(
            np.linalg.norm(
                right_3d -
                left_3d
            )
        )

        if width_m < 1.0:

            continue

        if width_m > 50.0:

            continue

        measurements.append(
            {
                "y": int(y),

                "left_x":
                    round(
                        left_x,
                        1
                    ),

                "right_x":
                    round(
                        right_x,
                        1
                    ),

                "pixel_width":
                    round(
                        pixel_width,
                        1
                    ),

                "left_depth_m":
                    round(
                        left_depth,
                        2
                    ),

                "right_depth_m":
                    round(
                        right_depth,
                        2
                    ),

                "width_m":
                    round(
                        width_m,
                        2
                    )
            }
        )

    if len(measurements) < 4:

        raise RuntimeError(
            "Not enough reliable 3D cross-sections."
        )

    return measurements


# ============================================================
# WIDTH OUTLIER FILTERING
# ============================================================

def filter_width_outliers(
    measurements
):

    values = np.array(
        [
            item["width_m"]
            for item in measurements
        ],
        dtype=np.float32
    )

    median = float(
        np.median(values)
    )

    deviations = np.abs(
        values -
        median
    )

    mad = float(
        np.median(
            deviations
        )
    )

    threshold = max(
        1.0,
        mad * 3.0
    )

    filtered = []

    for item in measurements:

        difference = abs(
            item["width_m"] -
            median
        )

        if difference <= threshold:

            filtered.append(
                item
            )

    if len(filtered) < 4:

        return measurements

    return filtered


# ============================================================
# WIDTH STATISTICS
# ============================================================

def calculate_width_statistics(
    measurements
):

    values = np.array(
        [
            item["width_m"]
            for item in measurements
        ],
        dtype=np.float32
    )

    return {

        "median_m":
            float(
                np.median(values)
            ),

        "mean_m":
            float(
                np.mean(values)
            ),

        "std_m":
            float(
                np.std(values)
            ),

        "min_m":
            float(
                np.min(values)
            ),

        "max_m":
            float(
                np.max(values)
            ),

        "q25_m":
            float(
                np.percentile(
                    values,
                    25
                )
            ),

        "q75_m":
            float(
                np.percentile(
                    values,
                    75
                )
            )
    }


# ============================================================
# CONSISTENCY
# ============================================================

def calculate_consistency(
    statistics
):

    median = statistics[
        "median_m"
    ]

    std = statistics[
        "std_m"
    ]

    if median <= 0:

        return 0.0

    variation = (
        std /
        median
    )

    consistency = (
        1.0 -
        min(
            variation,
            1.0
        )
    ) * 100.0

    return round(
        max(
            0.0,
            min(
                100.0,
                consistency
            )
        ),
        1
    )


# ============================================================
# CALIBRATION UNCERTAINTY
# ============================================================

def estimate_calibration_uncertainty(
    hfov
):

    assumed_error = 5.0

    return {

        "assumed_error_deg":
            assumed_error,

        "lower_fov_deg":
            max(
                MIN_HFOV,
                hfov -
                assumed_error
            ),

        "upper_fov_deg":
            min(
                MAX_HFOV,
                hfov +
                assumed_error
            )
    }


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    consistency,
    boundary_quality,
    depth_quality,
    hfov
):

    calibration_score = 60.0

    if 55 <= hfov <= 90:

        calibration_score += 5.0

    calibration_score = min(
        calibration_score,
        70.0
    )

    overall = (

        consistency *
        0.35

        +

        boundary_quality *
        0.20

        +

        depth_quality *
        0.20

        +

        calibration_score *
        0.25
    )

    overall = max(
        0.0,
        min(
            95.0,
            overall
        )
    )

    if overall >= 85:

        category = "HIGH"

    elif overall >= 70:

        category = "MEDIUM"

    elif overall >= 50:

        category = "LOW"

    else:

        category = "VERY LOW"

    return (
        round(
            overall,
            1
        ),
        category,
        round(
            calibration_score,
            1
        )
    )


# ============================================================
# WIDTH RANGE
# ============================================================

def calculate_width_range(
    statistics,
    hfov
):

    median = statistics[
        "median_m"
    ]

    std = statistics[
        "std_m"
    ]

    calibration_error = (
        median *
        0.12
    )

    statistical_error = (
        std *
        1.5
    )

    total_error = math.sqrt(
        calibration_error ** 2
        +
        statistical_error ** 2
    )

    lower = max(
        0.5,
        median -
        total_error
    )

    upper = (
        median +
        total_error
    )

    return (
        round(
            lower,
            2
        ),
        round(
            upper,
            2
        )
    )


# ============================================================
# WARNING GENERATOR
# ============================================================

def generate_warnings(
    consistency,
    boundary_quality,
    depth_quality,
    confidence,
    road_coverage,
    hfov
):

    warnings = []

    warnings.append(
        "Camera calibration is approximate. "
        "Survey-grade measurement requires calibrated "
        "camera parameters or a known reference."
    )

    if road_coverage < 5:

        warnings.append(
            "Very little road area was detected."
        )

    elif road_coverage > 80:

        warnings.append(
            "Road segmentation covers most of the image."
        )

    if boundary_quality < 50:

        warnings.append(
            "Road boundaries are unstable."
        )

    if depth_quality < 50:

        warnings.append(
            "Metric depth quality is limited."
        )

    if consistency < 60:

        warnings.append(
            "Cross-section width estimates vary significantly."
        )

    if confidence < 50:

        warnings.append(
            "Overall confidence is low."
        )

    elif confidence < 70:

        warnings.append(
            "Overall confidence is moderate-to-low."
        )

    return warnings


# ============================================================
# SEGMENTATION VISUALIZATION
# ============================================================

def create_segmentation_overlay(
    image,
    mask
):

    output = image.copy()

    road_pixels = (
        mask > 0
    )

    overlay = np.zeros_like(
        image
    )

    overlay[
        road_pixels
    ] = (
        0,
        180,
        0
    )

    return cv2.addWeighted(
        output,
        0.65,
        overlay,
        0.35,
        0
    )


# ============================================================
# BOUNDARY VISUALIZATION
# ============================================================

def create_boundary_overlay(
    image,
    left_line,
    right_line,
    measurements=None
):

    output = image.copy()

    height, width = (
        image.shape[:2]
    )

    ys = np.arange(
        int(height * 0.40),
        int(height * 0.97)
    )

    left_points = []

    right_points = []

    for y in ys:

        left_x = int(
            np.clip(
                np.polyval(
                    left_line,
                    y
                ),
                0,
                width - 1
            )
        )

        right_x = int(
            np.clip(
                np.polyval(
                    right_line,
                    y
                ),
                0,
                width - 1
            )
        )

        left_points.append(
            (
                left_x,
                int(y)
            )
        )

        right_points.append(
            (
                right_x,
                int(y)
            )
        )

    left_points = np.array(
        left_points
    )

    right_points = np.array(
        right_points
    )

    cv2.polylines(
        output,
        [left_points],
        False,
        (255, 80, 80),
        4
    )

    cv2.polylines(
        output,
        [right_points],
        False,
        (80, 80, 255),
        4
    )

    if measurements:

        for measurement in measurements:

            y = int(
                measurement["y"]
            )

            left_x = int(
                measurement["left_x"]
            )

            right_x = int(
                measurement["right_x"]
            )

            cv2.line(
                output,
                (
                    left_x,
                    y
                ),
                (
                    right_x,
                    y
                ),
                (0, 255, 255),
                2
            )

            cv2.circle(
                output,
                (
                    left_x,
                    y
                ),
                5,
                (255, 80, 80),
                -1
            )

            cv2.circle(
                output,
                (
                    right_x,
                    y
                ),
                5,
                (80, 80, 255),
                -1
            )

    return output


# ============================================================
# DEPTH VISUALIZATION
# ============================================================

def normalize_depth_for_visualization(
    depth
):

    valid = depth[
        np.isfinite(depth)
        &
        (depth > 0)
    ]

    if len(valid) == 0:

        return np.zeros_like(
            depth,
            dtype=np.uint8
        )

    low = np.percentile(
        valid,
        2
    )

    high = np.percentile(
        valid,
        98
    )

    if high <= low:

        high = low + 1.0

    normalized = (
        depth -
        low
    ) / (
        high -
        low
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    return (
        normalized *
        255
    ).astype(
        np.uint8
    )


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def make_json_safe(value):

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            make_json_safe(item)
            for item in value
        )

    return value


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def run_analysis(
    image,
    hfov
):

    start_time = time.time()

    # ========================================================
    # STEP 1
    # ========================================================

    validate_image(
        image
    )

    original_height, original_width = (
        image.shape[:2]
    )

    image = prepare_image(
        image
    )

    height, width = (
        image.shape[:2]
    )

    # ========================================================
    # STEP 2 — ROAD SEGMENTATION
    # ========================================================

    print(
        "STEP 1/8: Road segmentation"
    )

    raw_mask = get_road_mask(
        image
    )

    road_mask = clean_road_mask(
        raw_mask
    )

    road_mask = get_main_road_component(
        road_mask
    )

    road_coverage = (
        calculate_road_coverage(
            road_mask
        )
    )

    # ========================================================
    # STEP 3 — BOUNDARIES
    # ========================================================

    print(
        "STEP 2/8: Boundary extraction"
    )

    left_points, right_points = (
        extract_raw_boundaries(
            image,
            road_mask
        )
    )

    # ========================================================
    # STEP 4 — ROBUST FITTING
    # ========================================================

    print(
        "STEP 3/8: Robust boundary fitting"
    )

    left_line = robust_line_fit(
        left_points
    )

    right_line = robust_line_fit(
        right_points
    )

    left_quality = (
        calculate_boundary_quality(
            left_points,
            left_line
        )
    )

    right_quality = (
        calculate_boundary_quality(
            right_points,
            right_line
        )
    )

    boundary_quality = (
        left_quality[
            "quality_percent"
        ]
        +
        right_quality[
            "quality_percent"
        ]
    ) / 2.0

    # ========================================================
    # STEP 5 — VANISHING POINT
    # ========================================================

    print(
        "STEP 4/8: Vanishing point estimation"
    )

    vanishing_x, vanishing_y = (
        estimate_vanishing_point(
            left_line,
            right_line,
            image.shape
        )
    )

    # ========================================================
    # STEP 6 — DEPTH
    # ========================================================

    print(
        "STEP 5/8: Metric depth estimation"
    )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    depth = get_metric_depth(
        image_rgb
    )

    depth_quality = (
        calculate_depth_quality(
            depth
        )
    )

    # ========================================================
    # STEP 7 — CAMERA
    # ========================================================

    print(
        "STEP 6/8: Camera geometry"
    )

    intrinsics = (
        calculate_camera_intrinsics(
            width,
            height,
            hfov
        )
    )

    # ========================================================
    # STEP 8 — 3D WIDTH
    # ========================================================

    print(
        "STEP 7/8: 3D cross-section measurement"
    )

    measurements = (
        calculate_3d_widths(
            left_line,
            right_line,
            depth,
            intrinsics,
            image.shape
        )
    )

    print(
        "STEP 8/8: Robust statistics"
    )

    filtered_measurements = (
        filter_width_outliers(
            measurements
        )
    )

    statistics = (
        calculate_width_statistics(
            filtered_measurements
        )
    )

    consistency = (
        calculate_consistency(
            statistics
        )
    )

    (
        overall_confidence,
        confidence_category,
        calibration_confidence
    ) = calculate_confidence(
        consistency,
        boundary_quality,
        depth_quality,
        hfov
    )

    range_lower, range_upper = (
        calculate_width_range(
            statistics,
            hfov
        )
    )

    calibration = (
        estimate_calibration_uncertainty(
            hfov
        )
    )

    warnings = generate_warnings(
        consistency,
        boundary_quality,
        depth_quality,
        overall_confidence,
        road_coverage,
        hfov
    )

    processing_time = (
        time.time() -
        start_time
    )

    # ========================================================
    # RESULT
    # ========================================================

    return make_json_safe({

        "success": True,

        "pipeline_version":
            APP_VERSION,

        # ----------------------------------------------------
        # WIDTH
        # ----------------------------------------------------

        "estimated_width_m":
            round(
                statistics["median_m"],
                2
            ),

        "mean_width_m":
            round(
                statistics["mean_m"],
                2
            ),

        "median_width_m":
            round(
                statistics["median_m"],
                2
            ),

        "min_width_m":
            round(
                statistics["min_m"],
                2
            ),

        "max_width_m":
            round(
                statistics["max_m"],
                2
            ),

        "range_lower_m":
            range_lower,

        "range_upper_m":
            range_upper,

        "variation_m":
            round(
                statistics["std_m"],
                2
            ),

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        "consistency_percent":
            consistency,

        "boundary_quality_percent":
            round(
                boundary_quality,
                1
            ),

        "depth_quality_percent":
            depth_quality,

        "calibration_confidence":
            calibration_confidence,

        "overall_confidence":
            overall_confidence,

        "confidence_category":
            confidence_category,

        # ----------------------------------------------------
        # ROAD
        # ----------------------------------------------------

        "road_coverage_percent":
            round(
                road_coverage,
                2
            ),

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        "camera_fov_used":
            round(
                hfov,
                2
            ),

        "camera_fov_uncertainty_deg":
            calibration[
                "assumed_error_deg"
            ],

        "camera_intrinsics": {

            "fx":
                round(
                    intrinsics["fx"],
                    2
                ),

            "fy":
                round(
                    intrinsics["fy"],
                    2
                ),

            "cx":
                round(
                    intrinsics["cx"],
                    2
                ),

            "cy":
                round(
                    intrinsics["cy"],
                    2
                )
        },

        # ----------------------------------------------------
        # VANISHING POINT
        # ----------------------------------------------------

        "vanishing_point": {

            "x":
                round(
                    vanishing_x,
                    1
                ),

            "y":
                round(
                    vanishing_y,
                    1
                )
        },

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        "image_width_px":
            width,

        "image_height_px":
            height,

        "original_image_width_px":
            original_width,

        "original_image_height_px":
            original_height,

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------

        "depth_min_m":
            round(
                float(
                    np.min(depth)
                ),
                2
            ),

        "depth_max_m":
            round(
                float(
                    np.max(depth)
                ),
                2
            ),

        "depth_median_m":
            round(
                float(
                    np.median(depth)
                ),
                2
            ),

        # ----------------------------------------------------
        # CROSS SECTIONS
        # ----------------------------------------------------

        "cross_sections":
            [
                item["width_m"]
                for item
                in filtered_measurements
            ],

        "cross_section_details":
            filtered_measurements,

        "num_cross_sections":
            len(
                filtered_measurements
            ),

        # ----------------------------------------------------
        # BOUNDARIES
        # ----------------------------------------------------

        "boundary_left_quality":
            left_quality,

        "boundary_right_quality":
            right_quality,

        # ----------------------------------------------------
        # WARNINGS
        # ----------------------------------------------------

        "warnings":
            warnings,

        "warning":
            warnings[0]
            if warnings
            else
            "No major warnings.",

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        "processing_time_seconds":
            round(
                processing_time,
                2
            )
    })


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_road(
    file: UploadFile = File(...),
    hfov: float = Form(DEFAULT_HFOV)
):

    temporary_file = None

    request_start = time.time()

    try:

        # ----------------------------------------------------
        # FILENAME
        # ----------------------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="No filename supplied."
            )

        # ----------------------------------------------------
        # FILE TYPE
        # ----------------------------------------------------

        allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp"
        }

        extension = os.path.splitext(
            file.filename
        )[1].lower()

        if extension not in allowed_extensions:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported image format. "
                    "Use JPG, JPEG, PNG, WEBP or BMP."
                )
            )

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # ----------------------------------------------------
        # TEMP FILE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temporary:

            temporary.write(
                file_bytes
            )

            temporary_file = (
                temporary.name
            )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            temporary_file
        )

        if image is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded file could not "
                    "be decoded as an image."
                )
            )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validate_image(
            image
        )

        # ----------------------------------------------------
        # FOV
        # ----------------------------------------------------

        hfov = validate_hfov(
            hfov
        )

        print()
        print("=" * 60)
        print("NEW ROAD ANALYSIS REQUEST")
        print("=" * 60)

        print(
            f"File: {file.filename}"
        )

        print(
            f"Image shape: {image.shape}"
        )

        print(
            f"HFOV: {hfov:.2f} degrees"
        )

        print("=" * 60)

        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        result = run_analysis(
            image,
            hfov
        )

        result[
            "request_time_seconds"
        ] = round(
            time.time()
            -
            request_start,
            2
        )

        return result

    except HTTPException:

        raise

    except ValueError as error:

        print(
            "Validation error:",
            str(error)
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("ANALYSIS ERROR")
        print("=" * 60)

        print(
            str(error)
        )

        traceback.print_exc()

        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail=(
                "Road analysis failed: "
                +
                str(error)
            )
        )

    finally:

        if (
            temporary_file
            and
            os.path.exists(
                temporary_file
            )
        ):

            try:

                os.remove(
                    temporary_file
                )

            except Exception:

                pass


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/models")
def models():

    return {

        "road_segmentation": {

            "loaded":
                road_model is not None,

            "model":
                ROAD_MODEL_PATH,

            "type":
                "semantic segmentation",

            "road_class":
                0
        },

        "depth": {

            "loaded":
                depth_model is not None,

            "model":
                DEPTH_MODEL_NAME,

            "type":
                "metric depth estimation"
        }
    }


# ============================================================
# SYSTEM INFORMATION
# ============================================================

@app.get("/system")
def system():

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "pipeline": [

            "image validation",

            "road segmentation",

            "mask cleaning",

            "road component detection",

            "adaptive boundary extraction",

            "robust boundary fitting",

            "vanishing point estimation",

            "metric depth estimation",

            "camera geometry",

            "3D cross-section measurement",

            "outlier rejection",

            "width statistics",

            "uncertainty estimation",

            "confidence scoring"
        ],

        "limitations": [

            "Approximate camera calibration",

            "Single-image metric uncertainty",

            "Segmentation errors",

            "Depth estimation errors",

            "Occlusion and unusual road geometry"
        ]
    }


# ============================================================
# END
# ============================================================

# ============================================================
# ROAD INFORMATION
# ============================================================

@app.get("/road-info/{highway_name}")
def road_info(highway_name: str):

    info = get_highway_info(highway_name)

    if info is None:

        return {
            "found": False,
            "message": (
                "Highway not found. "
                "Available highways: "
                "NH 44, NH 48, NH 16, NH 27, "
                "NH 19, NH 66, NH 65"
            )
        }

    return {
        "found": True,
        "highway": highway_name.upper(),
        "data": info
    }