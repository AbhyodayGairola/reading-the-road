import cv2
import numpy as np

# ==========================================
# READ THE ROAD
# PERSPECTIVE-CONSTRAINED BOUNDARIES
# ==========================================

IMAGE_PATH = "test_road.jpg"
MASK_PATH = "road_mask.png"

image = cv2.imread(IMAGE_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("test_road.jpg not found")

if mask is None:
    raise FileNotFoundError("road_mask.png not found")

height, width = image.shape[:2]

# ==========================================
# VANISHING POINT
# ==========================================
#
# For this development image, the road
# converges around this location.
#
# Later we will make this automatic.

VP_X = int(width * 0.50)
VP_Y = int(height * 0.72)

# ==========================================
# CLEAN ROAD MASK
# ==========================================

kernel = np.ones((9, 9), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ==========================================
# FIND BOUNDARIES USING PERSPECTIVE
# ==========================================

left_points = []
right_points = []

# Start below the vanishing point
start_y = int(height * 0.74)
end_y = int(height * 0.98)

for y in range(start_y, end_y, 5):

    row = mask[y]

    road_pixels = np.where(row > 0)[0]

    if len(road_pixels) < 20:
        continue

    # --------------------------------------
    # Expected positions based on perspective
    # --------------------------------------

    # As we move down the image, the distance
    # from the vanishing point should increase.

    scale = (y - VP_Y) / (end_y - VP_Y)

    # Expected left/right search regions.
    #
    # These are deliberately broad so that
    # the segmentation can determine the edge.

    expected_left = VP_X - int(
        scale * width * 0.50
    )

    expected_right = VP_X + int(
        scale * width * 0.50
    )

    # ======================================
    # LEFT EDGE
    # ======================================

    left_candidates = road_pixels[
        road_pixels < VP_X
    ]

    if len(left_candidates) > 0:

        # Choose the road pixel closest to
        # the expected perspective position.

        left_x = left_candidates[
            np.argmin(
                np.abs(
                    left_candidates -
                    expected_left
                )
            )
        ]

        left_points.append(
            (int(left_x), y)
        )

    # ======================================
    # RIGHT EDGE
    # ======================================

    right_candidates = road_pixels[
        road_pixels > VP_X
    ]

    if len(right_candidates) > 0:

        right_x = right_candidates[
            np.argmin(
                np.abs(
                    right_candidates -
                    expected_right
                )
            )
        ]

        right_points.append(
            (int(right_x), y)
        )

# ==========================================
# ROBUST LINE FITTING
# ==========================================

def fit_line(points):

    if len(points) < 3:
        return None

    points = np.array(
        points,
        dtype=np.float32
    )

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).flatten()

    if abs(vy) < 1e-6:
        return None

    return vx, vy, x0, y0


left_line = fit_line(left_points)
right_line = fit_line(right_points)

# ==========================================
# FORCE BOTH LINES THROUGH VP
# ==========================================

def line_from_vp(points, vp_x, vp_y):

    if len(points) < 3:
        return None

    points = np.array(
        points,
        dtype=np.float32
    )

    # Calculate direction from VP to points
    directions = points - np.array(
        [vp_x, vp_y],
        dtype=np.float32
    )

    # Average direction
    direction = np.mean(
        directions,
        axis=0
    )

    dx = direction[0]
    dy = direction[1]

    if abs(dy) < 1e-6:
        return None

    return dx, dy


left_direction = line_from_vp(
    left_points,
    VP_X,
    VP_Y
)

right_direction = line_from_vp(
    right_points,
    VP_X,
    VP_Y
)

# ==========================================
# DRAW RESULT
# ==========================================

output = image.copy()

# Draw vanishing point

cv2.circle(
    output,
    (VP_X, VP_Y),
    9,
    (0, 255, 255),
    -1
)

cv2.putText(
    output,
    "VANISHING POINT",
    (VP_X + 12, VP_Y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 255),
    2
)

# Draw detected points

for x, y in left_points:

    cv2.circle(
        output,
        (x, y),
        2,
        (255, 0, 0),
        -1
    )

for x, y in right_points:

    cv2.circle(
        output,
        (x, y),
        2,
        (0, 0, 255),
        -1
    )

# ==========================================
# DRAW LEFT BOUNDARY
# ==========================================

if left_direction is not None:

    dx, dy = left_direction

    scale = (
        (height - 1 - VP_Y) / dy
    )

    bottom_x = int(
        VP_X + dx * scale
    )

    cv2.line(
        output,
        (VP_X, VP_Y),
        (bottom_x, height - 1),
        (255, 0, 0),
        5
    )

# ==========================================
# DRAW RIGHT BOUNDARY
# ==========================================

if right_direction is not None:

    dx, dy = right_direction

    scale = (
        (height - 1 - VP_Y) / dy
    )

    bottom_x = int(
        VP_X + dx * scale
    )

    cv2.line(
        output,
        (VP_X, VP_Y),
        (bottom_x, height - 1),
        (0, 0, 255),
        5
    )

# ==========================================
# LABELS
# ==========================================

cv2.putText(
    output,
    "PERSPECTIVE-CONSTRAINED BOUNDARIES",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    "BLUE = LEFT",
    (20, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    2
)

cv2.putText(
    output,
    "RED = RIGHT",
    (20, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 255),
    2
)

# ==========================================
# SAVE
# ==========================================

cv2.imwrite(
    "perspective_boundaries.jpg",
    output
)

print()
print("========================================")
print("PERSPECTIVE BOUNDARIES")
print("========================================")

print(
    f"Vanishing point: ({VP_X}, {VP_Y})"
)

print(
    f"Left points: {len(left_points)}"
)

print(
    f"Right points: {len(right_points)}"
)

print(
    f"Left boundary: "
    f"{left_direction is not None}"
)

print(
    f"Right boundary: "
    f"{right_direction is not None}"
)

print()
print("Saved:")
print("  perspective_boundaries.jpg")