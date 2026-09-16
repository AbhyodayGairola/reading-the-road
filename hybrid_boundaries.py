import cv2
import numpy as np

# ==========================================
# READ THE ROAD
# HYBRID ROAD BOUNDARY DETECTION
# ==========================================

IMAGE_PATH = "test_road.jpg"
MASK_PATH = "road_mask.png"

image = cv2.imread(IMAGE_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not find test_road.jpg")

if mask is None:
    raise FileNotFoundError("Could not find road_mask.png")

height, width = image.shape[:2]

# ==========================================
# 1. CLEAN ROAD MASK
# ==========================================

kernel = np.ones((7, 7), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ==========================================
# 2. EDGE DETECTION
# ==========================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

edges = cv2.Canny(
    blur,
    50,
    150
)

# ==========================================
# 3. ONLY CONSIDER LOWER ROAD REGION
# ==========================================

roi_mask = np.zeros_like(edges)

# Ignore sky/horizon.
roi_top = int(height * 0.45)

roi_mask[roi_top:height, :] = 255

edges = cv2.bitwise_and(
    edges,
    roi_mask
)

# ==========================================
# 4. HIGHLIGHT EDGES THAT ARE NEAR ROAD MASK
# ==========================================

# Dilate road mask slightly so nearby physical
# road boundaries can be associated with it.

expanded_road = cv2.dilate(
    mask,
    np.ones((15, 15), np.uint8),
    iterations=1
)

road_edges = cv2.bitwise_and(
    edges,
    expanded_road
)

# ==========================================
# 5. HOUGH LINE DETECTION
# ==========================================

lines = cv2.HoughLinesP(
    road_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=20,
    minLineLength=25,
    maxLineGap=50
)

left_lines = []
right_lines = []

if lines is not None:

    center = width / 2
    bottom_y = height - 1

    for line in lines:

        x1, y1, x2, y2 = line.flatten()

        dx = x2 - x1
        dy = y2 - y1

        # Ignore very short or nearly horizontal lines
        if abs(dx) < 3:
            continue

        if abs(dy) < 10:
            continue

        slope = dy / dx

        length = np.sqrt(
            dx * dx + dy * dy
        )

        if length < 25:
            continue

        # Ignore weakly angled lines
        if abs(slope) < 0.25:
            continue

        # Avoid division by zero
        if abs(y2 - y1) < 1:
            continue

        # --------------------------------------
        # Project line toward bottom of image
        # --------------------------------------

        x_bottom = x1 + (
            (bottom_y - y1) / (y2 - y1)
        ) * (x2 - x1)

        # --------------------------------------
        # LEFT ROAD EDGE
        # --------------------------------------

        if x_bottom < center and slope < 0:

            left_lines.append(
                (x1, y1, x2, y2, length)
            )

        # --------------------------------------
        # RIGHT ROAD EDGE
        # --------------------------------------

        elif x_bottom > center and slope > 0:

            right_lines.append(
                (x1, y1, x2, y2, length)
            )
# ==========================================
# 6. SELECT STRONGEST LINES
# ==========================================

left_lines.sort(
    key=lambda x: x[4],
    reverse=True
)

right_lines.sort(
    key=lambda x: x[4],
    reverse=True
)

# Keep strongest candidates
left_lines = left_lines[:20]
right_lines = right_lines[:20]

# ==========================================
# 7. DRAW ALL CANDIDATE LINES
# ==========================================

output = image.copy()

for x1, y1, x2, y2, length in left_lines:

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (255, 180, 0),
        2
    )

for x1, y1, x2, y2, length in right_lines:

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (0, 180, 255),
        2
    )

# ==========================================
# 8. FIT A SINGLE LEFT BOUNDARY
# ==========================================

def fit_boundary(lines):

    if len(lines) < 2:
        return None

    points = []

    for x1, y1, x2, y2, length in lines:

        points.append((x1, y1))
        points.append((x2, y2))

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

    y1 = roi_top
    y2 = height - 1

    x1 = int(
        x0 + (y1 - y0) * vx / vy
    )

    x2 = int(
        x0 + (y2 - y0) * vx / vy
    )

    return x1, y1, x2, y2


left_boundary = fit_boundary(left_lines)
right_boundary = fit_boundary(right_lines)

# ==========================================
# 9. DRAW FINAL BOUNDARIES
# ==========================================

if left_boundary is not None:

    x1, y1, x2, y2 = left_boundary

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        6
    )

if right_boundary is not None:

    x1, y1, x2, y2 = right_boundary

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        6
    )

# ==========================================
# 10. LABELS
# ==========================================

cv2.putText(
    output,
    "HYBRID BOUNDARIES",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    "BLUE = LEFT",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 0, 0),
    2
)

cv2.putText(
    output,
    "RED = RIGHT",
    (20, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)

# ==========================================
# 11. SAVE
# ==========================================

cv2.imwrite(
    "hybrid_boundaries.jpg",
    output
)

cv2.imwrite(
    "road_edges.jpg",
    road_edges
)

print()
print("========================================")
print("HYBRID BOUNDARY DETECTION COMPLETE")
print("========================================")

print(
    f"Left candidate lines: {len(left_lines)}"
)

print(
    f"Right candidate lines: {len(right_lines)}"
)

print(
    f"Left boundary fitted: {left_boundary is not None}"
)

print(
    f"Right boundary fitted: {right_boundary is not None}"
)

print()
print("Saved:")
print("  hybrid_boundaries.jpg")
print("  road_edges.jpg")