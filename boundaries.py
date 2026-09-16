import cv2
import numpy as np

# ==========================================
# READ THE ROAD - IMPROVED BOUNDARY DETECTION
# ==========================================

IMAGE_PATH = "test_road.jpg"
MASK_PATH = "road_mask.png"

image = cv2.imread(IMAGE_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not find test_road.jpg")

if mask is None:
    raise FileNotFoundError("Could not find road_mask.png")

height, width = mask.shape

# ==========================================
# STEP 1 - CLEAN MASK
# ==========================================

kernel = np.ones((9, 9), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ==========================================
# STEP 2 - USE LOWER ROAD REGION
# ==========================================

# Ignore the distant horizon.
# The lower part gives us stronger geometric evidence.

roi_top = int(height * 0.50)
roi_bottom = int(height * 0.95)

roi = np.zeros_like(mask)

roi[roi_top:roi_bottom, :] = mask[
    roi_top:roi_bottom, :
]

# ==========================================
# STEP 3 - FIND ROAD EDGE CANDIDATES
# ==========================================

left_points = []
right_points = []

for y in range(roi_top, roi_bottom, 5):

    row = roi[y]

    # Find all road pixels
    xs = np.where(row > 0)[0]

    if len(xs) < 50:
        continue

    # --------------------------------------
    # Instead of blindly using the first
    # pixel, look for transitions.
    # --------------------------------------

    transitions = []

    previous = False

    for x in range(width):

        current = row[x] > 0

        if current != previous:
            transitions.append(x)

        previous = current

    # Need at least two transitions
    if len(transitions) < 2:
        continue

    # ======================================
    # FIND ROAD REGION CLOSE TO IMAGE CENTER
    # ======================================

    center = width / 2

    candidate_pairs = []

    for i in range(len(transitions) - 1):

        x1 = transitions[i]
        x2 = transitions[i + 1]

        if x2 - x1 < 30:
            continue

        midpoint = (x1 + x2) / 2

        distance_from_center = abs(
            midpoint - center
        )

        candidate_pairs.append(
            (
                distance_from_center,
                x1,
                x2
            )
        )

    if not candidate_pairs:
        continue

    # Select candidate closest to image center
    candidate_pairs.sort(
        key=lambda item: item[0]
    )

    _, left_x, right_x = candidate_pairs[0]

    left_points.append(
        (left_x, y)
    )

    right_points.append(
        (right_x, y)
    )

# ==========================================
# STEP 4 - REMOVE OUTLIERS
# ==========================================

def remove_outliers(points):

    if len(points) < 4:
        return points

    points = np.array(points)

    xs = points[:, 0]

    median = np.median(xs)

    deviation = np.abs(xs - median)

    threshold = np.median(deviation) * 3 + 20

    filtered = points[
        deviation < threshold
    ]

    return [
        tuple(point.astype(int))
        for point in filtered
    ]


left_points = remove_outliers(left_points)
right_points = remove_outliers(right_points)

# ==========================================
# STEP 5 - DRAW RESULT
# ==========================================

output = image.copy()

# Draw detected points

for x, y in left_points:

    cv2.circle(
        output,
        (x, y),
        3,
        (255, 0, 0),
        -1
    )

for x, y in right_points:

    cv2.circle(
        output,
        (x, y),
        3,
        (0, 0, 255),
        -1
    )

# ==========================================
# STEP 6 - FIT LEFT LINE
# ==========================================

if len(left_points) >= 3:

    points = np.array(
        left_points,
        dtype=np.float32
    )

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).flatten()

    if abs(vy) > 1e-6:

        y1 = roi_top
        y2 = roi_bottom

        x1 = int(
            x0 + (y1 - y0) * vx / vy
        )

        x2 = int(
            x0 + (y2 - y0) * vx / vy
        )

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            5
        )

# ==========================================
# STEP 7 - FIT RIGHT LINE
# ==========================================

if len(right_points) >= 3:

    points = np.array(
        right_points,
        dtype=np.float32
    )

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).flatten()

    if abs(vy) > 1e-6:

        y1 = roi_top
        y2 = roi_bottom

        x1 = int(
            x0 + (y1 - y0) * vx / vy
        )

        x2 = int(
            x0 + (y2 - y0) * vx / vy
        )

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            5
        )

# ==========================================
# LABELS
# ==========================================

cv2.putText(
    output,
    "LEFT ROAD EDGE",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)

cv2.putText(
    output,
    "RIGHT ROAD EDGE",
    (20, 75),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

# ==========================================
# SAVE
# ==========================================

cv2.imwrite(
    "road_boundaries.jpg",
    output
)

print()
print("================================")
print("IMPROVED BOUNDARY DETECTION")
print("================================")

print(
    f"Left boundary points: {len(left_points)}"
)

print(
    f"Right boundary points: {len(right_points)}"
)

print()
print("Saved:")
print("  road_boundaries.jpg")