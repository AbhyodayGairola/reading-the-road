import cv2
import numpy as np

# ==========================================
# READ THE ROAD
# PERSPECTIVE GEOMETRY
# ==========================================

IMAGE_PATH = "test_road.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        "Could not find test_road.jpg"
    )

height, width = image.shape[:2]

output = image.copy()

# ==========================================
# APPROXIMATE ROAD VANISHING REGION
# ==========================================

# For this first experiment, use the upper
# central region of the road.

vanishing_x = int(width * 0.40)
vanishing_y = int(height * 0.57)

# Draw vanishing point
cv2.circle(
    output,
    (vanishing_x, vanishing_y),
    10,
    (0, 255, 255),
    -1
)

cv2.putText(
    output,
    "VANISHING POINT",
    (vanishing_x + 15, vanishing_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 255),
    2
)

# ==========================================
# DRAW PERSPECTIVE RAYS
# ==========================================

# Candidate left road edge
left_bottom = (
    int(width * 0.31),
    height
)

# Candidate right road edge
right_bottom = (
    int(width * 0.90),
    height
)

cv2.line(
    output,
    (vanishing_x, vanishing_y),
    left_bottom,
    (255, 0, 0),
    3
)

cv2.line(
    output,
    (vanishing_x, vanishing_y),
    right_bottom,
    (0, 0, 255),
    3
)

# ==========================================
# CROSS-SECTIONS
# ==========================================

# Draw several horizontal measurement lines.

for fraction in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:

    y = int(height * fraction)

    cv2.line(
        output,
        (0, y),
        (width, y),
        (0, 255, 0),
        1
    )

    cv2.putText(
        output,
        f"Cross-section y={y}",
        (10, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1
    )

# ==========================================
# SAVE
# ==========================================

cv2.imwrite(
    "perspective_analysis.jpg",
    output
)

print()
print("========================================")
print("PERSPECTIVE ANALYSIS COMPLETE")
print("========================================")

print(f"Image: {width} x {height}")

print(
    f"Vanishing point: "
    f"({vanishing_x}, {vanishing_y})"
)

print()
print("Saved:")
print("  perspective_analysis.jpg")