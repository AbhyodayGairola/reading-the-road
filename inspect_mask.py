import cv2
import numpy as np

IMAGE_PATH = "test_road.jpg"
MASK_PATH = "road_mask.png"

image = cv2.imread(IMAGE_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("test_road.jpg not found")

if mask is None:
    raise FileNotFoundError("road_mask.png not found")

height, width = mask.shape

# ==========================================
# SHOW ROAD MASK WITH CONTOURS
# ==========================================

output = image.copy()

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask,
    connectivity=8
)

components = []

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area < 500:
        continue

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    components.append(
        (area, x, y, w, h, i)
    )

# Sort largest first
components.sort(reverse=True)

print()
print("========================================")
print("ROAD MASK COMPONENTS")
print("========================================")

for number, component in enumerate(components[:10], 1):

    area, x, y, w, h, label = component

    print(
        f"{number}: "
        f"area={area}, "
        f"x={x}, y={y}, "
        f"w={w}, h={h}"
    )

    # Give each large component a contour
    component_mask = np.uint8(
        labels == label
    ) * 255

    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 255),
        2
    )

# ==========================================
# DRAW IMAGE CENTER
# ==========================================

cv2.line(
    output,
    (width // 2, 0),
    (width // 2, height),
    (255, 0, 255),
    2
)

cv2.putText(
    output,
    "IMAGE CENTER",
    (width // 2 + 5, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 255),
    2
)

# ==========================================
# SAVE
# ==========================================

cv2.imwrite(
    "mask_components.jpg",
    output
)

print()
print("Saved:")
print("  mask_components.jpg")