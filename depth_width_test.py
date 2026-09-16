import cv2
import numpy as np

IMAGE_FILE = "test_road.jpg"
DEPTH_FILE = "depth_map.png"
MASK_FILE = "road_mask.png"

# ============================================================
# LOAD
# ============================================================

image = cv2.imread(IMAGE_FILE)
depth = cv2.imread(DEPTH_FILE, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(MASK_FILE, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("test_road.jpg not found")

if depth is None:
    raise FileNotFoundError("depth_map.png not found")

if mask is None:
    raise FileNotFoundError("road_mask.png not found")

height, width = image.shape[:2]

depth = cv2.resize(depth, (width, height))
mask = cv2.resize(mask, (width, height))

# ============================================================
# VANISHING POINT
# ============================================================

vp_x = int(width * 0.50)
vp_y = int(height * 0.72)

# ============================================================
# ROAD EDGES
# ============================================================

road = mask > 100

bottom_y = height - 5

xs = np.where(road[bottom_y])[0]

if len(xs) < 20:
    for y in range(height - 5, int(height * 0.75), -5):
        xs = np.where(road[y])[0]

        if len(xs) >= 20:
            bottom_y = y
            break

if len(xs) < 20:
    raise RuntimeError("Could not detect road edges.")

left_bottom = int(xs.min())
right_bottom = int(xs.max())

# ============================================================
# BOUNDARY FUNCTION
# ============================================================

def boundary_x(y, bottom_x):

    return vp_x + (bottom_x - vp_x) * (
        (y - vp_y) / (bottom_y - vp_y)
    )

# ============================================================
# CROSS-SECTIONS
# ============================================================

rows = np.linspace(
    int(height * 0.78),
    bottom_y,
    8
).astype(int)

results = []

print("=" * 70)
print("DEPTH-NORMALIZED ROAD WIDTH TEST")
print("=" * 70)

print(f"Image          : {width} x {height}")
print(f"Vanishing point: ({vp_x}, {vp_y})")

print("\n")
print(
    f"{'Y':>5} "
    f"{'Width(px)':>12} "
    f"{'Depth':>12} "
    f"{'Width/Depth':>15}"
)

print("-" * 70)

for y in rows:

    left_x = boundary_x(y, left_bottom)
    right_x = boundary_x(y, right_bottom)

    pixel_width = right_x - left_x

    center_x = int((left_x + right_x) / 2)

    # Small depth patch around road center
    y1 = max(0, y - 3)
    y2 = min(height, y + 4)

    x1 = max(0, center_x - 3)
    x2 = min(width, center_x + 4)

    depth_patch = depth[y1:y2, x1:x2]

    depth_value = float(np.mean(depth_patch))

    if depth_value > 1:

        normalized_width = pixel_width / depth_value

        results.append(normalized_width)

    else:

        normalized_width = 0

    print(
        f"{y:5d} "
        f"{pixel_width:12.2f} "
        f"{depth_value:12.2f} "
        f"{normalized_width:15.3f}"
    )

# ============================================================
# CONSISTENCY
# ============================================================

results = np.array(results)

mean_normalized = np.mean(results)
std_normalized = np.std(results)

cv = std_normalized / mean_normalized

consistency = max(
    0,
    100 * (1 - cv)
)

print("\n" + "=" * 70)

print(
    f"Mean normalized width : "
    f"{mean_normalized:.3f}"
)

print(
    f"Variation              : "
    f"{std_normalized:.3f}"
)

print(
    f"Depth-normalized       : "
    f"{consistency:.1f}%"
)

print("=" * 70)

# ============================================================
# VISUALIZATION
# ============================================================

output = image.copy()

for y in rows:

    left_x = int(boundary_x(y, left_bottom))
    right_x = int(boundary_x(y, right_bottom))

    cv2.line(
        output,
        (left_x, y),
        (right_x, y),
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"{right_x-left_x}px",
        (right_x - 90, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

cv2.putText(
    output,
    f"Depth-normalized consistency: {consistency:.1f}%",
    (15, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2
)

cv2.imwrite(
    "depth_width_analysis.jpg",
    output
)

print("\nSaved:")
print("depth_width_analysis.jpg")