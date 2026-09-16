import cv2
import numpy as np

# ============================================================
# FILES
# ============================================================

IMAGE_FILE = "test_road.jpg"
DEPTH_FILE = "depth_map.png"

# ============================================================
# LOAD IMAGE + DEPTH
# ============================================================

image = cv2.imread(IMAGE_FILE)
depth = cv2.imread(DEPTH_FILE, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not find test_road.jpg")

if depth is None:
    raise FileNotFoundError("Could not find depth_map.png")

height, width = image.shape[:2]

# Resize depth to image size if necessary
depth = cv2.resize(depth, (width, height))

# ============================================================
# VANISHING POINT
# Same perspective assumption used previously
# ============================================================

vp_x = int(width * 0.50)
vp_y = int(height * 0.72)

# ============================================================
# ROAD EDGE POINTS
#
# These are the approximate bottom positions of the
# left and right road boundaries.
#
# We will automatically determine them from the road mask.
# ============================================================

road_mask = cv2.imread("road_mask.png", cv2.IMREAD_GRAYSCALE)

if road_mask is None:
    raise FileNotFoundError("Could not find road_mask.png")

road_mask = cv2.resize(road_mask, (width, height))

# Convert to binary
mask = road_mask > 100

# Search near the bottom of the image
bottom_y = height - 5

xs = np.where(mask[bottom_y])[0]

if len(xs) < 20:
    # Search slightly higher if the exact bottom row is unreliable
    for y in range(height - 5, int(height * 0.75), -5):
        xs = np.where(mask[y])[0]
        if len(xs) >= 20:
            bottom_y = y
            break

if len(xs) < 20:
    raise RuntimeError("Could not find enough road pixels.")

left_bottom_x = int(xs.min())
right_bottom_x = int(xs.max())

print("=" * 55)
print("ROAD WIDTH GEOMETRY ANALYSIS")
print("=" * 55)

print(f"Image size       : {width} x {height}")
print(f"Vanishing point  : ({vp_x}, {vp_y})")
print(f"Left road edge   : x={left_bottom_x}")
print(f"Right road edge  : x={right_bottom_x}")

# ============================================================
# INTERPOLATE ROAD EDGES
# Both edges are forced through the same vanishing point.
# ============================================================

def boundary_x(y, bottom_x):
    if y == vp_y:
        return vp_x

    return vp_x + (bottom_x - vp_x) * (
        (y - vp_y) / (bottom_y - vp_y)
    )

# ============================================================
# MEASURE CROSS-SECTIONS
# ============================================================

rows = np.linspace(
    int(height * 0.78),
    bottom_y,
    8
).astype(int)

measurements = []

print("\nCross-section measurements:")
print("-" * 55)

for y in rows:

    left_x = boundary_x(y, left_bottom_x)
    right_x = boundary_x(y, right_bottom_x)

    pixel_width = right_x - left_x

    midpoint_x = int((left_x + right_x) / 2)

    # Depth around the road center
    y1 = max(0, y - 3)
    y2 = min(height, y + 4)

    x1 = max(0, midpoint_x - 3)
    x2 = min(width, midpoint_x + 4)

    local_depth = depth[y1:y2, x1:x2]

    depth_value = float(np.mean(local_depth))

    measurements.append(
        (y, left_x, right_x, pixel_width, depth_value)
    )

    print(
        f"Y={y:3d} | "
        f"Left={left_x:7.1f} | "
        f"Right={right_x:7.1f} | "
        f"Width={pixel_width:7.1f}px | "
        f"Depth={depth_value:6.1f}"
    )

# ============================================================
# CONSISTENCY ANALYSIS
# ============================================================

pixel_widths = np.array([m[3] for m in measurements])

mean_width = np.mean(pixel_widths)
std_width = np.std(pixel_widths)

if mean_width > 0:
    consistency = max(
        0,
        100 * (1 - std_width / mean_width)
    )
else:
    consistency = 0

print("\n" + "=" * 55)
print(f"Mean pixel width : {mean_width:.2f}px")
print(f"Width variation  : {std_width:.2f}px")
print(f"Consistency      : {consistency:.1f}%")
print("=" * 55)

# ============================================================
# VISUALIZATION
# ============================================================

output = image.copy()

# Vanishing point
cv2.circle(
    output,
    (vp_x, vp_y),
    7,
    (0, 255, 255),
    -1
)

# Draw boundaries
cv2.line(
    output,
    (vp_x, vp_y),
    (left_bottom_x, bottom_y),
    (255, 0, 0),
    3
)

cv2.line(
    output,
    (vp_x, vp_y),
    (right_bottom_x, bottom_y),
    (0, 0, 255),
    3
)

# Draw cross sections
for y, left_x, right_x, pixel_width, depth_value in measurements:

    lx = int(left_x)
    rx = int(right_x)

    cv2.line(
        output,
        (lx, y),
        (rx, y),
        (0, 255, 0),
        2
    )

    cv2.circle(
        output,
        (lx, y),
        4,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        output,
        (rx, y),
        4,
        (0, 0, 255),
        -1
    )

# Information panel
cv2.putText(
    output,
    f"Mean width: {mean_width:.1f}px",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    f"Consistency: {consistency:.1f}%",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.imwrite(
    "width_geometry.jpg",
    output
)

print("\nSaved visualization:")
print("width_geometry.jpg")