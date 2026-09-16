import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# ============================================================
# SETTINGS
# ============================================================

IMAGE_FILE = "test_road.jpg"
MASK_FILE = "road_mask.png"

# Approximate horizontal field of view.
# We will test several values later.
HFOV_DEG = 70.0

# Same vanishing-point assumption used by our boundary system
VP_X_RATIO = 0.50
VP_Y_RATIO = 0.72

# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_FILE)

if image is None:
    raise FileNotFoundError(IMAGE_FILE)

height, width = image.shape[:2]

# ============================================================
# LOAD ROAD MASK
# ============================================================

mask = cv2.imread(MASK_FILE, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(MASK_FILE)

mask = cv2.resize(mask, (width, height))

road = mask > 100

# ============================================================
# FIND BOTTOM ROAD EDGES
# ============================================================

bottom_y = height - 5

xs = np.where(road[bottom_y])[0]

if len(xs) < 20:

    for y in range(height - 5, int(height * 0.75), -5):

        xs = np.where(road[y])[0]

        if len(xs) >= 20:
            bottom_y = y
            break

if len(xs) < 20:
    raise RuntimeError("Could not find road pixels.")

left_bottom = int(xs.min())
right_bottom = int(xs.max())

# ============================================================
# VANISHING POINT
# ============================================================

vp_x = width * VP_X_RATIO
vp_y = height * VP_Y_RATIO

# ============================================================
# CAMERA INTRINSICS
# ============================================================

cx = width / 2
cy = height / 2

hfov_rad = np.radians(HFOV_DEG)

fx = width / (2 * np.tan(hfov_rad / 2))
fy = fx

print("=" * 70)
print("3D ROAD WIDTH ESTIMATION")
print("=" * 70)

print(f"Image size       : {width} x {height}")
print(f"Assumed HFOV     : {HFOV_DEG:.1f} degrees")
print(f"Estimated fx     : {fx:.2f}")
print(f"Estimated fy     : {fy:.2f}")
print(f"Vanishing point  : ({vp_x:.1f}, {vp_y:.1f})")

# ============================================================
# LOAD METRIC DEPTH MODEL
# ============================================================

print("\nLoading metric depth model...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
)

# ============================================================
# RUN DEPTH
# ============================================================

print("Estimating metric depth...")

pil_image = Image.open(IMAGE_FILE).convert("RGB")

result = depth_pipe(pil_image)

depth = result["predicted_depth"].squeeze().cpu().numpy()

# Depth model resolution may differ from original image
depth = cv2.resize(
    depth,
    (width, height),
    interpolation=cv2.INTER_LINEAR
)

print(
    f"Depth range      : "
    f"{depth.min():.2f} - {depth.max():.2f} m"
)

# ============================================================
# BOUNDARY FUNCTION
# ============================================================

def boundary_x(y, bottom_x):

    return vp_x + (bottom_x - vp_x) * (
        (y - vp_y) / (bottom_y - vp_y)
    )

# ============================================================
# DEPTH SAMPLING
# ============================================================

def sample_depth(x, y):

    x = int(round(x))
    y = int(round(y))

    radius = 3

    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)

    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)

    patch = depth[y1:y2, x1:x2]

    return float(np.median(patch))


# ============================================================
# PIXEL -> 3D
# ============================================================

def pixel_to_3d(x, y, z):

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z

    return np.array([X, Y, Z])


# ============================================================
# CROSS-SECTIONS
# ============================================================

rows = np.linspace(
    int(height * 0.80),
    bottom_y,
    7
).astype(int)

measurements = []

print("\n")
print(
    f"{'Y':>5} "
    f"{'Depth L':>10} "
    f"{'Depth R':>10} "
    f"{'Width(m)':>12}"
)

print("-" * 55)

for y in rows:

    left_x = boundary_x(y, left_bottom)
    right_x = boundary_x(y, right_bottom)

    left_depth = sample_depth(left_x, y)
    right_depth = sample_depth(right_x, y)

    left_point = pixel_to_3d(
        left_x,
        y,
        left_depth
    )

    right_point = pixel_to_3d(
        right_x,
        y,
        right_depth
    )

    # Lateral road width.
    # X is the camera's horizontal coordinate.
    width_m = abs(
        right_point[0] - left_point[0]
    )

    measurements.append(width_m)

    print(
        f"{y:5d} "
        f"{left_depth:10.2f} "
        f"{right_depth:10.2f} "
        f"{width_m:12.2f}"
    )

# ============================================================
# ROBUST ESTIMATE
# ============================================================

measurements = np.array(measurements)

# Ignore the very nearest section because monocular depth
# and the road mask are generally least reliable there.
valid = measurements[:-1]

median_width = np.median(valid)
mean_width = np.mean(valid)
std_width = np.std(valid)

print("\n" + "=" * 70)

print(f"Median estimated width : {median_width:.2f} m")
print(f"Mean estimated width   : {mean_width:.2f} m")
print(f"Variation              : {std_width:.2f} m")

print("=" * 70)

# ============================================================
# FOV SENSITIVITY TEST
# ============================================================

print("\nFOV sensitivity test:")
print("-" * 45)

for fov in [60, 65, 70, 75, 80]:

    fov_rad = np.radians(fov)

    test_fx = width / (
        2 * np.tan(fov_rad / 2)
    )

    test_widths = []

    for y in rows[:-1]:

        left_x = boundary_x(y, left_bottom)
        right_x = boundary_x(y, right_bottom)

        left_depth = sample_depth(left_x, y)
        right_depth = sample_depth(right_x, y)

        left_X = (
            (left_x - cx) *
            left_depth /
            test_fx
        )

        right_X = (
            (right_x - cx) *
            right_depth /
            test_fx
        )

        test_widths.append(
            abs(right_X - left_X)
        )

    print(
        f"HFOV {fov:2d}° -> "
        f"{np.median(test_widths):.2f} m"
    )

# ============================================================
# VISUALIZATION
# ============================================================

output = image.copy()

# Vanishing point
cv2.circle(
    output,
    (int(vp_x), int(vp_y)),
    7,
    (0, 255, 255),
    -1
)

# Boundary lines
cv2.line(
    output,
    (int(vp_x), int(vp_y)),
    (left_bottom, bottom_y),
    (255, 0, 0),
    3
)

cv2.line(
    output,
    (int(vp_x), int(vp_y)),
    (right_bottom, bottom_y),
    (0, 0, 255),
    3
)

# Cross-sections
for y, width_m in zip(rows, measurements):

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
        f"{width_m:.2f} m",
        (int((left_x + right_x) / 2) - 35, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

# Final result
cv2.putText(
    output,
    f"Estimated width: {median_width:.2f} m",
    (15, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    f"Approx. HFOV: {HFOV_DEG:.0f} deg",
    (15, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)

cv2.imwrite(
    "road_3d_width.jpg",
    output
)

print("\nSaved visualization:")
print("road_3d_width.jpg")