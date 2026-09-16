import cv2
import numpy as np

# ==========================================
# READ THE ROAD
# CROSS-SECTION ANALYSIS
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
# CLEAN MASK
# ==========================================

kernel = np.ones((7, 7), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ==========================================
# CROSS-SECTION LOCATIONS
# ==========================================

# We avoid the extreme bottom because the
# camera/car may occupy that region.

rows = [
    int(height * 0.60),
    int(height * 0.65),
    int(height * 0.70),
    int(height * 0.75),
    int(height * 0.80),
    int(height * 0.85),
    int(height * 0.90),
]

output = image.copy()

measurements = []

# ==========================================
# ANALYZE EACH CROSS-SECTION
# ==========================================

for y in rows:

    row = mask[y]

    # Find pixels classified as road
    road_pixels = np.where(row > 0)[0]

    if len(road_pixels) < 20:

        measurements.append(
            (y, None, None, None)
        )

        continue

    # --------------------------------------
    # Find continuous road segments
    # --------------------------------------

    segments = []

    start = road_pixels[0]
    previous = road_pixels[0]

    for x in road_pixels[1:]:

        # A gap means a new segment
        if x > previous + 3:

            segments.append(
                (start, previous)
            )

            start = x

        previous = x

    segments.append(
        (start, previous)
    )

    # Remove tiny segments
    segments = [
        (x1, x2)
        for x1, x2 in segments
        if x2 - x1 >= 20
    ]

    if not segments:

        measurements.append(
            (y, None, None, None)
        )

        continue

    # ======================================
    # SELECT MAIN ROAD REGION
    # ======================================

    # At each cross-section, prefer the
    # widest substantial segment.

    segments.sort(
        key=lambda s: s[1] - s[0],
        reverse=True
    )

    left_x, right_x = segments[0]

    pixel_width = right_x - left_x

    measurements.append(
        (
            y,
            left_x,
            right_x,
            pixel_width
        )
    )

    # ======================================
    # DRAW CROSS-SECTION
    # ======================================

    cv2.line(
        output,
        (left_x, y),
        (right_x, y),
        (0, 255, 255),
        3
    )

    # Draw endpoints
    cv2.circle(
        output,
        (left_x, y),
        6,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        output,
        (right_x, y),
        6,
        (0, 0, 255),
        -1
    )

    # Label
    cv2.putText(
        output,
        f"{pixel_width}px",
        (
            max(5, left_x),
            max(20, y - 8)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

# ==========================================
# TITLE
# ==========================================

cv2.putText(
    output,
    "ROAD CROSS-SECTIONS",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    "YELLOW = APPARENT ROAD WIDTH",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2
)

# ==========================================
# SAVE
# ==========================================

cv2.imwrite(
    "cross_sections.jpg",
    output
)

# ==========================================
# PRINT MEASUREMENTS
# ==========================================

print()
print("========================================")
print("CROSS-SECTION ANALYSIS")
print("========================================")

print()
print(
    f"{'Row':>8} "
    f"{'Left':>8} "
    f"{'Right':>8} "
    f"{'Width(px)':>12}"
)

print("-" * 42)

valid_widths = []

for y, left, right, width_px in measurements:

    if width_px is None:

        print(
            f"{y:>8} "
            f"{'--':>8} "
            f"{'--':>8} "
            f"{'NO ROAD':>12}"
        )

    else:

        print(
            f"{y:>8} "
            f"{left:>8} "
            f"{right:>8} "
            f"{width_px:>12}"
        )

        valid_widths.append(width_px)

# ==========================================
# CONSISTENCY
# ==========================================

if len(valid_widths) >= 2:

    mean_width = np.mean(valid_widths)
    std_width = np.std(valid_widths)

    print()
    print(
        f"Mean apparent width: "
        f"{mean_width:.1f} px"
    )

    print(
        f"Width variation: "
        f"{std_width:.1f} px"
    )

print()
print("Saved:")
print("  cross_sections.jpg")