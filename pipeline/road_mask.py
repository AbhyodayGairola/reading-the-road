from ultralytics import YOLO
import cv2
import numpy as np

# ==========================================
# READ THE ROAD - ROAD MASK
# ==========================================

IMAGE_PATH = "test_road.jpg"
MODEL_PATH = "yolo26n-sem.pt"

# From the model output:
# 0 = road
ROAD_CLASS = 0

print("Loading model...")
model = YOLO(MODEL_PATH)

print("Analyzing image...")

results = model.predict(
    source=IMAGE_PATH,
    imgsz=1024,
    verbose=False
)

result = results[0]

# Read original image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not find {IMAGE_PATH}"
    )

height, width = image.shape[:2]

# ==========================================
# GET SEMANTIC CLASS MAP
# ==========================================

if result.semantic_mask is None:
    raise RuntimeError(
        "The model did not return a semantic mask."
    )

class_map = result.semantic_mask.data.cpu().numpy()

# Make sure the mask matches the original image
if class_map.shape != (height, width):
    class_map = cv2.resize(
        class_map,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

# ==========================================
# EXTRACT ROAD
# ==========================================

road_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

road_mask[class_map == ROAD_CLASS] = 255

# ==========================================
# CLEAN ROAD MASK
# ==========================================

kernel = np.ones((7, 7), np.uint8)

road_mask = cv2.morphologyEx(
    road_mask,
    cv2.MORPH_CLOSE,
    kernel
)

road_mask = cv2.morphologyEx(
    road_mask,
    cv2.MORPH_OPEN,
    kernel
)

# ==========================================
# CREATE OVERLAY
# ==========================================

overlay = image.copy()

overlay[road_mask == 255] = (0, 255, 0)

result_image = cv2.addWeighted(
    image,
    0.65,
    overlay,
    0.35,
    0
)

# ==========================================
# SAVE RESULTS
# ==========================================

cv2.imwrite(
    "road_mask.png",
    road_mask
)

cv2.imwrite(
    "road_overlay.jpg",
    result_image
)

# ==========================================
# STATISTICS
# ==========================================

road_pixels = np.count_nonzero(road_mask)
total_pixels = height * width
road_percentage = (
    road_pixels / total_pixels
) * 100

print()
print("================================")
print("ROAD MASK COMPLETE")
print("================================")

print(f"Image size: {width} x {height}")
print(f"Road pixels: {road_pixels}")
print(f"Road coverage: {road_percentage:.2f}%")

print()
print("Saved:")
print("  road_mask.png")
print("  road_overlay.jpg")