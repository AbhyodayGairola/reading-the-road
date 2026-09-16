from ultralytics import YOLO
import cv2
import os

# ==========================================
# READ THE ROAD - ROAD SEGMENTATION TEST
# ==========================================

IMAGE_PATH = "test_road.jpg"
MODEL_PATH = "yolo26n-sem.pt"

# Load semantic segmentation model
print("Loading segmentation model...")
model = YOLO(MODEL_PATH)

# Run inference
print("Analyzing road image...")
results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=False,
    verbose=False
)

result = results[0]

# Get the model's class names
names = result.names

print("\nDetected classes:")
for class_id, class_name in names.items():
    print(f"  {class_id}: {class_name}")

# Create visualization
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not find {IMAGE_PATH}")

# Draw segmentation result
annotated = result.plot()

# Save result
output_path = "segmentation_result.jpg"
cv2.imwrite(output_path, annotated)

print("\n================================")
print("SEGMENTATION COMPLETE")
print("================================")
print(f"Output saved to: {output_path}")