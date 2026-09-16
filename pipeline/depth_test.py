from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

print("Loading Depth Anything V2 Metric Outdoor...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
)

print("Loading road image...")

image = Image.open("test_road.jpg").convert("RGB")

print("Estimating metric depth...")

result = depth_pipe(image)

depth = result["predicted_depth"].squeeze().cpu().numpy()

print("\nMetric depth statistics:")
print("Shape:", depth.shape)
print("Minimum:", float(depth.min()))
print("Maximum:", float(depth.max()))
print("Mean:", float(depth.mean()))
print("Median:", float(np.median(depth)))

# Normalize ONLY for visualization
depth_norm = cv2.normalize(
    depth,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite("metric_depth_map.png", depth_norm)

print("\nSaved: metric_depth_map.png")