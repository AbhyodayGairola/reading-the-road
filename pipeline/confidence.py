import numpy as np

# ============================================================
# INPUTS FROM OUR CURRENT PIPELINE
# ============================================================

# Road-width estimates from different cross-sections
widths = np.array([
    11.05,
    12.11,
    11.61,
    12.54,
    13.13,
    13.15
])

# Width estimates caused by different possible camera FOVs
fov_widths = np.array([
    10.16,   # 60°
    11.21,   # 65°
    12.32,   # 70°
    13.50,   # 75°
    14.77    # 80°
])

# ============================================================
# 1. WIDTH CONSISTENCY
# ============================================================

median_width = np.median(widths)
std_width = np.std(widths)

variation_ratio = std_width / median_width

width_consistency = max(
    0,
    min(100, 100 * (1 - variation_ratio))
)

# ============================================================
# 2. CAMERA CALIBRATION UNCERTAINTY
# ============================================================

fov_median = np.median(fov_widths)

fov_uncertainty = (
    (np.max(fov_widths) - np.min(fov_widths))
    / fov_median
)

calibration_confidence = max(
    0,
    min(100, 100 * (1 - fov_uncertainty))
)

# ============================================================
# 3. OVERALL CONFIDENCE
# ============================================================

# Current prototype weights
overall_confidence = (
    0.55 * width_consistency +
    0.45 * calibration_confidence
)

# ============================================================
# 4. ESTIMATE RANGE
# ============================================================

lower = np.percentile(widths, 10)
upper = np.percentile(widths, 90)

# Add calibration uncertainty
calibration_factor = fov_uncertainty / 2

range_lower = lower * (1 - calibration_factor)
range_upper = upper * (1 + calibration_factor)

# ============================================================
# RESULT
# ============================================================

print("=" * 65)
print("ROAD WIDTH CONFIDENCE ANALYSIS")
print("=" * 65)

print(f"Median width             : {median_width:.2f} m")
print(f"Cross-section variation  : {std_width:.2f} m")
print(f"Width consistency        : {width_consistency:.1f}%")

print()

print(f"FOV-based width range    : "
      f"{np.min(fov_widths):.2f} - {np.max(fov_widths):.2f} m")

print(f"Calibration confidence   : "
      f"{calibration_confidence:.1f}%")

print()

print(f"Estimated width range    : "
      f"{range_lower:.2f} - {range_upper:.2f} m")

print(f"Overall confidence       : "
      f"{overall_confidence:.1f}%")

print()

if overall_confidence >= 80:
    status = "HIGH CONFIDENCE"
elif overall_confidence >= 60:
    status = "MEDIUM CONFIDENCE"
else:
    status = "LOW CONFIDENCE"

print(f"STATUS                   : {status}")

print("=" * 65)

print("\nReason:")
print("- Cross-section consistency is reasonably strong.")
print("- Camera calibration is currently UNKNOWN.")
print("- Therefore metric width should NOT be treated as survey-grade.")