import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import math

# Independent re-derivation of the I1 top-bin Wilson 95% CI from the artifact's n + realized rate.
# Artifact (raw_output/i1_calibration_RAW.txt line 15):
#   (.95,1]  105   0.9657    0.6952 [0.602,0.775]    -27.05  MISCAL
n = 105
phat = 0.6952  # implies k = 73 successes (73/105 = 0.695238)
k = round(phat * n)
p = k / n
z = 1.959963984540054
denom = 1 + z * z / n
center = (p + z * z / (2 * n)) / denom
half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
lo, hi = center - half, center + half
print(f"k={k}, p={p:.6f}")
print(f"Wilson 95% CI = [{lo:.3f}, {hi:.3f}]  (doc/artifact: [0.602, 0.775])")
print(f"delta_pp = {(p - 0.9657) * 100:.2f}  (doc/artifact: -27.05)")
print(f"forecast 0.9657 inside CI? {lo <= 0.9657 <= hi}")

# Same check for the (.9,.95] bin from the I1 doc table: n=1017, realized 0.811, CI [0.786,0.834]
n2 = 1017
p2 = round(0.811 * n2) / n2
denom2 = 1 + z * z / n2
center2 = (p2 + z * z / (2 * n2)) / denom2
half2 = (z / denom2) * math.sqrt(p2 * (1 - p2) / n2 + z * z / (4 * n2 * n2))
print(f"\n(.9,.95]: k={round(0.811*n2)}, p={p2:.4f}, Wilson CI=[{center2-half2:.3f},{center2+half2:.3f}] (doc: [0.786,0.834])")
