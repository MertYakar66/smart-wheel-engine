import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import json

d = json.load(open('docs/verification_artifacts/vnv_2026-06-01/calibration_full.json'))
print('total:', d['total'], 'skipped_no_price:', d['skipped_no_price'], 'n_dates:', len(d['dates']))

print('\n-- calibration_all (doc table in VNV section 3) --')
for row in d['calibration_all']:
    print(row)

print('\n-- top_bin_by_regime (doc: 2022 rate-bear realized 0.577, gap -0.345) --')
for row in d['top_bin_by_regime']:
    print(row)

print('\n-- ev_realism (doc: pearson -0.018; Q5 mean realized +106.55; sign-gate +85.22 / -9.45) --')
print(json.dumps(d['ev_realism'], indent=1)[:2000])
