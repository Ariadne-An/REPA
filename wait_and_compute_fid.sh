#!/bin/bash

echo "Waiting for image generation to complete..."

while true; do
  ema_count=$(ls eval_outputs/ema_vs_nonema/samples_ema/*.png 2>/dev/null | wc -l)
  nonema_count=$(ls eval_outputs/ema_vs_nonema/samples_nonema/*.png 2>/dev/null | wc -l)
  
  echo "[$(date '+%H:%M:%S')] EMA: $ema_count/300, non-EMA: $nonema_count/300"
  
  if [ "$ema_count" -eq 300 ] && [ "$nonema_count" -eq 300 ]; then
    echo "Both tasks completed! Computing FID..."
    break
  fi
  
  sleep 30
done

# Compute FID for both
python -c "
from cleanfid import fid
import json

real_dir = '/workspace/data/val_images_512'

# EMA FID
ema_dir = 'eval_outputs/ema_vs_nonema/samples_ema'
ema_fid = fid.compute_fid(real_dir, ema_dir)
print(f'EMA FID: {ema_fid:.2f}')

# Non-EMA FID
nonema_dir = 'eval_outputs/ema_vs_nonema/samples_nonema'
nonema_fid = fid.compute_fid(real_dir, nonema_dir)
print(f'Non-EMA FID: {nonema_fid:.2f}')

# Save results
results = {
    'ema': {'fid': ema_fid},
    'nonema': {'fid': nonema_fid}
}

with open('eval_outputs/ema_vs_nonema/fid_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n=== COMPARISON ===')
print(f'EMA FID: {ema_fid:.2f}')
print(f'Non-EMA FID: {nonema_fid:.2f}')
if ema_fid < nonema_fid:
    print(f'EMA is BETTER (lower FID by {nonema_fid - ema_fid:.2f})')
else:
    print(f'Non-EMA is BETTER (lower FID by {ema_fid - nonema_fid:.2f})')
"
