# MODMA ERP training/evaluation with HEEGNet

This repository now includes `modma_train.py` for cross-subject training and evaluation on the MODMA ERP dataset.

## 1) Prepare labels sheet

Create an `.xlsx` file with **exactly** these columns:

- `subject_id` (int)
- `label` (int)
- `filename` (string; exact raw filename inside your dataset directory)

Example rows:

| subject_id | label | filename |
|---:|---:|---|
| 2010002 | 0 | 02010002erp 20150416 1131.raw |
| 2010004 | 1 | 02010004erp 20141219 1602.raw |

## 2) Install dependencies in conda env

```bash
conda activate HEEGNet
pip install -U mne openpyxl
```

## 3) Run training + testing (cross-subject)

```bash
python modma_train.py \
  --data-root /home/hasan/EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015 \
  --labels-xlsx /home/hasan/EEG/Dataset/MODMA_128_channel_resting/labels_modma.xlsx \
  --epochs 100 \
  --batch-size 50 \
  --domains-per-batch 5 \
  --lr 0.001 \
  --weight-decay 1e-4 \
  --swd-weight 0.01 \
  --input-align
```

## 4) Output files

After completion:

- `results/modma_fold_results.csv` (per-fold/per-domain metrics)
- `results/modma_summary.csv` (mean/std summary)

## 5) Notes

- This script assumes **one session per subject** (domain = subject_session with session=1).
- Cross-subject protocol is enforced via `GroupKFold` / `LeaveOneGroupOut` on subject ids.
- If your ERP eventing differs, update `build_trials_from_raw()` in `modma_train.py`.
