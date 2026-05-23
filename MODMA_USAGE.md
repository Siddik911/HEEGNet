# MODMA ERP training/evaluation with HEEGNet

This repository includes `modma_train.py` for cross-subject training and evaluation on MODMA ERP `.raw` files.

## 1) Labels spreadsheet

You can pass the original MODMA subject sheet directly.

The script now auto-detects common column names:
- subject id: `subject id` / `subject_id`
- label: `type` / `label` / `group` / `diagnosis`
- filename (optional): `filename` / `file` / `raw file`

If no filename column is present, it matches files by prefix using subject id,
for example `02010002` -> `02010002erp 20150416 1131.raw`.

Label mapping for string labels:
- `MDD`, `depression`, `patient` -> `1`
- `HC`, `control`, `healthy` -> `0`

## 2) Install dependencies

```bash
conda activate HEEGNet
pip install -U mne openpyxl
```

## 3) Run training + testing (cross-subject)

```bash
python modma_train.py \
  --data-root "/home/hasan/EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015" \
  --labels-xlsx "/home/hasan/EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx" \
  --epochs 100 \
  --batch-size 50 \
  --domains-per-batch 5 \
  --lr 0.001 \
  --weight-decay 1e-4 \
  --swd-weight 0.01 \
  --input-align
```


If you hit OOM/"killed", start with safer memory settings:

```bash
python modma_train.py \
  --data-root "/home/hasan/EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015" \
  --labels-xlsx "/home/hasan/EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx" \
  --epochs 60 \
  --batch-size 16 \
  --domains-per-batch 4 \
  --epoch-seconds 4 \
  --max-epochs-per-subject 80 \
  --dtype float32 \
  --input-align
```

## 4) Outputs

- `results/modma_fold_results.csv` (fold/domain metrics)
- `results/modma_summary.csv` (mean/std summary)

## 5) Notes

- Assumes one session per subject (session is set to 1 internally).
- Cross-subject protocol uses `GroupKFold` / `LeaveOneGroupOut` on subject id.
- Current epoching uses fixed-length windows; if you need event-locked ERP epoching,
  adjust `build_trials_from_raw()` in `modma_train.py`.
