import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

import nets.batchnorm as bn
import nets.functionals as fn
from nets.callbacks import EarlyStopping, MomentumBatchNormScheduler
from nets.model import HEEGNet
from nets.trainer import Trainer
from nets.utils.data import DomainDataset, StratifiedDomainDataLoader


def build_trials_from_raw(root: Path, labels_xlsx: Path, sfreq=250, tmin=0.0, tmax=1.0):
    """Build trial tensors from MODMA .raw files.

    This implementation expects MNE to be installed and that each .raw file can be
    epoched into fixed windows [tmin, tmax]. Adjust this function to match your exact
    event coding and preprocessing policy.
    """
    import mne

    df = pd.read_excel(labels_xlsx)
    normalized = {c: c.strip().lower().replace("_", " ") for c in df.columns}
    rename_map = {}
    for original, norm in normalized.items():
        if norm in {"subject id", "subjectid"}:
            rename_map[original] = "subject_id"
        elif norm in {"filename", "file", "raw file", "raw filename", "raw"}:
            rename_map[original] = "filename"
        elif norm in {"label", "class", "group", "type", "diagnosis"}:
            rename_map[original] = "label"
    df = df.rename(columns=rename_map)

    required = {"subject_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Could not find required label columns. "
            f"Needed {required}, found {set(df.columns)}"
        )

    label_map = {
        "mdd": 1,
        "depression": 1,
        "patient": 1,
        "hc": 0,
        "control": 0,
        "healthy": 0,
    }

    raw_files = sorted(root.glob("*.raw"))
    if not raw_files:
        raise FileNotFoundError(f"No .raw files found in {root}")

    def _normalize_subject_key(value):
        s = ''.join(ch for ch in str(value).strip() if ch.isdigit())
        return s.lstrip('0') or s

    def _filename_subject_key(path):
        stem = path.stem
        prefix = ''
        for ch in stem:
            if ch.isdigit():
                prefix += ch
            else:
                break
        return _normalize_subject_key(prefix)

    def resolve_raw_path(subject_id, maybe_filename):
        if isinstance(maybe_filename, str) and maybe_filename.strip():
            p = root / maybe_filename.strip()
            if p.exists():
                return p

        sid = _normalize_subject_key(subject_id)
        exact = [p for p in raw_files if _filename_subject_key(p) == sid]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise RuntimeError(
                f"Multiple raw files match subject_id={subject_id}: {[c.name for c in exact]}"
            )

        fuzzy = [p for p in raw_files if p.name.startswith(str(subject_id).strip())]
        if len(fuzzy) == 1:
            return fuzzy[0]
        if len(fuzzy) > 1:
            raise RuntimeError(
                f"Multiple raw files match subject_id={subject_id}: {[c.name for c in fuzzy]}"
            )

        raise FileNotFoundError(
            f"Could not resolve raw file for subject_id={subject_id}. "
            "Add a filename column or verify file naming."
        )

    all_x, all_y, all_subjects, all_sessions = [], [], [], []

    for _, row in df.iterrows():
        subject_id_raw = str(row["subject_id"]).strip()
        subject_id = subject_id_raw

        raw_label = row["label"]
        if isinstance(raw_label, str):
            key = raw_label.strip().lower()
            if key in label_map:
                label = label_map[key]
            else:
                raise ValueError(f"Unsupported string label '{raw_label}' for subject {subject_id_raw}")
        else:
            label = int(raw_label)

        raw_path = resolve_raw_path(subject_id_raw, row["filename"] if "filename" in row else None)

        raw = mne.io.read_raw_egi(str(raw_path), preload=True, verbose="ERROR")
        raw.filter(1.0, 45.0, verbose="ERROR")
        raw.notch_filter(50.0, verbose="ERROR")
        raw.resample(sfreq, verbose="ERROR")

        duration = tmax - tmin
        events = mne.make_fixed_length_events(raw, id=1, duration=duration)
        epochs = mne.Epochs(raw, events, event_id={"fixed": 1}, tmin=tmin, tmax=tmax,
                            baseline=(tmin, 0.0) if tmin < 0 else None,
                            preload=True, verbose="ERROR")
        x = epochs.get_data()  # [n_trials, n_channels, n_times]

        y = np.full((x.shape[0],), label, dtype=np.int64)
        s = np.full((x.shape[0],), int("".join(ch for ch in subject_id if ch.isdigit()) or 0), dtype=np.int64)
        sess = np.ones((x.shape[0],), dtype=np.int64)

        all_x.append(x)
        all_y.append(y)
        all_subjects.append(s)
        all_sessions.append(sess)

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)
    sessions = np.concatenate(all_sessions, axis=0)
    return X, y, subjects, sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-xlsx", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--domains-per-batch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--swd-weight", type=float, default=0.01)
    parser.add_argument("--input-align", action="store_true")
    parser.add_argument("--no-domain-adaptation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = dict(
        epochs=args.epochs,
        batch_size_train=args.batch_size,
        domains_per_batch=args.domains_per_batch,
        validation_size=0.2,
        dtype=torch.float64,
        lr=args.lr,
        input_align=args.input_align,
        weight_decay=args.weight_decay,
        swd_weight=args.swd_weight,
        mdl_kwargs=dict(
            bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
            domain_adaptation=not args.no_domain_adaptation,
        ),
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X, labels, subjects, sessions = build_trials_from_raw(args.data_root, args.labels_xlsx)

    domain_labels = [f"{sub}_{sess}" for sub, sess in zip(subjects, sessions)]
    domain = sklearn.preprocessing.LabelEncoder().fit_transform(domain_labels)
    domain = torch.from_numpy(domain)

    if cfg["input_align"]:
        for i in domain.unique():
            X[domain == i] = fn.euler_align(X[domain == i])

    X = torch.from_numpy(X)
    y = torch.from_numpy(sklearn.preprocessing.LabelEncoder().fit_transform(labels))

    n_classes = int(y.max().item() + 1)
    subject_count = len(np.unique(subjects))
    if subject_count < 10:
        cv_outer = sklearn.model_selection.LeaveOneGroupOut()
    else:
        cv_outer = sklearn.model_selection.GroupKFold(n_splits=10)

    cv_outer_group = subjects
    cv_inner_group = [f"{d.item()}_{l.item()}" for d, l in zip(domain, y)]
    cv_inner_group = sklearn.preprocessing.LabelEncoder().fit_transform(cv_inner_group)

    mdl_kwargs = deepcopy(cfg["mdl_kwargs"])
    mdl_kwargs["num_classes"] = n_classes
    mdl_kwargs["num_electrodes"] = X.shape[1]
    mdl_kwargs["chunk_size"] = X.shape[2]
    mdl_kwargs["domains"] = domain.unique()

    records = []

    def sfuda_offline(dataset, model):
        model.eval()
        model.domainadapt_finetune(
            dataset.features.to(dtype=cfg["dtype"], device=device),
            dataset.labels.to(device=device),
            dataset.domains,
            None,
        )

    for ix_fold, (fit, test) in enumerate(cv_outer.split(X, y, cv_outer_group)):
        cv_inner = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=cfg["validation_size"]
        )
        train, val = next(cv_inner.split(X[fit], y[fit], cv_inner_group[fit]))

        du = domain[fit][train].unique()
        domains_per_batch = min(cfg["domains_per_batch"], len(du))

        ds_train = DomainDataset(X[fit][train], y[fit][train], domain[fit][train])
        ds_val = DomainDataset(X[fit][val], y[fit][val], domain[fit][val])

        loader_train = StratifiedDomainDataLoader(
            ds_train,
            cfg["batch_size_train"],
            domains_per_batch=domains_per_batch,
            shuffle=True,
            drop_last=False,
        )
        loader_val = torch.utils.data.DataLoader(ds_val, batch_size=len(ds_val))

        model = HEEGNet(device=device, dtype=cfg["dtype"], **mdl_kwargs).to(device=device, dtype=cfg["dtype"])

        es = EarlyStopping(metric="val_loss", higher_is_better=False, patience=20, verbose=False)
        bn_sched = MomentumBatchNormScheduler(
            epochs=cfg["epochs"] - 10,
            bs0=cfg["batch_size_train"],
            bs=cfg["batch_size_train"] / cfg["domains_per_batch"],
            tau0=0.85,
        )

        trainer = Trainer(
            max_epochs=cfg["epochs"],
            min_epochs=70,
            callbacks=[es, bn_sched],
            loss=torch.nn.CrossEntropyLoss(),
            device=device,
            dtype=cfg["dtype"],
            swd_weight=cfg["swd_weight"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        trainer.fit(model, train_dataloader=loader_train, val_dataloader=loader_val)

        sfuda_offline_net = deepcopy(model)

        for td in domain[test].unique():
            ds_test = DomainDataset(
                X[test][domain[test] == td],
                y[test][domain[test] == td],
                domain[test][domain[test] == td],
            )
            loader_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))
            sfuda_offline(ds_test, sfuda_offline_net)
            res = trainer.test(sfuda_offline_net, dataloader=loader_test)
            records.append(dict(dataset="modma_erp", fold=ix_fold, domain=int(td.item()), **res))

    resdf = pd.DataFrame(records)
    summary = resdf.groupby(["dataset"]).agg(["mean", "std"]).round(4)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    resdf.to_csv(out_dir / "modma_fold_results.csv", index=False)
    summary.to_csv(out_dir / "modma_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
