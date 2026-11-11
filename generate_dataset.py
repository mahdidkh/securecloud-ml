"""Generate a SYNTHETIC dataset of 5-minute security-event windows (SecureCloud AI, ML module).

Each row mirrors the features the dashboard backend can compute from the events stored by the
collector (GuardDuty findings, security-group changes, event timestamps...).
The lab produces too few real events to train a model, so the dataset is synthetic.

Label 0 = normal window, label 1 = attack window. Labels are used ONLY to evaluate (and to pick
the threshold on the validation split); they never train the detector.

Realism / difficulty (corrections vs. the first version):
  * normal traffic is over-dispersed and contains BENIGN look-alikes (deployment bursts,
    maintenance windows with several security-group changes, low/medium GuardDuty clusters);
  * attacks overlap with normal traffic (weak intensity, medium severity), so the problem is not
    trivially separable;
  * `mean_event_gap_s` (mean seconds between two consecutive events of the window) is a
    continuous variable: it removes the massive duplicates produced by purely discrete counters.
    NEW FEATURE: the backend must compute it from the event timestamps at inference time.

Splits: train (normal only) / val (normal + attacks) / test (normal + attacks) /
        contam (attacks only, pool used ONLY by the training-contamination experiment).
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ["event_count", "guardduty_count", "max_severity", "mean_severity",
            "sg_change_count", "distinct_event_types", "mean_event_gap_s"]
WINDOW_S = 300.0


def _gap(ev, rng, shift):
    g = WINDOW_S / (ev + 1.0) * rng.lognormal(shift, 0.7, len(ev))
    g = np.where(g > WINDOW_S, WINDOW_S * rng.uniform(0.6, 1.0, len(ev)), g)   # no ties at the window length
    return np.round(g, 2)


def normal_windows(n, rng):
    ev = rng.negative_binomial(2, 0.4, n)                                   # mean ~3, over-dispersed
    ev = ev + np.where(rng.random(n) < 0.04, rng.poisson(10, n), 0)         # benign deployment bursts
    gd = rng.binomial(np.minimum(ev, 3), 0.08)
    gd = gd + np.where(rng.random(n) < 0.02, rng.integers(2, 6, n), 0)      # benign low-severity clusters
    ev = np.maximum(ev, gd)
    mx = np.where(gd > 0, np.where(rng.random(n) < 0.12, rng.uniform(4, 7, n), rng.uniform(1, 4, n)), 0.0)
    mean = np.where(gd > 0, mx * rng.uniform(0.6, 1.0, n), 0.0)
    sg = rng.binomial(np.minimum(ev, 4), 0.12)
    sg = sg + np.where(rng.random(n) < 0.03, rng.integers(2, 7, n), 0)      # benign maintenance windows
    ev = np.maximum(ev, sg + gd)
    types = np.where(ev == 0, 0, np.minimum(1 + rng.poisson(0.8, n), np.maximum(ev, 1)))
    return pd.DataFrame({"event_count": ev, "guardduty_count": gd, "max_severity": mx.round(1),
                         "mean_severity": mean.round(1), "sg_change_count": sg,
                         "distinct_event_types": types, "mean_event_gap_s": _gap(ev, rng, 0.0)})


def attack_windows(n, rng):
    rows = []
    kinds = rng.choice(["guardduty_burst", "sg_tampering", "mixed", "critical_single"], n,
                       p=[0.3, 0.3, 0.25, 0.15])
    for k in kinds:
        if k == "guardduty_burst":
            gd = int(rng.integers(2, 16)); sg = int(rng.integers(0, 3)); mx = float(rng.uniform(4, 9))
            mean = mx * float(rng.uniform(0.6, 0.95)); ev = gd + sg + int(rng.poisson(3)); types = int(rng.integers(1, 5))
        elif k == "sg_tampering":
            sg = int(rng.integers(2, 15)); gd = int(rng.integers(0, 2)); ev = sg + gd + int(rng.poisson(2))
            mx = float(rng.uniform(2, 6)) if gd else 0.0; mean = mx * 0.8; types = int(rng.integers(1, 4))
        elif k == "mixed":
            gd = int(rng.integers(1, 7)); sg = int(rng.integers(1, 7)); mx = float(rng.uniform(3, 8.5))
            mean = mx * float(rng.uniform(0.6, 0.9)); ev = gd + sg + int(rng.poisson(2)); types = int(rng.integers(2, 6))
        else:  # critical_single: one high-severity finding in a quiet window (subtle)
            gd = int(rng.integers(1, 3)); sg = 0; mx = float(rng.uniform(5.5, 9.5)); mean = mx * float(rng.uniform(0.8, 1.0))
            ev = int(rng.integers(1, 6)); types = int(rng.integers(1, 3))
        ev = max(ev, gd + sg)
        rows.append({"event_count": ev, "guardduty_count": gd, "max_severity": round(mx, 1),
                     "mean_severity": round(mean, 1), "sg_change_count": sg,
                     "distinct_event_types": types, "scenario": k})
    df = pd.DataFrame(rows)
    df["mean_event_gap_s"] = _gap(df["event_count"].to_numpy(float), rng, -0.2)   # attacks are slightly burstier
    return df


def build(seed=42, n_train=5000, n_val_norm=1200, n_val_att=180, n_test_norm=1200, n_test_att=180,
          n_contam=600):
    """Raw (not de-duplicated) dataset; de-duplication across splits is done in the notebook."""
    rng = np.random.default_rng(seed)
    parts = []
    for split, nn, na in [("train", n_train, 0), ("val", n_val_norm, n_val_att),
                          ("test", n_test_norm, n_test_att), ("contam", 0, n_contam)]:
        if nn:
            norm = normal_windows(nn, rng); norm["label"] = 0; norm["scenario"] = "normal"; norm["split"] = split
            parts.append(norm)
        if na:
            att = attack_windows(na, rng); att["label"] = 1; att["split"] = split; parts.append(att)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = build(a.seed)
    df.to_csv(a.out, index=False)
    print(f"dataset written to {a.out}: {len(df)} rows")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
