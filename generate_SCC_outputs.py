import os
import numpy as np
import pandas as pd

REJECT_TAG = -1
NON_UM_COLS = {"true_label", "predicted_label", "is_misclassification", "probabilities"}

# is_risk:
#   True  -> higher value means more risky (worse)
#   False -> higher value means more confident (better), so convert to risk via risk = 1 - conf
UM_DIRECTION_DEFAULT = {
    "MaxProb Calculator": False,
    "Entropy Calculator": False,
    "AutoEncoder Loss (conv)": True,
    "Combined Calculator (ImageClassifier)": False,
    "Multiple Combined Calculator (6 - IrIrIrIrIrIr classifiers)": False,
}

# Measures already confidence in [0,1]
BOUNDED_CONF_01 = {
    "MaxProb Calculator",
    "Entropy Calculator",
}


def infer_um_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_UM_COLS]


def list_val_files(tmp_folder: str) -> list[str]:
    return sorted([f for f in os.listdir(tmp_folder) if f.endswith("_VAL.csv")])


def corresponding_test_file(val_filename: str) -> str:
    return val_filename.replace("_VAL.csv", "_TEST.csv")


def compute_metrics_omelet(y_true: np.ndarray, y_pred_with_reject: np.ndarray) -> dict:
    phi = np.count_nonzero(y_pred_with_reject == REJECT_TAG) / len(y_true)
    aw = np.sum(y_true == y_pred_with_reject) / len(y_true)
    ew = np.sum((y_true != y_pred_with_reject) & (y_pred_with_reject != REJECT_TAG)) / len(y_true)
    ew_ans = ew / (1.0 - phi) if (1.0 - phi) > 0 else 0.0
    return {"aw": aw, "phi": phi, "ew": ew, "ew_ans": ew_ans}


def apply_rejection(pred_labels: np.ndarray, risk_scores: np.ndarray, thr: float) -> np.ndarray:
    rej_mask = risk_scores > thr
    out = pred_labels.astype(object).copy()
    out[rej_mask] = REJECT_TAG
    return out


def find_reject_threshold_exact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_scores: np.ndarray,
    alr: float,
    *,
    use_strict_lt: bool = True,
) -> float | None:

    candidates = np.unique(risk_scores.astype(float))
    candidates = np.sort(candidates)[::-1]

    def ok(ew: float) -> bool:
        return ew < alr if use_strict_lt else ew <= alr

    best_thr = None
    best_metrics = None

    for thr in candidates:
        pred_with_reject = apply_rejection(y_pred, risk_scores, float(thr))
        metrics = compute_metrics_omelet(y_true, pred_with_reject)

        if ok(metrics["ew"]):
            best_thr = float(thr)
            best_metrics = metrics
            break

    if best_thr is None:
        return None

    if best_metrics is not None and best_metrics["aw"] == 0:
        return None

    return best_thr


def normalize_01_with_params(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    v = values.astype(float)
    denom = vmax - vmin

    if not np.isfinite(denom) or denom <= 0:
        out = np.zeros_like(v, dtype=float)
    else:
        out = (v - vmin) / denom

    out = np.clip(out, 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def looks_like_combined(um_name: str) -> bool:
    return ("Combined Calculator" in um_name) or ("Multiple Combined Calculator" in um_name)


def to_risk_raw(values: np.ndarray, um_name: str, is_risk: bool) -> np.ndarray:
    """
    Convert raw UM values to risk (higher = worse) without normalization
    """

    v = values.astype(float).copy()

    v = np.nan_to_num(
        v,
        nan=0.0,
        posinf=np.nanmax(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0.0,
        neginf=0.0,
    )

    # Handle combined scores [-1,1] → [0,1]
    if looks_like_combined(um_name) and (not is_risk):
        if np.nanmin(v) < 0.0:
            v = (v + 1.0) / 2.0


    if not is_risk:
        v = 1.0 - v

    return v


def to_risk_scaled_valtest(
    val_values: np.ndarray,
    test_values: np.ndarray,
    um_name: str,
    is_risk: bool,
) -> tuple[np.ndarray, np.ndarray]:

    risk_val_raw = to_risk_raw(val_values, um_name, is_risk=is_risk)
    risk_test_raw = to_risk_raw(test_values, um_name, is_risk=is_risk)

    # Already bounded measures → no normalization
    if (um_name in BOUNDED_CONF_01) or looks_like_combined(um_name):

        risk_val = np.clip(risk_val_raw, 0.0, 1.0)
        risk_test = np.clip(risk_test_raw, 0.0, 1.0)

        return risk_val, risk_test

    # Only AutoEncoder Loss reaches here
    vmin = np.nanmin(risk_val_raw)
    vmax = np.nanmax(risk_val_raw)

    risk_val = normalize_01_with_params(risk_val_raw, vmin, vmax)
    risk_test = normalize_01_with_params(risk_test_raw, vmin, vmax)

    return risk_val, risk_test


def generate_scc_outputs(
    tmp_folder: str = "tmp",
    output_folder: str = "SCC_outputs",
    alr_list: list[float] | None = None,
    um_direction: dict | None = None,
    use_strict_lt: bool = True,
):

    if alr_list is None:
        alr_list = [0.01]

    if um_direction is None:
        um_direction = UM_DIRECTION_DEFAULT

    os.makedirs(output_folder, exist_ok=True)

    val_files = list_val_files(tmp_folder)

    if len(val_files) == 0:
        raise FileNotFoundError(f"No *_VAL.csv found in '{tmp_folder}'")

    for alr in alr_list:

        print(f"\n==================== ALR = {alr} ====================\n")

        val_out = None
        test_out = None

        for val_file in val_files:

            base = val_file.replace("_VAL.csv", "")
            test_file = corresponding_test_file(val_file)

            val_path = os.path.join(tmp_folder, val_file)
            test_path = os.path.join(tmp_folder, test_file)

            if not os.path.exists(test_path):
                print(f"[WARN] Missing TEST file for {base}: {test_path}")
                continue

            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)

            if val_out is None:
                val_out = pd.DataFrame({"true_label": df_val["true_label"].values})
                test_out = pd.DataFrame({"true_label": df_test["true_label"].values})

            um_cols = infer_um_columns(df_val)

            for um in um_cols:

                if um not in um_direction:
                    print(f"[SKIP] UM '{um}' not in UM_DIRECTION. Add if needed.")
                    continue

                is_risk = um_direction[um]

                y_true_val = df_val["true_label"].to_numpy()
                y_pred_val = df_val["predicted_label"].to_numpy()

                risk_val, risk_test = to_risk_scaled_valtest(
                    df_val[um].to_numpy(),
                    df_test[um].to_numpy(),
                    um_name=um,
                    is_risk=is_risk,
                )

                thr = find_reject_threshold_exact(
                    y_true_val,
                    y_pred_val,
                    risk_val,
                    alr,
                    use_strict_lt=use_strict_lt,
                )

                col_name = f"{base}_SCC_{um}"

                if thr is None:
                    print(f"[SKIP] {base} — {um}: cannot meet ALR={alr}. Not writing this SCC column.")
                    continue

                val_pred = apply_rejection(y_pred_val, risk_val, thr)
                test_pred = apply_rejection(df_test["predicted_label"].to_numpy(), risk_test, thr)

                val_out[col_name] = val_pred
                test_out[col_name] = test_pred

                met_val = compute_metrics_omelet(y_true_val, val_pred)

                print(
                    f"[OK] {base} — {um} | thr={thr:.6f} | "
                    f"VAL aw={met_val['aw']:.4f}, phi={met_val['phi']:.4f}, "
                    f"ew={met_val['ew']:.6f} ({'<' if use_strict_lt else '<='} ALR={alr})"
                )

        val_out_path = os.path.join(output_folder, f"SCC_VALIDATION_ALR_{alr}.csv")
        test_out_path = os.path.join(output_folder, f"SCC_TEST_ALR_{alr}.csv")

        if val_out is None or test_out is None:
            raise RuntimeError(
                f"No outputs were produced for ALR={alr}. "
                f"Check that *_TEST.csv files exist and UM names match UM_DIRECTION."
            )

        val_out.to_csv(val_out_path, index=False)
        test_out.to_csv(test_out_path, index=False)

        print("\nSaved:")
        print(val_out_path)
        print(test_out_path)


if __name__ == "__main__":

    generate_scc_outputs(
        tmp_folder="debug/tmp",
        output_folder="debug/SCC_outputs",
        alr_list=[0.0, 0.01, 0.001, 0.0001, 0.02, 0.002, 0.0002, 0.03, 0.003, 0.0003, 0.04, 0.004, 0.0004],
        use_strict_lt=False,
    )