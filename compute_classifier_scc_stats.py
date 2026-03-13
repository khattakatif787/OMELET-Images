import os
import pandas as pd

# rejection value stored in SCC files
REJECT_TAG = "-1"

# folders
CLASSIFIER_FOLDER = "debug/tmp"
SCC_FOLDER = "debug/SCC_outputs"

OUTPUT_FILE = "debug/ENSEMBLE_outputs/classifier_scc_statistics.xlsx"


def compute_simple_classifier_stats(folder, file_suffix):

    rows = []

    for file in os.listdir(folder):

        if not file.endswith(file_suffix):
            continue

        path = os.path.join(folder, file)

        df = pd.read_csv(path)

        true = df["true_label"].astype(str)
        pred = df["predicted_label"].astype(str)

        correct = (true == pred).sum()
        incorrect = (true != pred).sum()
        total = len(df)

        rows.append({
            "classifier": file.replace(file_suffix, ""),
            "correct": correct,
            "incorrect": incorrect,
            "total": total
        })

    return pd.DataFrame(rows)


def compute_scc_stats(file_path, alr):

    df = pd.read_csv(file_path)

    true = df["true_label"].astype(str)

    rows = []

    for col in df.columns[1:]:

        pred = df[col].astype(str)

        rejected = (pred == REJECT_TAG).sum()

        correct = ((pred == true) & (pred != REJECT_TAG)).sum()

        incorrect = ((pred != true) & (pred != REJECT_TAG)).sum()

        total = len(df)

        rows.append({
            "ALR": float(alr),
            "SCC": col,
            "correct": correct,
            "incorrect": incorrect,
            "rejected": rejected,
            "total": total
        })

    return pd.DataFrame(rows)


def collect_all_scc_stats(scc_folder, prefix):

    all_rows = []

    for file in os.listdir(scc_folder):

        if not file.startswith(prefix):
            continue

        if not file.endswith(".csv"):
            continue

        alr = file.replace(prefix, "").replace(".csv", "")

        path = os.path.join(scc_folder, file)

        stats = compute_scc_stats(path, alr)

        all_rows.append(stats)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ----- SIMPLE CLASSIFIERS -----

    simple_val = compute_simple_classifier_stats(
        CLASSIFIER_FOLDER,
        "_VAL.csv"
    )

    simple_test = compute_simple_classifier_stats(
        CLASSIFIER_FOLDER,
        "_TEST.csv"
    )

    # ----- SCC STATS -----

    scc_val = collect_all_scc_stats(
        SCC_FOLDER,
        "SCC_VALIDATION_ALR_"
    )

    scc_test = collect_all_scc_stats(
        SCC_FOLDER,
        "SCC_TEST_ALR_"
    )

    # ----- WRITE EXCEL -----

    with pd.ExcelWriter(OUTPUT_FILE) as writer:

        simple_val.to_excel(
            writer,
            sheet_name="Simple_Classifiers_Validation",
            index=False
        )

        simple_test.to_excel(
            writer,
            sheet_name="Simple_Classifiers_Test",
            index=False
        )

        scc_val.to_excel(
            writer,
            sheet_name="SCC_Validation",
            index=False
        )

        scc_test.to_excel(
            writer,
            sheet_name="SCC_Test",
            index=False
        )

    print("Statistics saved to:", OUTPUT_FILE)