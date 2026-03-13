import os
import sys
import numpy
import pandas

REJECT_TAG = "-1"
ALLOWED_ALRS = {"0.0", "0.01", "0.001", "0.0001"}


def compute_omission_metrics(y_true: numpy.ndarray, y_fcc: numpy.ndarray, reject_tag=None) -> dict:
    met_dict = {}
    met_dict['phi'] = numpy.count_nonzero(y_fcc == reject_tag) / len(y_true)
    met_dict['aw'] = numpy.sum(y_true == y_fcc) / len(y_true)
    met_dict['ew'] = numpy.sum((y_true != y_fcc) & (y_fcc != reject_tag)) / len(y_true)
    met_dict['ew_ans'] = met_dict['ew'] / (1 - met_dict['phi']) if (1 - met_dict['phi']) > 0 else 0.0
    return met_dict


def get_scc_columns(df: pandas.DataFrame) -> list:
    sccs = []
    for col_name in df.columns:
        if col_name == "true_label":
            continue
        if str(col_name).startswith("Unnamed:"):
            continue
        sccs.append(col_name)
    return sccs


def process_one_alr(input_folder: str, output_folder: str, test_file: str, reject_tag: str = "-1") -> None:
    if "ALR_" not in test_file:
        print(f"Skipping file with unknown format: {test_file}")
        return

    alr = test_file.split("ALR_")[1].replace(".csv", "")

    if alr not in ALLOWED_ALRS:
        print(f"Skipping ALR={alr} because it is not in the allowed set")
        return

    val_file = test_file.replace("TEST", "VALIDATION")

    val_path = os.path.join(input_folder, val_file)
    test_path = os.path.join(input_folder, test_file)

    if not os.path.exists(val_path):
        print(f"Missing validation file for test file: {test_file}")
        print(f"Expected validation file: {val_file}")
        return

    val_res_df = pandas.read_csv(val_path, dtype=str)
    test_res_df = pandas.read_csv(test_path, dtype=str)

    if "true_label" not in val_res_df.columns or "true_label" not in test_res_df.columns:
        print(f"Missing 'true_label' column for ALR={alr}")
        return

    sccs = get_scc_columns(val_res_df)

    if len(sccs) == 0:
        print(f"No candidate SCC columns found for ALR={alr}")
        return

    gain = numpy.zeros((len(sccs), len(sccs)))
    drop = numpy.zeros((len(sccs), len(sccs)))

    y_val = val_res_df["true_label"].astype(str).to_numpy()
    y_test = test_res_df["true_label"].astype(str).to_numpy()

    scores_file = os.path.join(output_folder, "couples_scores.csv")

    for i in range(len(sccs)):
        first_name = sccs[i]
        first = val_res_df[first_name].astype(str).str.strip().to_numpy()
        first_t = test_res_df[first_name].astype(str).str.strip().to_numpy()

        for j in range(len(sccs)):
            second_name = sccs[j]
            second = val_res_df[second_name].astype(str).str.strip().to_numpy()
            second_t = test_res_df[second_name].astype(str).str.strip().to_numpy()

            gain[i, j] = numpy.sum((first == reject_tag) & (second == y_val)) / len(first)
            drop[i, j] = numpy.sum((first == reject_tag) & (second != y_val) & (second != reject_tag)) / len(first)

            rb_predict = numpy.where(first != reject_tag, first, second)
            rb_predict_t = numpy.where(first_t != reject_tag, first_t, second_t)

            val_fcc_metrics = compute_omission_metrics(y_val, rb_predict, reject_tag=reject_tag)
            test_fcc_metrics = compute_omission_metrics(y_test, rb_predict_t, reject_tag=reject_tag)

            with open(scores_file, "a") as file_handler:
                file_handler.write(
                    f"{alr},{first_name},{second_name},"
                    f"{gain[i, j]},{drop[i, j]},"
                    f"{val_fcc_metrics['aw']},{val_fcc_metrics['phi']},{val_fcc_metrics['ew']},"
                    f"{test_fcc_metrics['aw']},{test_fcc_metrics['phi']},{test_fcc_metrics['ew']}\n"
                )

    gain_df = pandas.DataFrame(data=gain, columns=sccs, index=sccs)
    gain_df.to_csv(os.path.join(output_folder, f"GAIN_ALR_{alr}.csv"))

    drop_df = pandas.DataFrame(data=drop, columns=sccs, index=sccs)
    drop_df.to_csv(os.path.join(output_folder, f"DROP_ALR_{alr}.csv"))

    print(f"Processed ALR={alr}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_scc_couples.py <input_folder>")
        print("Example: python run_scc_couples.py debug/SCC_outputs")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = os.path.join("debug", "SCC_couples")

    if not os.path.isdir(input_folder):
        print(f"Input folder not found: {input_folder}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    scores_file = os.path.join(output_folder, "couples_scores.csv")
    with open(scores_file, "w") as file_handler:
        file_handler.write(
            "alr,scc1,scc2,val_gain,val_drop,val_aw,val_phi,val_ew,test_aw,test_phi,test_ew\n"
        )

    files = os.listdir(input_folder)
    test_files = sorted([f for f in files if "TEST" in f and f.endswith(".csv")])

    if len(test_files) == 0:
        print(f"No TEST csv files found in: {input_folder}")
        sys.exit(1)

    for test_file in test_files:
        process_one_alr(input_folder, output_folder, test_file, reject_tag=REJECT_TAG)

    print(f"Done. Results saved in: {output_folder}")