import pandas as pd
from scipy.stats import mannwhitneyu


def cliffs_delta(classic, multiple):
    """
    Positive delta means scores are higher in the multiple-modality condition.
    """
    greater = 0
    lower = 0

    for m in multiple:
        for c in classic:
            if m > c:
                greater += 1
            elif m < c:
                lower += 1

    return (greater - lower) / (len(classic) * len(multiple))


def analyse_measure(df, measure):
    classic = df[df["condition"] == "classic"][measure].dropna()
    multiple = df[df["condition"] == "multiple"][measure].dropna()

    u_stat, p_value = mannwhitneyu(
        multiple,
        classic,
        alternative="two-sided"
    )

    delta = cliffs_delta(classic, multiple)

    return {
        "measure": measure,
        "n_classic": len(classic),
        "n_multiple": len(multiple),
        "classic_mean": classic.mean(),
        "multiple_mean": multiple.mean(),
        "U": u_stat,
        "p": p_value,
        "cliffs_delta": delta
    }


def main():
    df = pd.read_csv("data/anonymised_scores.csv")

    # Clean column names just in case there are spaces
    df.columns = df.columns.str.strip()

    # Clean condition values just in case there are spaces/capital letters
    df["condition"] = df["condition"].str.strip().str.lower()

    measures = [
        "total",
        "A_core",
        "B_compute",
        "C_learning_rate",
        "D_application",
        "E_process",
        "F_variants"
    ]

    results = [analyse_measure(df, measure) for measure in measures]
    results_df = pd.DataFrame(results)

    results_df.to_csv("results/statistical_results.csv", index=False)

    print(results_df.round(3))


if __name__ == "__main__":
    main()