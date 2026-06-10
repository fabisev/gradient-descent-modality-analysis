import pandas as pd
from scipy.stats import mannwhitneyu


def cliffs_delta(classic, multiple):
    """
    Compute Cliff's delta.

    Positive delta means scores are higher in the multiple-modality condition.
    Negative delta means scores are higher in the classic textbook-style condition.
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
    """
    Run a Mann--Whitney U test and compute Cliff's delta
    for one measure.
    """
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
        "classic_median": classic.median(),
        "classic_min": classic.min(),
        "classic_max": classic.max(),
        "multiple_mean": multiple.mean(),
        "multiple_median": multiple.median(),
        "multiple_min": multiple.min(),
        "multiple_max": multiple.max(),
        "U": u_stat,
        "p": p_value,
        "cliffs_delta": delta
    }


def clean_dataframe(df):
    """
    Clean column names and condition labels.
    """
    df.columns = df.columns.str.strip()

    df["condition"] = (
        df["condition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def run_analysis(input_file, measures, output_file, title):
    """
    Read one CSV file, analyse all selected measures,
    save the results, and print them.
    """
    df = pd.read_csv(input_file)
    df = clean_dataframe(df)

    results = [
        analyse_measure(df, measure)
        for measure in measures
    ]

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_file, index=False)

    print(f"\n{title}")
    print(results_df.round(3))


def main():
    # -------------------------
    # Post-test score analysis
    # -------------------------
    posttest_measures = [
        "total",
        "A_core",
        "B_compute",
        "C_learning_rate",
        "D_application",
        "E_process",
        "F_variants"
    ]

    run_analysis(
        input_file="data/anonymised_scores.csv",
        measures=posttest_measures,
        output_file="results/posttest_statistical_results.csv",
        title="Post-test results"
    )

    # -------------------------
    # Survey construct analysis
    # -------------------------
    survey_measures = [
        "confidence",
        "clarity",
        "cognitive_load",
        "mental_effort",
        "usefulness",
        "engagement"
    ]

    run_analysis(
        input_file="data/anonymised_survey_scores.csv",
        measures=survey_measures,
        output_file="results/survey_statistical_results.csv",
        title="Survey results"
    )


if __name__ == "__main__":
    main()