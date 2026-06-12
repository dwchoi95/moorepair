"""Statistical analysis over repeated runs recorded in overall.csv.

Each repetition of run.py appends its own rows, so every row is one
observation (problem x repetition). This script compares every approach
against a baseline following the guidelines of Arcuri & Briand (2014):

  - mean +- std over observations
  - Mann-Whitney U test (two-sided)
  - Holm-Bonferroni correction across comparisons per metric
  - Vargha-Delaney A12 effect size, P(approach > baseline)

Usage:
  python stats.py                                  # defaults
  python stats.py -b MooRepair -g 4 -v ALL
  python stats.py -m %RR "ΔTMU(%)" RPS "ATT(s)"    # choose metrics
  python stats.py --per-problem                    # tests within each problem
  python stats.py --init                           # analyze init_stats.csv
"""

import argparse

import pandas as pd
from scipy.stats import mannwhitneyu
from prettytable import PrettyTable

DEFAULT_METRICS = ["%RR", "ΔTMU(%)", "ΔET(%)", "ΔMU(%)", "RPS", "ATT(s)"]
# Direction of improvement, for interpreting A12
HIGHER_IS_BETTER = {"%RR", "ΔET(%)", "ΔMU(%)", "ΔTMU(%)"}

STATS_PATH = "stats.csv"


def to_numeric(series: pd.Series) -> pd.Series:
    """Parse '82.00%' / '0.0034' / 'N/A' into floats (NaN for N/A)."""
    s = series.astype(str).str.strip().str.rstrip("%")
    return pd.to_numeric(s, errors="coerce")


def a12(x: list, y: list) -> float:
    """Vargha-Delaney A12: P(x > y) + 0.5 * P(x == y)."""
    if not x or not y:
        return float("nan")
    gt = sum(1 for a in x for b in y if a > b)
    eq = sum(1 for a in x for b in y if a == b)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def a12_magnitude(a: float) -> str:
    """Conventional thresholds: 0.56 small, 0.64 medium, 0.71 large."""
    if pd.isna(a):
        return "N/A"
    d = abs(a - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (pd.isna(pvals[i]), pvals[i]))
    adjusted = [float("nan")] * m
    prev = 0.0
    rank = 0
    for i in order:
        if pd.isna(pvals[i]):
            continue
        rank += 1
        adj = min((m - rank + 1) * pvals[i], 1.0)
        adj = max(adj, prev)  # enforce monotonicity
        adjusted[i] = adj
        prev = adj
    return adjusted


def observations(df: pd.DataFrame, approach: str, metric: str) -> list[float]:
    """One observation per CSV row (problem x repetition) for the approach."""
    sub = df[df["Approach"] == approach]
    vals = to_numeric(sub[metric]).dropna()
    return vals.tolist()


def compare(df: pd.DataFrame, baseline: str, metrics: list[str],
            scope: str) -> list[dict]:
    approaches = [a for a in df["Approach"].unique() if a != baseline]
    if baseline not in df["Approach"].unique():
        raise SystemExit(f"Baseline '{baseline}' not found. "
                         f"Available: {sorted(df['Approach'].unique())}")
    results = []
    for metric in metrics:
        if metric not in df.columns:
            print(f"[skip] column not found: {metric}")
            continue
        base_obs = observations(df, baseline, metric)
        rows = []
        for app in approaches:
            obs = observations(df, app, metric)
            if obs and base_obs and \
                    (len(set(obs)) > 1 or len(set(base_obs)) > 1 or set(obs) != set(base_obs)):
                try:
                    p = mannwhitneyu(obs, base_obs, alternative="two-sided").pvalue
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            rows.append({
                "Scope": scope,
                "Metric": metric,
                "Baseline": baseline,
                "Approach": app,
                "n_base": len(base_obs),
                "n_app": len(obs),
                "base_mean": pd.Series(base_obs).mean(),
                "base_std": pd.Series(base_obs).std(),
                "app_mean": pd.Series(obs).mean(),
                "app_std": pd.Series(obs).std(),
                "p_value": p,
                "A12": a12(obs, base_obs),
            })
        # Holm-Bonferroni across approaches within this metric
        adj = holm_bonferroni([r["p_value"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = a
            r["magnitude"] = a12_magnitude(r["A12"])
        results.extend(rows)
    return results


def print_results(results: list[dict]):
    for metric in dict.fromkeys(r["Metric"] for r in results):
        rows = [r for r in results if r["Metric"] == metric]
        direction = "higher=better" if metric in HIGHER_IS_BETTER else "lower=better"
        table = PrettyTable([
            "Approach", "mean±std", "baseline mean±std",
            "p (MWU)", "p (Holm)", "Â12", "magnitude",
        ])
        table.title = f"{metric}  [{direction}]  vs {rows[0]['Baseline']} ({rows[0]['Scope']})"
        for r in rows:
            table.add_row([
                r["Approach"],
                f"{r['app_mean']:.3f}±{r['app_std']:.3f}" if pd.notna(r['app_mean']) else "N/A",
                f"{r['base_mean']:.3f}±{r['base_std']:.3f}" if pd.notna(r['base_mean']) else "N/A",
                f"{r['p_value']:.4f}" if pd.notna(r['p_value']) else "N/A",
                f"{r['p_holm']:.4f}" if pd.notna(r['p_holm']) else "N/A",
                f"{r['A12']:.3f}" if pd.notna(r['A12']) else "N/A",
                r["magnitude"],
            ])
        print(table)


def analyze_init(path: str):
    """Analyze init_stats.csv: per-loop syntax pass rates of the
    candidates from Variation.correct, and how many loops each buggy
    program needs to fill the initial population."""
    df = pd.read_csv(path)
    if "LLM" in df.columns and df["LLM"].nunique() > 1:
        for llm, sub in df.groupby("LLM"):
            print(f"\n##### LLM: {llm} #####")
            _analyze_init_df(sub)
        return
    _analyze_init_df(df)


def _analyze_init_df(df: pd.DataFrame):
    # ---- per-loop stats -------------------------------------------- #
    per_loop = df.groupby("Loop").agg(
        buggys=("BuggyID", "count"),
        requested=("Requested", "sum"),
        generated=("Generated", "sum"),
        passed=("SyntaxPassed", "sum"),
    ).reset_index()
    table = PrettyTable([
        "Loop", "#Buggys reaching", "LLM calls", "Non-empty",
        "Syntax passed", "Pass rate", "Avg passed/buggy",
    ])
    table.title = "Initialization: per-loop candidate pass rate"
    for _, r in per_loop.iterrows():
        table.add_row([
            int(r["Loop"]), int(r["buggys"]), int(r["requested"]),
            int(r["generated"]), int(r["passed"]),
            f"{r['passed'] / r['requested'] * 100:.2f}%",
            f"{r['passed'] / r['buggys']:.2f}",
        ])
    print(table)

    # ---- loops needed per buggy ------------------------------------ #
    # max Loop per (ProblemID, BuggyID) = loops needed to fill the
    # population (always 1 since initialization became single-attempt)
    loops_needed = df.groupby(["ProblemID", "BuggyID"])["Loop"].max()
    dist = loops_needed.value_counts().sort_index()
    total = len(loops_needed)
    table = PrettyTable(["Loops needed", "#Buggys", "Share", "Cumulative"])
    table.title = "Initialization: loops needed to fill the population"
    cum = 0
    for loops, count in dist.items():
        cum += count
        table.add_row([int(loops), int(count),
                       f"{count / total * 100:.2f}%",
                       f"{cum / total * 100:.2f}%"])
    print(table)
    print(f"\nBuggy programs: {total}, "
          f"done in 1 loop: {dist.get(1, 0) / total * 100:.2f}%, "
          f"mean loops: {loops_needed.mean():.2f}, "
          f"max loops: {int(loops_needed.max())}")
    overall = df["SyntaxPassed"].sum() / df["Requested"].sum()
    print(f"Overall syntax pass rate: {overall * 100:.2f}% "
          f"({int(df['SyntaxPassed'].sum())}/{int(df['Requested'].sum())} calls)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default="overall.csv",
                        help="Path to overall.csv (default: overall.csv)")
    parser.add_argument('-b', '--baseline', type=str, default="MooRepair",
                        help="Baseline approach label (default: MooRepair)")
    parser.add_argument('-g', '--gen', type=int, default=None,
                        help="Generation to analyze (default: max in file)")
    parser.add_argument('-v', '--verdict', type=str, default="ALL",
                        help="Verdict to analyze: ALL, WA, TLE, MLE (default: ALL)")
    parser.add_argument('-m', '--metrics', type=str, nargs='+',
                        default=DEFAULT_METRICS,
                        help=f"Metric columns (default: {DEFAULT_METRICS})")
    parser.add_argument('--per-problem', action='store_true', default=False,
                        help="Run tests within each problem (low power with few runs)")
    parser.add_argument('--init', action='store_true', default=False,
                        help="Analyze initialization stats (init_stats.csv) "
                             "instead of overall.csv")
    args = parser.parse_args()

    if args.init:
        path = args.file if args.file != "overall.csv" else "init_stats.csv"
        analyze_init(path)
        raise SystemExit(0)

    df = pd.read_csv(args.file)
    df["#Gen"] = pd.to_numeric(df["#Gen"], errors="coerce")
    gen = args.gen if args.gen is not None else int(df["#Gen"].max())
    df = df[(df["#Gen"] == gen) & (df["Verdict"] == args.verdict)]
    if df.empty:
        raise SystemExit(f"No rows for #Gen={gen}, Verdict={args.verdict}")

    n_obs = df.groupby("Approach").size()
    print(f"#Gen={gen}, Verdict={args.verdict}, observations per approach:\n{n_obs}\n")

    results = []
    if args.per_problem:
        for pid, sub in df.groupby("ProblemID"):
            results.extend(compare(sub, args.baseline, args.metrics, scope=pid))
    else:
        results = compare(df, args.baseline, args.metrics, scope="pooled")

    print_results(results)
    pd.DataFrame(results).to_csv(STATS_PATH, index=False)
    print(f"\nSaved: {STATS_PATH}")
