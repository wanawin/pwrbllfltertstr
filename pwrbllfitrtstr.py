import pandas as pd
from collections import Counter

# ============================================================
# 🔧 Adjustable Parameters
# ============================================================
HOT_COLD_WINDOW = 6   # last N draws for hot/cold calculation
DUE_WINDOW = 2        # last N draws for due calculation
INPUT_FILE = "pwrbll.txt"
FILTER_FILE = "test 4pwrballfilters.txt"

OUT_FULL = "filter_results"
OUT_FLAGGED = "flagged_filters"
OUT_REVERSAL = "reversal_candidates"

# ============================================================
# 🔹 Helpers
# ============================================================
def load_draws(path):
    draws = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                draws.append([int(x) for x in parts[:5]])
    return draws

def compute_hot_cold_due(history, idx):
    start = max(0, idx - HOT_COLD_WINDOW)
    window = history[start:idx]
    flat = [d for draw in window for d in draw]

    counts = Counter(flat)
    if not counts:
        return [], [], []

    max_freq = max(counts.values())
    min_freq = min(counts.values())
    hot = [d for d, c in counts.items() if c == max_freq]
    cold = [d for d, c in counts.items() if c == min_freq]

    recent = set([d for draw in history[max(0, idx-DUE_WINDOW):idx] for d in draw])
    all_digits = set(range(10))
    due = list(all_digits - recent)

    return hot, cold, due

def variant_sums(draw):
    ones = [d % 10 for d in draw]
    tens = [d // 10 for d in draw]
    pos_nums = draw
    pos_sums = [t + o for t, o in zip(tens, ones)]
    return {
        "full": sum(draw),
        "ones": sum(ones),
        "tens": sum(tens),
        "pos": pos_nums,
        "possum": pos_sums
    }

def layman_explanation(expr: str) -> str:
    """Turn filter expression into a human-readable explanation."""
    repl = {
        "seed": "seed sum",
        "winner": "winner sum",
        "==": "equals",
        "<=": "is less than or equal to",
        ">=": "is greater than or equal to",
        "<": "is less than",
        ">": "is greater than",
        " and ": " AND ",
        " or ": " OR "
    }
    result = expr
    for k, v in repl.items():
        result = result.replace(k, v)
    return f"Eliminate if {result}"

# ============================================================
# 🔹 Core Evaluation
# ============================================================
def evaluate_filters(draws, filters):
    results = []
    for i in range(1, len(draws)):
        seed = draws[i-1]
        winner = draws[i]

        seed_vars = variant_sums(seed)
        win_vars = variant_sums(winner)

        hot, cold, due = compute_hot_cold_due(draws, i)

        for f in filters:
            fid, expr = f["id"], f["expr"]
            explanation = layman_explanation(expr)

            for variant in ["full", "ones", "tens"]:
                try:
                    s_val, w_val = seed_vars[variant], win_vars[variant]
                    keep = eval(expr, {}, {"seed": s_val, "winner": w_val})
                    status = "OK"
                except Exception:
                    keep = True
                    status = "FLAGGED"

                results.append({
                    "filter_id": fid,
                    "variant": variant,
                    "eliminated": not keep,
                    "hot_digits": hot,
                    "cold_digits": cold,
                    "due_digits": due,
                    "status": status,
                    "layman_explanation": explanation
                })

            for j in range(5):
                for label, key in [(f"pos{j+1}", "pos"), (f"possum{j+1}", "possum")]:
                    try:
                        s_val, w_val = seed_vars[key][j], win_vars[key][j]
                        keep = eval(expr, {}, {"seed": s_val, "winner": w_val})
                        status = "OK"
                    except Exception:
                        keep = True
                        status = "FLAGGED"

                    results.append({
                        "filter_id": fid,
                        "variant": label,
                        "eliminated": not keep,
                        "hot_digits": hot,
                        "cold_digits": cold,
                        "due_digits": due,
                        "status": status,
                        "layman_explanation": explanation
                    })
    return pd.DataFrame(results)

# ============================================================
# 🔹 Save Helper
# ============================================================
def save_both(df, base):
    csv_path = f"{base}.csv"
    txt_path = f"{base}.txt"
    df.to_csv(csv_path, index=False)
    df.to_csv(txt_path, index=False, sep="\t")
    print(f"Saved {csv_path} and {txt_path}")

# ============================================================
# 🔹 Main
# ============================================================
def main():
    draws = load_draws(INPUT_FILE)

    filters = []
    with open(FILTER_FILE) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                filters.append({"id": parts[0], "expr": parts[1]})

    df = evaluate_filters(draws, filters)

    # Save full results
    save_both(df, OUT_FULL)

    # Flagged
    flagged = df[df["status"] == "FLAGGED"].drop_duplicates(["filter_id","variant"])
    save_both(flagged, OUT_FLAGGED)

    # Reversal candidates (≥75%)
    summary = (
        df.groupby(["filter_id", "variant", "status", "layman_explanation"])
          .eliminated.agg(["sum","count"])
          .reset_index()
    )
    summary["pct_eliminated"] = summary["sum"] / summary["count"]
    reversal = summary[summary["pct_eliminated"] >= 0.75]
    save_both(reversal, OUT_REVERSAL)

if __name__ == "__main__":
    main()
