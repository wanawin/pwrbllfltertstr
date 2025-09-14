import pandas as pd
import os

# ========================
# Adjustable Parameters
# ========================
HOT_LOOKBACK = 6   # last N draws for hot/cold
DUE_LOOKBACK = 2   # last N draws for due
INPUT_FILE = "pwrbll.txt"

# ========================
# Load Draws
# ========================
def load_draws(filename):
    draws = []
    with open(filename) as f:
        for line in f:
            parts = line.strip().split()
            # Find the "03-18-22-27-33" block
            for token in parts:
                if "-" in token and token.count("-") == 4:
                    nums = [int(x) for x in token.split("-")]
                    draws.append(nums)
                    break
    return draws

# ========================
# Helpers
# ========================
def digit_sum(nums):
    return sum(nums)

def unique_digits(nums):
    return len(set(nums))

def spread(nums):
    return max(nums) - min(nums)

def is_triple(nums):
    return any(nums.count(x) == 3 for x in set(nums))

def hot_cold_due(draws, lookback=HOT_LOOKBACK, due_back=DUE_LOOKBACK):
    """Return hot, cold, due digits based on last N draws."""
    flat = [n for draw in draws[-lookback:] for n in draw]
    freq = pd.Series(flat).value_counts()

    hot = list(freq.nlargest(3).index)
    cold = list(freq.nsmallest(3).index)

    recent = set(n for draw in draws[-due_back:] for n in draw)
    all_nums = set(range(1, 70))  # Powerball pool max
    due = list(all_nums - recent)

    return hot, cold, due

# ========================
# Apply Filters
# ========================
def apply_filter(filter_expr, seed, combo, hot, cold, due):
    """Interpret filter expressions dynamically."""
    # Replace keywords with function calls
    expr = filter_expr
    expr = expr.replace("combo sum", str(digit_sum(combo)))
    expr = expr.replace("seed sum", str(digit_sum(seed)))
    expr = expr.replace("combo spread", str(spread(combo)))
    expr = expr.replace("combo unique", str(unique_digits(combo)))
    expr = expr.replace("is triple", str(is_triple(combo)))
    expr = expr.replace("hot", str(hot))
    expr = expr.replace("cold", str(cold))
    expr = expr.replace("due", str(due))
    expr = expr.replace("combo", str(combo))
    expr = expr.replace("seed", str(seed))

    try:
        return eval(expr)
    except Exception as e:
        return f"ERROR: {e}"

# ========================
# Main
# ========================
def main():
    draws = load_draws(INPUT_FILE)
    results = []

    # Example filter list — replace with your uploaded filters
    filters = [
        "digit_sum(combo) < 10 and len(set(combo) & set(seed)) >= 3",
        "digit_sum(combo) > 35",
        "is_triple(combo)"
    ]

    for i in range(1, len(draws)):
        seed = draws[i - 1]
        combo = draws[i]
        hot, cold, due = hot_cold_due(draws[:i])

        for f in filters:
            outcome = apply_filter(f, seed, combo, hot, cold, due)
            results.append({
                "Seed": seed,
                "Combo": combo,
                "Filter": f,
                "Outcome": outcome
            })

    df = pd.DataFrame(results)

    # Save CSV + TXT
    df.to_csv("filter_results.csv", index=False)
    with open("filter_results.txt", "w") as f:
        for _, row in df.iterrows():
            f.write(f"Seed: {row['Seed']} | Combo: {row['Combo']} | Filter: {row['Filter']} | Outcome: {row['Outcome']}\n")

    print("✅ Results saved to filter_results.csv and filter_results.txt")

if __name__ == "__main__":
    main()
