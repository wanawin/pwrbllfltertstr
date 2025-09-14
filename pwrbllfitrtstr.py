# pwrbll_filter_runner.py
# -----------------------------------------------------------
# Powerball filter runner (variant-to-itself only)
# - Robust loader for pwrbll.txt (ignores dates / "Powerball: xx")
# - Reverse-input toggle (newest→oldest => chronological)
# - Precomputes 13 variants
# - Reads filters: one "filter_id, expression" per line
# - Evaluates each filter across all variants
# - Outputs:
#     * filter_results.(csv|txt)        - summary per filter+variant
#     * flagged_filters.(csv|txt)       - unparseable filters
#     * reversal_candidates.(csv|txt)   - ≥75% eliminated
#     * filter_results_detailed.(csv|txt) [optional] per-row debug
# -----------------------------------------------------------

import re
import csv
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

import pandas as pd

# =======================
# Config (tune here)
# =======================
INPUT_DRAWS_FILE   = "pwrbll.txt"
FILTER_LIST_FILE   = "test 4pwrballfilters.txt"

REVERSE_INPUT      = False   # True if pwrbll.txt is newest→oldest (will flip to chronological)
HOT_COLD_WINDOW    = 6       # last N draws for hot/cold
DUE_WINDOW         = 2       # last N draws for due
WRITE_DETAILED     = True    # also write per-row detailed results (bigger files)

OUT_FULL_BASE      = "filter_results"
OUT_FLAGGED_BASE   = "flagged_filters"
OUT_REVERSE_BASE   = "reversal_candidates"
OUT_DETAILED_BASE  = "filter_results_detailed"   # only if WRITE_DETAILED = True

# =======================
# Robust draw loader
# =======================
def load_draws(path: str, reverse_input: bool = False) -> List[List[int]]:
    """
    Parses pwrbll.txt lines of the form:
        Sat, Aug 30, 2025   03-18-22-27-33,   Powerball: 17
        Mon, Aug 25, 2025   16-19-34-37-64,   Powerball: 22
    and returns [[3,18,22,27,33], [16,19,34,37,64], ...]
    """
    draws: List[List[int]] = []
    pat = re.compile(r'(?<!\d)(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})(?!\d)')
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            nums = [int(g) for g in m.groups()]
            draws.append(nums)
    if reverse_input:
        draws.reverse()
    return draws

# =======================
# Variant precompute
# =======================
def five_positions(draw: List[int]) -> List[int]:
    return list(draw)

def ones_digits(draw: List[int]) -> List[int]:
    return [n % 10 for n in draw]

def tens_digits(draw: List[int]) -> List[int]:
    return [n // 10 for n in draw]

def pos_digit_sums(draw: List[int]) -> List[int]:
    return [(n // 10) + (n % 10) for n in draw]

def full_sum(draw: List[int]) -> int:
    # Sum of the 5 two-digit numbers (not digit-sum of all digits)
    return sum(draw)

def variant_value(draw: List[int], variant: str) -> Any:
    """
    Returns the scalar (for full/ones/tens) or positional value for a given variant key:
      - 'full'       -> int
      - 'ones'       -> sum of ones digits across 5 positions
      - 'tens'       -> sum of tens digits across 5 positions
      - 'pos1'..'pos5' -> the number at that position (int)
      - 'possum1'..'possum5' -> digit-sum (tens+ones) at that position (int)
    """
    if variant == "full":
        return full_sum(draw)
    if variant == "ones":
        return sum(ones_digits(draw))
    if variant == "tens":
        return sum(tens_digits(draw))

    if variant.startswith("possum"):
        idx = int(variant[-1]) - 1
        return pos_digit_sums(draw)[idx]

    if variant.startswith("pos"):
        idx = int(variant[-1]) - 1
        return five_positions(draw)[idx]

    raise ValueError(f"Unknown variant {variant}")

def all_variants() -> List[str]:
    base = ["full", "ones", "tens"]
    base += [f"pos{i}" for i in range(1, 6)]
    base += [f"possum{i}" for i in range(1, 6)]
    return base  # 13 variants

# =======================
# Hot/Cold/Due (variant-aware)
# =======================
def atoms_for_hotcold(window_draws: List[List[int]], variant: str) -> List[int]:
    """
    Returns the list of 'atoms' to count frequencies on, for a given variant.
    We use digits (0..9) for ones/tens/possum, and digits of the 5 numbers for 'full' and 'pos*'.
    """
    atoms: List[int] = []
    if variant == "ones":
        for d in window_draws:
            atoms.extend(ones_digits(d))
    elif variant == "tens":
        for d in window_draws:
            atoms.extend(tens_digits(d))
    elif variant.startswith("possum"):
        j = int(variant[-1]) - 1
        for d in window_draws:
            atoms.append(pos_digit_sums(d)[j])
    elif variant.startswith("pos"):
        j = int(variant[-1]) - 1
        for d in window_draws:
            # use digits of that position's number
            n = five_positions(d)[j]
            atoms.extend([n // 10, n % 10])
    elif variant == "full":
        for d in window_draws:
            for n in d:
                atoms.extend([n // 10, n % 10])
    else:
        # Fallback: all digits in draw
        for d in window_draws:
            for n in d:
                atoms.extend([n // 10, n % 10])
    return atoms

def compute_hot_cold_due(history: List[List[int]], idx: int, variant: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Compute hot/cold/due for a given seed index and variant.
    - hot: most frequent atoms in last HOT_COLD_WINDOW draws
    - cold: least frequent atoms in last HOT_COLD_WINDOW draws
    - due: atoms (0..9) not seen in last DUE_WINDOW draws (digit-level)
    """
    start = max(0, idx - HOT_COLD_WINDOW)
    window = history[start:idx]
    atoms = atoms_for_hotcold(window, variant)
    cnt = Counter(atoms)

    if cnt:
        maxf = max(cnt.values())
        minf = min(cnt.values())
        hot = sorted([a for a, c in cnt.items() if c == maxf])
        cold = sorted([a for a, c in cnt.items() if c == minf])
    else:
        hot, cold = [], []

    # due: digits 0..9 not present in last DUE_WINDOW draws (digit-level)
    recent_draws = history[max(0, idx - DUE_WINDOW):idx]
    recent_digits = []
    for d in recent_draws:
        for n in d:
            recent_digits.extend([n // 10, n % 10])
    due = sorted(list(set(range(10)) - set(recent_digits)))

    return hot, cold, due

# =======================
# Utility helpers exposed to expressions
# =======================
def spread_value(v) -> int:
    """If v is a 5-number draw, return number spread; if scalar, 0."""
    if isinstance(v, (list, tuple)) and len(v) == 5:
        return max(v) - min(v)
    return 0

def unique_digits_count(v) -> int:
    """Count unique digits in a 5-number draw or digits of a scalar."""
    digits = []
    if isinstance(v, (list, tuple)):
        for n in v:
            digits.extend([n // 10, n % 10])
    else:
        for d in str(int(v)):
            digits.append(int(d))
    return len(set(digits))

def is_triple_draw(v) -> bool:
    """True if any digit appears at least 3 times within the 5-number draw (digit-level)."""
    digits = []
    if isinstance(v, (list, tuple)):
        for n in v:
            digits.extend([n // 10, n % 10])
    else:
        for d in str(int(v)):
            digits.append(int(d))
    c = Counter(digits)
    return any(x >= 3 for x in c.values())

def shared_digits_count(seed_draw, win_draw) -> int:
    """Count shared digits (0..9) between seed and winner draws."""
    s = []
    w = []
    for n in seed_draw:
        s.extend([n // 10, n % 10])
    for n in win_draw:
        w.extend([n // 10, n % 10])
    return len(set(s) & set(w))

# =======================
# Safe-ish evaluator
# =======================
ALLOWED_GLOBALS = {
    "spread": spread_value,
    "unique_digits": unique_digits_count,
    "is_triple": is_triple_draw,
    "shared_digits": shared_digits_count,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "len": len,
    "set": set,
    "any": any,
    "all": all,
    "sorted": sorted,
}

def layman_explanation(expr: str) -> str:
    repl = {
        "seed": "seed value",
        "winner": "winner value",
        "==": "equals",
        "<=": "is ≤",
        ">=": "is ≥",
        "<": "is <",
        ">": "is >",
        " and ": " AND ",
        " or ": " OR "
    }
    text = expr
    for k, v in repl.items():
        text = text.replace(k, v)
    return f"Eliminate if {text}"

# =======================
# Filter file loader
# =======================
def load_filters(path: str) -> List[Tuple[str, str]]:
    """
    Expects each non-empty line as:  filter_id, expression
    Ignores comment lines starting with #.
    """
    out: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                # treat whole line as unparseable id
                out.append((f"BAD_LINE:{line[:24]}", ""))
                continue
            fid, expr = parts
            out.append((fid, expr))
    return out

# =======================
# Runner
# =======================
def run() -> None:
    draws = load_draws(INPUT_DRAWS_FILE, reverse_input=REVERSE_INPUT)
    if len(draws) < 2:
        raise RuntimeError("Not enough draws parsed from pwrbll.txt")

    variants = all_variants()
    total_pairs = len(draws) - 1

    filters = load_filters(FILTER_LIST_FILE)

    # summaries
    summary_rows: List[Dict[str, Any]] = []
    flagged_rows: List[Dict[str, Any]] = []
    reverse_rows: List[Dict[str, Any]] = []

    # optional detailed
    detailed_rows: List[Dict[str, Any]] = []

    for fid, expr in filters:
        for variant in variants:
            eliminated = 0
            tested = 0
            status = "OK"
            explanation = layman_explanation(expr) if expr else "Unparseable (empty expression)"

            for i in range(1, len(draws)):
                seed_draw = draws[i - 1]
                win_draw = draws[i]
                seed_val = variant_value(seed_draw, variant)
                win_val  = variant_value(win_draw, variant)

                # per-variant hot/cold/due
                hot, cold, due = compute_hot_cold_due(draws, i, variant)

                ctx = {
                    "seed": seed_val,
                    "winner": win_val,
                    "hot": hot,
                    "cold": cold,
                    "due": due,
                    # expose draws too (some filters use draw-level logic)
                    "seed_draw": seed_draw,
                    "winner_draw": win_draw,
                }

                keep = True
                if not expr or "nan" in expr.lower():
                    status = "FLAGGED"
                else:
                    try:
                        keep = not bool(eval(expr, ALLOWED_GLOBALS, ctx))  # expr True => eliminate
                    except Exception:
                        status = "FLAGGED"
                        keep = True  # treat as safe (no elimination) when broken

                eliminated += (0 if keep else 1)
                tested += 1

                if WRITE_DETAILED:
                    detailed_rows.append({
                        "filter_id": fid,
                        "variant": variant,
                        "index": i,
                        "seed_value": seed_val,
                        "winner_value": win_val,
                        "seed_draw": seed_draw,
                        "winner_draw": win_draw,
                        "hot_digits": hot,
                        "cold_digits": cold,
                        "due_digits": due,
                        "eliminated": (not keep),
                        "status": status,
                        "layman_explanation": explanation
                    })

            stat = f"{eliminated}/{tested}"
            summary_rows.append({
                "filter_id": fid,
                "variant": variant,
                "eliminated": eliminated,
                "total": tested,
                "stat": stat,
                "status": status,
                "layman_explanation": explanation
            })

            # track flagged
            if status == "FLAGGED":
                flagged_rows.append({
                    "filter_id": fid,
                    "variant": variant,
                    "stat": stat,
                    "expression": expr,
                    "layman_explanation": explanation
                })

            # reverse candidate
            if tested > 0 and (eliminated / tested) >= 0.75:
                reverse_rows.append({
                    "filter_id": fid,
                    "variant": variant,
                    "eliminated": eliminated,
                    "total": tested,
                    "stat": stat,
                    "threshold": "≥75%",
                    "layman_explanation": explanation
                })

    # ---------- write helpers ----------
    def save_both(df: pd.DataFrame, base: str) -> None:
        csv_path = f"{base}.csv"
        txt_path = f"{base}.txt"
        df.to_csv(csv_path, index=False)
        df.to_csv(txt_path, index=False, sep="\t")
        print(f"Saved {csv_path} and {txt_path}")

    # ---------- write outputs ----------
    summary_df = pd.DataFrame(summary_rows)
    save_both(summary_df, OUT_FULL_BASE)

    flagged_df = pd.DataFrame(flagged_rows).drop_duplicates(subset=["filter_id", "variant"])
    save_both(flagged_df, OUT_FLAGGED_BASE)

    reverse_df = pd.DataFrame(reverse_rows).drop_duplicates(subset=["filter_id", "variant"])
    save_both(reverse_df, OUT_REVERSE_BASE)

    if WRITE_DETAILED:
        detailed_df = pd.DataFrame(detailed_rows)
        save_both(detailed_df, OUT_DETAILED_BASE)

    # quick console summary
    print("\n--- Summary ---")
    print(f"Total filter+variant rows: {len(summary_df)}")
    print(f"FLAGGED rows: {len(flagged_df)}")
    print(f"REVERSE candidates: {len(reverse_df)}")

# =======================
# Entry point
# =======================
if __name__ == "__main__":
    run()
