# pwrbll_filter_app.py
# -----------------------------------------------------------
# Streamlit app to run Powerball filters (variant-to-itself only)
# - Robust loader for pwrbll.txt (ignores dates/"Powerball: xx")
# - Reverse-input toggle (newest→oldest => chronological)
# - Precomputes 13 variants
# - Reads filters: one "filter_id, expression" per line (comments with '#')
# - Evaluates each filter across all variants (self-to-self, no cross)
# - Outputs:
#     * filter_results.(csv|txt)        - summary per filter+variant
#     * flagged_filters.(csv|txt)       - unparseable filters
#     * reversal_candidates.(csv|txt)   - ≥75% eliminated
#     * filter_results_detailed.(csv|txt) [optional] per-row debug
# -----------------------------------------------------------

from __future__ import annotations

import re
import io
from collections import Counter
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st

# =======================
# Helpers (UI)
# =======================
def download_pair(df: pd.DataFrame, base: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    txt_bytes = df.to_csv(index=False, sep="\t").encode("utf-8")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label=f"⬇️ Download {base}.csv",
            data=csv_bytes,
            file_name=f"{base}.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label=f"⬇️ Download {base}.txt",
            data=txt_bytes,
            file_name=f"{base}.txt",
            mime="text/plain",
        )

# =======================
# Robust draw loader
# =======================
def load_draws_from_text(text: str, reverse_input: bool = False) -> List[List[int]]:
    """
    Parse lines like:
      Sat, Aug 30, 2025   03-18-22-27-33,   Powerball: 17
    Return [[3,18,22,27,33], ...]
    Works with dashes/commas/spaces; ignores the Powerball number.
    """
    draws: List[List[int]] = []
    pat = re.compile(r'(?<!\d)(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})(?!\d)')
    for line in text.splitlines():
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
            n = five_positions(d)[j]
            atoms.extend([n // 10, n % 10])
    elif variant == "full":
        for d in window_draws:
            for n in d:
                atoms.extend([n // 10, n % 10])
    else:
        for d in window_draws:
            for n in d:
                atoms.extend([n // 10, n % 10])
    return atoms

def compute_hot_cold_due(
    history: List[List[int]],
    idx: int,
    variant: str,
    hot_cold_window: int,
    due_window: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    Compute hot/cold/due for a given seed index and variant.
    - hot: most frequent atoms in last HOT_COLD_WINDOW draws
    - cold: least frequent atoms in last HOT_COLD_WINDOW draws
    - due: atoms (0..9) not seen in last DUE_WINDOW draws (digit-level)
    """
    start = max(0, idx - hot_cold_window)
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
    recent_draws = history[max(0, idx - due_window):idx]
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
# Filter loader
# =======================
def load_filters_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Expects each non-empty line as:  filter_id, expression
    Ignores comment lines starting with #.
    """
    out: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            out.append((f"BAD_LINE:{line[:24]}", ""))
            continue
        fid, expr = parts
        out.append((fid, expr))
    return out

# =======================
# Evaluator
# =======================
def evaluate(
    draws: List[List[int]],
    filters: List[Tuple[str, str]],
    hot_cold_window: int,
    due_window: int,
    write_detailed: bool
) -> Dict[str, pd.DataFrame]:

    variants = ["full", "ones", "tens"] + [f"pos{i}" for i in range(1, 6)] + [f"possum{i}" for i in range(1, 6)]
    summary_rows = []
    flagged_rows = []
    reverse_rows = []
    detailed_rows = []

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

                hot, cold, due = compute_hot_cold_due(draws, i, variant, hot_cold_window, due_window)

                ctx = {
                    "seed": seed_val,
                    "winner": win_val,
                    "hot": hot,
                    "cold": cold,
                    "due": due,
                    "seed_draw": seed_draw,
                    "winner_draw": win_draw,
                }

                keep = True
                if not expr or "nan" in expr.lower():
                    status = "FLAGGED"
                else:
                    try:
                        # By convention: expression evaluates to True => eliminate
                        keep = not bool(eval(expr, ALLOWED_GLOBALS, ctx))
                    except Exception:
                        status = "FLAGGED"
                        keep = True

                eliminated += (0 if keep else 1)
                tested += 1

                if write_detailed:
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

            if status == "FLAGGED":
                flagged_rows.append({
                    "filter_id": fid,
                    "variant": variant,
                    "stat": stat,
                    "expression": expr,
                    "layman_explanation": explanation
                })

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

    summary_df = pd.DataFrame(summary_rows)
    flagged_df = pd.DataFrame(flagged_rows).drop_duplicates(subset=["filter_id", "variant"])
    reverse_df = pd.DataFrame(reverse_rows).drop_duplicates(subset=["filter_id", "variant"])
    detailed_df = pd.DataFrame(detailed_rows) if write_detailed else pd.DataFrame()

    return {
        "summary": summary_df,
        "flagged": flagged_df,
        "reverse": reverse_df,
        "detailed": detailed_df,
    }

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Powerball Filter Runner", layout="wide")
st.title("🎰 Powerball Filter Runner (variant → itself)")

with st.sidebar:
    st.header("Settings")
    reverse_input = st.checkbox("Input is newest → oldest (reverse to chronological)", value=False)
    hot_cold_window = st.number_input("Hot/Cold lookback (draws)", min_value=1, max_value=100, value=6, step=1)
    due_window = st.number_input("Due lookback (draws)", min_value=1, max_value=20, value=2, step=1)
    write_detailed = st.checkbox("Write detailed per-row results", value=True)

    st.markdown("---")
    st.caption("Upload files or leave empty to use repo files (same folder names).")
    up_draws = st.file_uploader("Upload pwrbll.txt", type=["txt"])
    up_filters = st.file_uploader("Upload filter list (id, expression per line)", type=["txt", "csv"])

run_btn = st.button("▶️ Run filters")

if run_btn:
    # Load pwrbll.txt
    if up_draws is not None:
        draws_text = up_draws.read().decode("utf-8", errors="ignore")
    else:
        try:
            with open("pwrbll.txt", encoding="utf-8") as f:
                draws_text = f.read()
        except FileNotFoundError:
            st.error("pwrbll.txt not found and no upload provided.")
            st.stop()

    draws = load_draws_from_text(draws_text, reverse_input=reverse_input)
    if len(draws) < 2:
        st.error("Failed to parse at least 2 draws from pwrbll.txt. Check file format.")
        st.stop()

    # Load filters
    if up_filters is not None:
        filters_text = up_filters.read().decode("utf-8", errors="ignore")
    else:
        try:
            with open("test 4pwrballfilters.txt", encoding="utf-8") as f:
                filters_text = f.read()
        except FileNotFoundError:
            st.error("Filter file not found and no upload provided.")
            st.stop()

    filters = load_filters_from_text(filters_text)
    if not filters:
        st.error("No filters parsed from filter file.")
        st.stop()

    st.success(f"Parsed {len(draws)} draws and {len(filters)} filters. Running…")

    dfs = evaluate(
        draws=draws,
        filters=filters,
        hot_cold_window=hot_cold_window,
        due_window=due_window,
        write_detailed=write_detailed
    )

    summary_df = dfs["summary"]
    flagged_df = dfs["flagged"]
    reverse_df = dfs["reverse"]
    detailed_df = dfs["detailed"]

    # Show & download
    st.subheader("Results: Summary (per filter × variant)")
    st.dataframe(summary_df, use_container_width=True, height=400)
    download_pair(summary_df, "filter_results")

    st.subheader("Flagged Filters (need rewrite)")
    st.dataframe(flagged_df, use_container_width=True, height=250)
    download_pair(flagged_df, "flagged_filters")

    st.subheader("Reversal Candidates (≥ 75% eliminated)")
    st.dataframe(reverse_df, use_container_width=True, height=250)
    download_pair(reverse_df, "reversal_candidates")

    if write_detailed and not detailed_df.empty:
        st.subheader("Detailed (per row tested)")
        st.dataframe(detailed_df.head(500), use_container_width=True, height=300)
        download_pair(detailed_df, "filter_results_detailed")

    st.info(f"Done. Total summary rows: {len(summary_df)} | Flagged: {len(flagged_df)} | Reverse: {len(reverse_df)}")
