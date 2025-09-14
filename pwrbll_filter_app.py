# pwrbll_filter_app.py
from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st

# =======================
# App / runtime config
# =======================
REVERSE_THRESHOLD = 0.75   # mark as reversal candidate when ≥ 75% eliminated
MAX_DETAILED_ROWS = 200_000  # hard cap to prevent OOM in Streamlit

# =======================
# Small UI helper
# =======================
def download_pair(df: pd.DataFrame, base: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    txt_bytes = df.to_csv(index=False, sep="\t").encode("utf-8")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(f"⬇️ Download {base}.csv", csv_bytes, f"{base}.csv", "text/csv")
    with c2:
        st.download_button(f"⬇️ Download {base}.txt", txt_bytes, f"{base}.txt", "text/plain")

# =======================
# Robust draw loader
# =======================
def load_draws_from_text(text: str, reverse_input: bool = False) -> List[List[int]]:
    """
    Parse lines like:
      Sat, Aug 30, 2025   03-18-22-27-33,   Powerball: 17
    Returns [[3,18,22,27,33], ...]
    Works with dashes/commas/spaces; ignores the trailing Powerball number.
    """
    draws: List[List[int]] = []
    pat = re.compile(r'(?<!\d)(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})(?!\d)')
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            nums = [int(g) for g in m.groups()]  # e.g. '03-18-22-27-33,' -> [3,18,22,27,33]
            draws.append(nums)
    if reverse_input:
        draws.reverse()  # newest→oldest to chronological
    return draws

# =======================
# Variants (13)
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
    return ["full", "ones", "tens"] + [f"pos{i}" for i in range(1,6)] + [f"possum{i}" for i in range(1,6)]

# =======================
# Hot / Cold / Due (variant-aware)
# =======================
def atoms_for_hotcold(window_draws: List[List[int]], variant: str) -> List[int]:
    atoms: List[int] = []
    if variant == "ones":
        for d in window_draws: atoms.extend(ones_digits(d))
    elif variant == "tens":
        for d in window_draws: atoms.extend(tens_digits(d))
    elif variant.startswith("possum"):
        j = int(variant[-1]) - 1
        for d in window_draws: atoms.append(pos_digit_sums(d)[j])
    elif variant.startswith("pos"):
        j = int(variant[-1]) - 1
        for d in window_draws:
            n = five_positions(d)[j]
            atoms.extend([n // 10, n % 10])
    elif variant == "full":
        for d in window_draws:
            for n in d: atoms.extend([n // 10, n % 10])
    else:
        for d in window_draws:
            for n in d: atoms.extend([n // 10, n % 10])
    return atoms

def compute_hot_cold_due(
    history: List[List[int]], idx: int, variant: str, hot_cold_window: int, due_window: int
) -> Tuple[List[int], List[int], List[int]]:
    start = max(0, idx - hot_cold_window)
    window = history[start:idx]
    atoms = atoms_for_hotcold(window, variant)
    cnt = Counter(atoms)

    if cnt:
        maxf, minf = max(cnt.values()), min(cnt.values())
        hot  = sorted([a for a, c in cnt.items() if c == maxf])
        cold = sorted([a for a, c in cnt.items() if c == minf])
    else:
        hot, cold = [], []

    recent_draws = history[max(0, idx - due_window):idx]
    recent_digits = []
    for d in recent_draws:
        for n in d:
            recent_digits.extend([n // 10, n % 10])
    due = sorted(list(set(range(10)) - set(recent_digits)))
    return hot, cold, due

# =======================
# Helpers available in expressions
# =======================
def spread_value(v) -> int:
    if isinstance(v, (list, tuple)) and len(v) == 5:
        return max(v) - min(v)
    return 0

def unique_digits_count(v) -> int:
    digits = []
    if isinstance(v, (list, tuple)):
        for n in v: digits.extend([n // 10, n % 10])
    else:
        for d in str(int(v)): digits.append(int(d))
    return len(set(digits))

def is_triple_draw(v) -> bool:
    digits = []
    if isinstance(v, (list, tuple)):
        for n in v: digits.extend([n // 10, n % 10])
    else:
        for d in str(int(v)): digits.append(int(d))
    return any(c >= 3 for c in Counter(digits).values())

def shared_digits_count(seed_draw, win_draw) -> int:
    s, w = [], []
    for n in seed_draw: s.extend([n // 10, n % 10])
    for n in win_draw:  w.extend([n // 10, n % 10])
    return len(set(s) & set(w))

ALLOWED_GLOBALS = {
    "spread": spread_value,
    "unique_digits": unique_digits_count,
    "is_triple": is_triple_draw,
    "shared_digits": shared_digits_count,
    "min": min, "max": max, "abs": abs, "sum": sum,
    "len": len, "set": set, "any": any, "all": all, "sorted": sorted,
}

def layman_explanation(expr: str) -> str:
    repl = {
        "seed": "seed value",
        "winner": "winner value",
        "==": "equals", "<=": "is ≤", ">=": "is ≥", "<": "is <", ">": "is >",
        " and ": " AND ", " or ": " OR "
    }
    text = expr
    for k, v in repl.items(): text = text.replace(k, v)
    return f"Eliminate if {text}"

# =======================
# Filter loader
# =======================
def load_filters_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Each non-empty line: filter_id, expression
    Lines starting with # are ignored. Split only on first comma.
    """
    out: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            out.append((f"BAD_LINE:{line[:24]}", ""))
        else:
            out.append((parts[0], parts[1]))
    return out

# =======================
# Evaluator (memory safe)
# =======================
def evaluate(
    draws: List[List[int]],
    filters: List[Tuple[str, str]],
    hot_cold_window: int,
    due_window: int,
    write_detailed: bool
) -> Dict[str, pd.DataFrame]:

    variants = all_variants()
    summary_rows, flagged_rows, reverse_rows = [], [], []
    detailed_rows: List[Dict[str, Any]] = []

    for fid, expr in filters:
        for variant in variants:
            eliminated, tested = 0, 0
            status = "OK"
            explanation = layman_explanation(expr) if expr else "Unparseable (empty expression)"

            for i in range(1, len(draws)):
                seed_draw, win_draw = draws[i-1], draws[i]
                seed_val, win_val   = variant_value(seed_draw, variant), variant_value(win_draw, variant)
                hot, cold, due      = compute_hot_cold_due(draws, i, variant, hot_cold_window, due_window)

                ctx = {
                    "seed": seed_val, "winner": win_val,
                    "hot": hot, "cold": cold, "due": due,
                    "seed_draw": seed_draw, "winner_draw": win_draw,
                }

                keep = True
                if not expr or "nan" in expr.lower():
                    status = "FLAGGED"
                else:
                    try:
                        # By convention: expression True => eliminate
                        keep = not bool(eval(expr, ALLOWED_GLOBALS, ctx))
                    except Exception:
                        status = "FLAGGED"
                        keep = True

                eliminated += (0 if keep else 1)
                tested += 1

                if write_detailed and len(detailed_rows) < MAX_DETAILED_ROWS:
                    detailed_rows.append({
                        "filter_id": fid, "variant": variant, "index": i,
                        "seed_value": seed_val, "winner_value": win_val,
                        "seed_draw": seed_draw, "winner_draw": win_draw,
                        "hot_digits": hot, "cold_digits": cold, "due_digits": due,
                        "eliminated": (not keep), "status": status,
                        "layman_explanation": explanation
                    })

            stat = f"{eliminated}/{tested}"
            summary_rows.append({
                "filter_id": fid, "variant": variant,
                "eliminated": eliminated, "total": tested,
                "stat": stat, "status": status,
                "layman_explanation": explanation
            })

            if status == "FLAGGED":
                flagged_rows.append({
                    "filter_id": fid, "variant": variant, "stat": stat,
                    "expression": expr, "layman_explanation": explanation
                })

            if tested > 0 and (eliminated / tested) >= REVERSE_THRESHOLD:
                reverse_rows.append({
                    "filter_id": fid, "variant": variant,
                    "eliminated": eliminated, "total": tested, "stat": stat,
                    "threshold": f"≥{int(REVERSE_THRESHOLD*100)}%",
                    "layman_explanation": explanation
                })

    return {
        "summary": pd.DataFrame(summary_rows),
        "flagged": pd.DataFrame(flagged_rows).drop_duplicates(subset=["filter_id","variant"]),
        "reverse": pd.DataFrame(reverse_rows).drop_duplicates(subset=["filter_id","variant"]),
        "detailed": pd.DataFrame(detailed_rows) if write_detailed else pd.DataFrame(),
    }

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Powerball Filter Runner", layout="wide")
st.title("🎰 Powerball Filter Runner (variant → itself)")

with st.sidebar:
    st.header("Settings")
    reverse_input   = st.checkbox("Input is newest → oldest (reverse to chronological)", value=False)
    hot_cold_window = st.number_input("Hot/Cold lookback (draws)", 1, 100, 6, 1)
    due_window      = st.number_input("Due lookback (draws)", 1, 20, 2, 1)
    write_detailed  = st.checkbox("Write detailed per-row results", value=False)
    st.caption(f"(Detailed rows are capped at {MAX_DETAILED_ROWS:,} to avoid OOM.)")

    st.divider()
    st.caption("Upload files or leave blank to use repo files `pwrbll.txt` and `test 4pwrballfilters.txt`.")
    up_draws   = st.file_uploader("Upload pwrbll.txt", type=["txt"])
    up_filters = st.file_uploader("Upload filter list (id, expression per line)", type=["txt","csv"])

run_btn = st.button("▶️ Run filters")

if run_btn:
    try:
        # ----- Load draws -----
        if up_draws is not None:
            draws_text = up_draws.read().decode("utf-8", errors="ignore")
        else:
            with open("pwrbll.txt", encoding="utf-8") as f:
                draws_text = f.read()

        draws = load_draws_from_text(draws_text, reverse_input=reverse_input)
        if len(draws) < 2:
            st.error("Failed to parse at least 2 draws from pwrbll.txt. Check file format.")
            st.stop()

        # ----- Load filters -----
        if up_filters is not None:
            filters_text = up_filters.read().decode("utf-8", errors="ignore")
        else:
            with open("test 4pwrballfilters.txt", encoding="utf-8") as f:
                filters_text = f.read()

        filters = load_filters_from_text(filters_text)
        if not filters:
            st.error("No filters parsed from filter file.")
            st.stop()

        st.success(f"Parsed {len(draws)} draws and {len(filters)} filters. Running…")

        # ----- Evaluate -----
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

        # ----- Show & download -----
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
            st.caption(f"Showing first 50,000 rows (of {len(detailed_df):,}). "
                       "Disable detailed mode for faster runs.")
            st.dataframe(detailed_df.head(50_000), use_container_width=True, height=320)
            download_pair(detailed_df, "filter_results_detailed")

        st.info(f"Done. Summary rows: {len(summary_df):,} | Flagged: {len(flagged_df):,} | Reverse: {len(reverse_df):,}")

    except Exception as e:
        st.exception(e)
        st.stop()
