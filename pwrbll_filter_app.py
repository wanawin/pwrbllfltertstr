# pwrbll_filter_app.py  — Streamlit Powerball Filter Runner (variant → itself)
from __future__ import annotations

import re
import io
from collections import Counter
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st

# =======================
# Runtime / safety config
# =======================
REVERSE_THRESHOLD = 0.75          # mark as reversal candidate when ≥ 75% eliminated
MAX_DETAILED_ROWS = 200_000       # hard cap to prevent OOM in Streamlit

# =======================
# UI helper
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
    Return [[3,18,22,27,33], ...]
    Works with dashes/commas/spaces; ignores trailing 'Powerball: xx'.
    """
    draws: List[List[int]] = []
    pat = re.compile(r'(?<!\d)(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})(?!\d)')
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            draws.append([int(g) for g in m.groups()])
    if reverse_input:
        draws.reverse()  # newest→oldest -> chronological
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
    return ["full", "ones", "tens"] + [f"pos{i}" for i in range(1, 6)] + [f"possum{i}" for i in range(1, 6)]

# =======================
# Hot / Cold / Due (variant-aware)
# =======================
def atoms_for_hotcold(window_draws: List[List[int]], variant: str) -> List[int]:
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
    else:  # full
        for d in window_draws:
            for n in d:
                atoms.extend([n // 10, n % 10])
    return atoms

def compute_hot_cold_due(history: List[List[int]], idx: int, variant: str,
                         hot_cold_window: int, due_window: int):
    start = max(0, idx - hot_cold_window)
    window = history[start:idx]
    cnt = Counter(atoms_for_hotcold(window, variant))
    if cnt:
        maxf, minf = max(cnt.values()), min(cnt.values())
        hot  = sorted([k for k, c in cnt.items() if c == maxf])
        cold = sorted([k for k, c in cnt.items() if c == minf])
    else:
        hot, cold = [], []
    recent = history[max(0, idx - due_window):idx]
    rdig = []
    for d in recent:
        for n in d:
            rdig.extend([n // 10, n % 10])
    due = sorted(list(set(range(10)) - set(rdig)))
    return hot, cold, due

# =======================
# Helpers available in expressions
# =======================
def spread_value(v) -> int:
    return (max(v) - min(v)) if isinstance(v, (list, tuple)) and len(v) == 5 else 0

def unique_digits_count(v) -> int:
    digits = []
    if isinstance(v, (list, tuple)):
        for n in v:
            digits.extend([n // 10, n % 10])
    else:
        for ch in str(int(v)):
            digits.append(int(ch))
    return len(set(digits))

def is_triple_draw(v) -> bool:
    digits = []
    if isinstance(v, (list, tuple)):
        for n in v:
            digits.extend([n // 10, n % 10])
    else:
        for ch in str(int(v)):
            digits.append(int(ch))
    return any(c >= 3 for c in Counter(digits).values())

def shared_digits_count(seed_draw, win_draw) -> int:
    s, w = [], []
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
    "min": min, "max": max, "abs": abs, "sum": sum, "len": len,
    "set": set, "any": any, "all": all, "sorted": sorted,
}

def layman_explanation(expr: str) -> str:
    repl = {
        "seed": "seed value", "winner": "winner value",
        "==": "equals", "<=": "is ≤", ">=": "is ≥", "<": "is <", ">": "is >",
        " and ": " AND ", " or ": " OR "
    }
    text = expr
    for k, v in repl.items():
        text = text.replace(k, v)
    return f"Eliminate if {text}"

# =======================
# Legacy token normalization
# =======================
LEGACY_MAP = [
    # common legacy names → runner names
    (r"\bcombo_sum\b", "winner"),
    (r"\bcombo_total\b", "winner"),
    (r"\bcombo\b", "winner"),
    (r"\bseed_sum\b", "seed"),
    (r"\bseed_total\b", "seed"),
    (r"\bones_total\b", "winner"),
    (r"\btens_total\b", "winner"),
    (r"\bfull_combo\b", "winner"),
    # boolean ops and punctuation
    (r"\b&\b", " and "),
    (r"\b\|\b", " or "),
    (r"“|”|‘|’", "\""),
    (r"≤", "<="),
    (r"≥", ">="),
    (r"≠", "!="),
    (r"–", "-"),
]

def normalize_expression(expr: str) -> str:
    if not isinstance(expr, str):
        return ""
    e = expr.strip()
    if not e:
        return ""
    e = e.strip("\"'")
    if "see prior" in e.lower() or "see conversation" in e.lower():
        return ""
    for pat, repl in LEGACY_MAP:
        e = re.sub(pat, repl, e, flags=re.IGNORECASE)
    e = re.sub(r"\bAND\b", "and", e)
    e = re.sub(r"\bOR\b",  "or",  e)
    return e

def load_filters_any(text: str) -> List[Tuple[str, str]]:
    """
    Accepts:
      - simple lines:   id, expression
      - Batch CSV/TXT with headers; picks 'expression'/'expr'/'applicable_if'
    """
    lines = text.splitlines()
    if not lines:
        return []

    header_like = ("," in lines[0]) and (("expression" in lines[0].lower()) or ("applicable" in lines[0].lower()))
    if header_like:
        df = pd.read_csv(io.StringIO(text))
        # choose expression column
        expr_col = None
        for c in df.columns:
            lc = c.lower()
            if lc in ("expression", "expr"):
                expr_col = c
                break
        if expr_col is None:
            for c in df.columns:
                if "applicable" in c.lower():
                    expr_col = c
                    break
        if expr_col is None:
            return []

        # choose id column or synthesize
        id_col = None
        for cand in ("id", "filter_id", "name"):
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            id_col = df.columns[0]

        out: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            fid = str(row[id_col])
            expr_raw = row[expr_col]
            expr = normalize_expression("" if pd.isna(expr_raw) else str(expr_raw))
            if expr:
                out.append((fid, expr))
        return out

    # fallback: id, expression lines
    out: List[Tuple[str, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2:
            fid, expr = parts[0], normalize_expression(parts[1])
            if expr:
                out.append((fid, expr))
    return out

# =======================
# Evaluator
# =======================
def evaluate(draws: List[List[int]], filters: List[Tuple[str, str]],
             hot_cold_window: int, due_window: int, write_detailed: bool) -> Dict[str, pd.DataFrame]:

    variants = all_variants()
    summary_rows, flagged_rows, reverse_rows = [], [], []
    detailed_rows: List[Dict[str, Any]] = []

    for fid, expr in filters:
        for v in variants:
            eliminated = tested = 0
            status = "OK"
            explanation = layman_explanation(expr) if expr else "Unparseable"

            for i in range(1, len(draws)):
                seed_draw, win_draw = draws[i - 1], draws[i]
                seed_val, win_val   = variant_value(seed_draw, v), variant_value(win_draw, v)
                hot, cold, due      = compute_hot_cold_due(draws, i, v, hot_cold_window, due_window)

                # Base context for expressions
                ctx = {
                    "seed": seed_val, "winner": win_val,
                    "hot": hot, "cold": cold, "due": due,
                    "seed_draw": seed_draw, "winner_draw": win_draw,
                    "variant_name": v,
                }

                # ---- Legacy-friendly aliases (so Batch CSVs run without editing) ----
                alias: Dict[str, Any] = {}

                # full / ones / tens totals
                alias.update({
                    "seed_full":               variant_value(seed_draw, "full"),
                    "winner_full":             variant_value(win_draw,  "full"),
                    "seed_ones_total":         variant_value(seed_draw, "ones"),
                    "winner_ones_total":       variant_value(win_draw,  "ones"),
                    "seed_tens_total":         variant_value(seed_draw, "tens"),
                    "winner_tens_total":       variant_value(win_draw,  "tens"),
                    "combo_sum":               variant_value(win_draw,  "full"),
                    "combo_total":             variant_value(win_draw,  "full"),
                    "combo_ones_total":        variant_value(win_draw,  "ones"),
                    "combo_tens_total":        variant_value(win_draw,  "tens"),
                })

                # positional numbers and digit-sums (pos1..pos5, possum1..possum5)
                for j in range(1, 6):
                    alias[f"seed_pos{j}_number"]     = variant_value(seed_draw, f"pos{j}")
                    alias[f"winner_pos{j}_number"]   = variant_value(win_draw,  f"pos{j}")
                    alias[f"seed_pos{j}_digitsum"]   = variant_value(seed_draw, f"possum{j}")
                    alias[f"winner_pos{j}_digitsum"] = variant_value(win_draw,  f"possum{j}")
                    # common legacy short names (default to winner side)
                    alias[f"pos{j}_number"]   = alias[f"winner_pos{j}_number"]
                    alias[f"pos{j}_digitsum"] = alias[f"winner_pos{j}_digitsum"]

                # expose exact legacy spellings used in some files
                alias.update({
                    "pos1_number": alias["winner_pos1_number"],
                    "pos2_number": alias["winner_pos2_number"],
                    "pos3_number": alias["winner_pos3_number"],
                    "pos4_number": alias["winner_pos4_number"],
                    "pos5_number": alias["winner_pos5_number"],
                    "pos1_digitsum": alias["winner_pos1_digitsum"],
                    "pos2_digitsum": alias["winner_pos2_digitsum"],
                    "pos3_digitsum": alias["winner_pos3_digitsum"],
                    "pos4_digitsum": alias["winner_pos4_digitsum"],
                    "pos5_digitsum": alias["winner_pos5_digitsum"],
                })

                ctx.update(alias)
                # ---- end aliases ----

                keep = True
                if not expr:
                    status = "FLAGGED"
                else:
                    try:
                        # Convention: expression True => eliminate
                        keep = not bool(eval(expr, ALLOWED_GLOBALS, ctx))
                    except Exception:
                        status = "FLAGGED"
                        keep = True

                eliminated += (0 if keep else 1)
                tested += 1

                if write_detailed and len(detailed_rows) < MAX_DETAILED_ROWS:
                    detailed_rows.append({
                        "filter_id": fid, "variant": v, "index": i,
                        "seed_value": seed_val, "winner_value": win_val,
                        "seed_draw": seed_draw, "winner_draw": win_draw,
                        "hot_digits": hot, "cold_digits": cold, "due_digits": due,
                        "eliminated": (not keep), "status": status,
                        "layman_explanation": explanation
                    })

            stat = f"{eliminated}/{tested}"
            summary_rows.append({
                "filter_id": fid, "variant": v, "eliminated": eliminated,
                "total": tested, "stat": stat, "status": status,
                "layman_explanation": explanation
            })
            if status == "FLAGGED":
                flagged_rows.append({
                    "filter_id": fid, "variant": v, "stat": stat,
                    "expression": expr, "layman_explanation": explanation
                })
            if tested > 0 and (eliminated / tested) >= REVERSE_THRESHOLD:
                reverse_rows.append({
                    "filter_id": fid, "variant": v, "eliminated": eliminated,
                    "total": tested, "stat": stat, "threshold": f"≥{int(REVERSE_THRESHOLD*100)}%",
                    "layman_explanation": explanation
                })

    return {
        "summary": pd.DataFrame(summary_rows),
        "flagged": pd.DataFrame(flagged_rows).drop_duplicates(subset=["filter_id", "variant"]),
        "reverse": pd.DataFrame(reverse_rows).drop_duplicates(subset=["filter_id", "variant"]),
        "detailed": pd.DataFrame(detailed_rows) if write_detailed else pd.DataFrame(),
    }

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Filter Runner", layout="wide")
st.title("🎰 Filter Runner (variant → itself)")

with st.sidebar:
    st.header("Settings")
    reverse_input   = st.checkbox("Input is newest → oldest (reverse to chronological)", value=True)
    hot_cold_window = st.number_input("Hot/Cold lookback (draws)", 1, 100, 6, 1)
    due_window      = st.number_input("Due lookback (draws)", 1, 20, 2, 1)
    write_detailed  = st.checkbox("Write detailed per-row results", value=False)
    st.caption(f"(Detailed rows capped at {MAX_DETAILED_ROWS:,} to avoid OOM.)")

    st.divider()
    st.caption("Upload files or leave blank to use repo files `pwrbll.txt` and `test 4pwrballfilters.txt`.")
    up_draws   = st.file_uploader("Upload pwrbll.txt", type=["txt", "csv"])
    up_filters = st.file_uploader("Upload filters (Batch CSV/TXT or id,expression)", type=["txt", "csv"])

run_btn = st.button("▶️ Run filters")

if run_btn:
    try:
        # Load draws
        if up_draws is not None:
            draws_text = up_draws.read().decode("utf-8", errors="ignore")
        else:
            with open("pwrbll.txt", encoding="utf-8") as f:
                draws_text = f.read()
        draws = load_draws_from_text(draws_text, reverse_input=reverse_input)
        if len(draws) < 2:
            st.error("Failed to parse at least 2 draws from pwrbll.txt.")
            st.stop()

        # Load filters
        if up_filters is not None:
            filters_text = up_filters.read().decode("utf-8", errors="ignore")
        else:
            with open("test 4pwrballfilters.txt", encoding="utf-8") as f:
                filters_text = f.read()
        filters = load_filters_any(filters_text)
        if not filters:
            st.error("No usable expressions found. Make sure there's an 'expression' column or id,expression lines.")
            st.stop()

        st.success(f"Parsed {len(draws)} draws and {len(filters)} filters. Running…")

        dfs = evaluate(draws, filters, hot_cold_window, due_window, write_detailed)

        summary_df  = dfs["summary"]
        flagged_df  = dfs["flagged"]
        reverse_df  = dfs["reverse"]
        detailed_df = dfs["detailed"]

        st.subheader("Results: Summary (per filter × variant)")
        st.dataframe(summary_df, use_container_width=True, height=400)
        download_pair(summary_df, "filter_results")

        st.subheader("Flagged Filters (need rewrite)")
        st.dataframe(flagged_df, use_container_width=True, height=260)
        download_pair(flagged_df, "flagged_filters")

        st.subheader("Reversal Candidates (≥ 75% eliminated)")
        st.dataframe(reverse_df, use_container_width=True, height=260)
        download_pair(reverse_df, "reversal_candidates")

        if write_detailed and not detailed_df.empty:
            st.subheader("Detailed (per row tested)")
            st.caption(f"Showing first 50,000 rows (of {len(detailed_df):,}).")
            st.dataframe(detailed_df.head(50_000), use_container_width=True, height=320)
            download_pair(detailed_df, "filter_results_detailed")

        st.info(f"Done. Summary rows: {len(summary_df):,} | Flagged: {len(flagged_df):,} | Reverse: {len(reverse_df):,}")

    except Exception as e:
        st.exception(e)
        st.stop()
