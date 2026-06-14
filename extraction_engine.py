"""
Document-agnostic structured extraction + deterministic analytics.

This module turns OCR'd pages into a verifiable table of records, then computes
all aggregates/audits with pandas (NOT the LLM) so numbers can never be
hallucinated. The LLM is only used to *read* each page into structured rows and,
later, to *format* the pre-computed numbers — it never invents or recomputes them.
"""

import io
import json
import re
from collections import defaultdict

import pandas as pd

from llm_provider import ask_llm


# Metadata keys the extractor attaches to every record (prefixed with "_").
META_KEYS = (
    "_doc", "_page", "_position", "_unclear_fields", "_handwritten_fields",
    "_is_duplicate_copy", "_has_signature", "_notes", "_is_duplicate",
    "_duplicate_group", "_duplicate_count",
)


# ---------------------------------------------------------------------------
# 1. Per-page LLM extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a precise data-extraction engine. Extract structured \
records from the OCR text of ONE page of a scanned document.

ABSOLUTE RULES — follow exactly:
- Extract ONLY values that are actually present and readable in the text.
- NEVER guess, infer, autocomplete, translate, or fabricate any value.
- If a value is missing, unreadable, partial, or uncertain, set it to null.
- Do not invent names, dates, numbers, amounts, quantities, or IDs.
- Preserve numbers and codes exactly as written (do not "fix" them).

WHAT TO EXTRACT:
- Identify the main repeating record type on the page (e.g. receipt, invoice line,
  transaction, employee, item, row). Extract EACH instance as one record.
- If the page has a single record (one form), return one record.
- If the page has no extractable structured data, return an empty list.
- Use consistent snake_case field names across all records.

For EACH record, also include these metadata fields:
- "_position": 1-based position of the record on the page, top to bottom (integer)
- "_unclear_fields": list of field names whose values were hard to read / low
  confidence (those fields MUST be null in the record)
- "_handwritten_fields": list of field names that appear handwritten rather than
  printed (only if determinable; else [])
- "_is_duplicate_copy": true if the record is explicitly marked as a duplicate /
  customer copy / copy, else false
- "_has_signature": true / false / null — whether a signature is present, only if
  determinable from the text
- "_notes": short note for anything suspicious, inconsistent, or crossed-out;
  else null

Return ONLY a JSON object of the form: {"records": [ { ... }, ... ]}
No markdown, no code fences, no explanation.

OCR text of page %(page_no)s:
\"\"\"
%(page_text)s
\"\"\"
JSON:"""


def _safe_json_obj(raw: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def extract_records_from_pages(pages: list, doc_id: str, progress_cb=None) -> list:
    """
    Run per-page structured extraction over a document's OCR pages.

    pages       : list of {"page": int, "text": str, ...} (the ocr_json["pages"])
    doc_id      : document identifier, tagged onto every record as "_doc"
    progress_cb : optional callable(done, total, page_no) for UI progress.

    Returns a flat list of record dicts, each tagged with _doc and _page.
    """
    records: list = []
    total = len(pages)

    for idx, page in enumerate(pages, start=1):
        page_no = page.get("page", idx)
        page_text = (page.get("text") or "").strip()

        if progress_cb:
            progress_cb(idx - 1, total, page_no)

        if not page_text:
            continue

        prompt = _EXTRACTION_PROMPT % {"page_no": page_no, "page_text": page_text[:6000]}

        try:
            parsed = _safe_json_obj(ask_llm(prompt))
        except Exception:
            parsed = None

        if not parsed:
            continue

        page_records = parsed.get("records") or []
        for pos, rec in enumerate(page_records, start=1):
            if not isinstance(rec, dict):
                continue
            rec.setdefault("_position", pos)
            rec["_doc"] = doc_id
            rec["_page"] = page_no
            records.append(rec)

    if progress_cb:
        progress_cb(total, total, None)

    return flag_duplicates(records)


# ---------------------------------------------------------------------------
# 2. Duplicate detection (deterministic — across all pages)
# ---------------------------------------------------------------------------

def _content_signature(rec: dict) -> str:
    """Signature from content fields only (ignore metadata) for dedup matching."""
    items = []
    for k, v in rec.items():
        if k.startswith("_"):
            continue
        if v is None or str(v).strip() == "":
            continue
        items.append((k.lower(), str(v).strip().lower()))
    items.sort()
    return json.dumps(items, ensure_ascii=False)


def flag_duplicates(records: list) -> list:
    """
    Group records with identical content signatures and annotate each with
    _is_duplicate / _duplicate_group / _duplicate_count. Idempotent.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        sig = _content_signature(rec)
        if sig == "[]":          # entirely empty record — don't group
            continue
        groups[sig].append(i)

    gid = 0
    for sig, idxs in groups.items():
        if len(idxs) > 1:
            gid += 1
            for i in idxs:
                records[i]["_is_duplicate"] = True
                records[i]["_duplicate_group"] = gid
                records[i]["_duplicate_count"] = len(idxs)

    for rec in records:
        rec.setdefault("_is_duplicate", False)
        rec.setdefault("_duplicate_group", None)
        rec.setdefault("_duplicate_count", 1)

    return records


# ---------------------------------------------------------------------------
# 3. Records → DataFrame
# ---------------------------------------------------------------------------

def records_to_dataframe(records: list) -> pd.DataFrame:
    """
    Build a display/export DataFrame. Content columns first, metadata last.
    List-valued fields are joined to comma strings for spreadsheet friendliness.
    """
    if not records:
        return pd.DataFrame()

    content_keys, meta_present = [], []
    for rec in records:
        for k in rec.keys():
            if k.startswith("_"):
                if k not in meta_present:
                    meta_present.append(k)
            elif k not in content_keys:
                content_keys.append(k)

    ordered = content_keys + [k for k in META_KEYS if k in meta_present]
    rows = []
    for rec in records:
        row = {}
        for k in ordered:
            v = rec.get(k)
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v) if v else ""
            elif isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            row[k] = v
        rows.append(row)

    return pd.DataFrame(rows, columns=ordered)


# ---------------------------------------------------------------------------
# 4. Numeric coercion helpers
# ---------------------------------------------------------------------------

# Strip only genuine currency/format tokens — NOT letters, so "A101" or "KW-1234"
# never masquerade as a number.
_CURRENCY_RE = re.compile(
    r"[,\s$£€%]|kwd|kd|usd|sar|aed|qar|bhd|omr|eur|gbp|inr|rs",
    re.IGNORECASE,
)
_PURE_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?$")


def _to_number(val):
    """Parse a value to float ONLY if it is genuinely numeric (after removing
    currency symbols/commas). Letters or internal dashes (IDs, plates) -> None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = _CURRENCY_RE.sub("", s).strip()
    if _PURE_NUMBER_RE.fullmatch(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a column to numbers using the strict parser above."""
    return s.map(_to_number).astype("float64")


_DATE_PATTERNS = ("date", "day")
_VEHICLE_PATTERNS = ("vehicle", "plate", "car", "truck", "asset", "lorry", "bus")
_ODO_PATTERNS = ("odometer", "odo", "km", "kilometer", "kilometre", "mileage", "meter", "reading")
# Identifiers expected to be UNIQUE (receipt/invoice numbers). Repeated values here
# are an audit issue — unlike vehicle numbers, which repeat legitimately.
_UNIQUE_ID_HINTS = ("receipt", "invoice", "bill", "txn", "transaction", "voucher", "serial", "ref")


def _name_matches(col: str, patterns) -> bool:
    c = col.lower()
    return any(p in c for p in patterns)


def _is_unique_id_col(col: str) -> bool:
    """A column meant to hold unique identifiers (receipt/invoice no). Vehicle/
    plate columns are explicitly excluded — their values repeat by design."""
    c = col.lower()
    if _name_matches(col, _VEHICLE_PATTERNS):
        return False
    if any(h in c for h in _UNIQUE_ID_HINTS):
        return True
    return c in ("no", "id", "no.") or c.endswith("_no") or c.endswith("_id") or "number" in c


def _is_identifier_col(col: str) -> bool:
    """Any identifier (unique id OR vehicle/plate) — never summed as a measure."""
    return _is_unique_id_col(col) or _name_matches(col, _VEHICLE_PATTERNS)


def _numeric_columns(df: pd.DataFrame, threshold: float = 0.6) -> list:
    """Columns where >= threshold of non-null values are genuine numbers AND the
    column is not an identifier (summing receipt/vehicle numbers is meaningless)."""
    out = []
    for col in df.columns:
        if col.startswith("_") or _is_identifier_col(col):
            continue
        non_null = df[col].astype(str).str.strip().replace("", None).dropna()
        if non_null.empty:
            continue
        if _numeric_series(non_null).notna().mean() >= threshold:
            out.append(col)
    return out


# ---------------------------------------------------------------------------
# 5. Deterministic data profile  (the authoritative numbers)
# ---------------------------------------------------------------------------

def compute_data_profile(df: pd.DataFrame, max_groups: int = 60) -> str:
    """
    Produce a markdown profile of EXACT, pandas-computed aggregates.
    This is handed to the LLM as authoritative — it must not recompute.
    Covers: row/null counts, numeric totals, group-by sums for category columns,
    date-wise sums, duplicates, and per-vehicle odometer spans.
    """
    if df is None or df.empty:
        return "No structured records available."

    content_cols = [c for c in df.columns if not c.startswith("_")]
    num_cols = _numeric_columns(df)
    out = [f"Total records: **{len(df)}**", f"Columns: {', '.join(content_cols)}", ""]

    # ── Missing / completeness per column ──
    out.append("**Completeness (non-null counts):**")
    for col in content_cols:
        non_null = df[col].astype(str).str.strip().replace("", None).notna().sum()
        out.append(f"- {col}: {int(non_null)}/{len(df)} present, {len(df) - int(non_null)} missing")
    out.append("")

    # ── Numeric totals ──
    if num_cols:
        out.append("**Numeric column totals (exact):**")
        for col in num_cols:
            nums = _numeric_series(df[col])
            out.append(
                f"- {col}: sum={nums.sum():,.3f}, mean={nums.mean():,.3f}, "
                f"min={nums.min():,.3f}, max={nums.max():,.3f}, count={int(nums.notna().sum())}"
            )
        out.append("")

    # ── Group-by category sums ──
    # Skip unique-id columns (grouping by a unique key is pointless) but KEEP
    # vehicle/plate columns — "total by vehicle" is exactly what's wanted.
    cat_cols = []
    for col in content_cols:
        if col in num_cols or _is_unique_id_col(col):
            continue
        nun = df[col].astype(str).str.strip().replace("", None).dropna().nunique()
        if _name_matches(col, _DATE_PATTERNS) or 1 < nun <= 50:
            cat_cols.append(col)

    for cat in cat_cols:
        key = df[cat].astype(str).str.strip().replace("", "(blank)")
        out.append(f"**By {cat}:**")
        counts = key.value_counts().head(max_groups)
        for val, cnt in counts.items():
            line = f"- {val}: {cnt} record(s)"
            for ncol in num_cols:
                grp = _numeric_series(df[ncol]).groupby(key).sum()
                if val in grp.index:
                    line += f", total {ncol}={grp.loc[val]:,.3f}"
            out.append(line)
        out.append("")

    # ── Duplicates ──
    if "_is_duplicate" in df.columns:
        dup_rows = df[df["_is_duplicate"] == True]  # noqa: E712
        n_groups = dup_rows["_duplicate_group"].nunique() if not dup_rows.empty else 0
        out.append(f"**Duplicates:** {len(dup_rows)} record(s) across {n_groups} duplicate group(s).")
        out.append("")

    # ── Per-vehicle odometer span (km travelled) ──
    veh_col = next((c for c in content_cols if _name_matches(c, _VEHICLE_PATTERNS)), None)
    odo_col = next((c for c in content_cols if _name_matches(c, _ODO_PATTERNS) and c in num_cols), None)
    if veh_col and odo_col:
        out.append(f"**Odometer span by {veh_col} (using {odo_col}):**")
        tmp = pd.DataFrame({
            "veh": df[veh_col].astype(str).str.strip().replace("", None),
            "odo": _numeric_series(df[odo_col]),
        }).dropna()
        if not tmp.empty:
            grp = tmp.groupby("veh")["odo"].agg(["min", "max", "count"])
            grp["km_travelled"] = grp["max"] - grp["min"]
            for veh, r in grp.head(max_groups).iterrows():
                out.append(
                    f"- {veh}: first={r['min']:,.0f}, last={r['max']:,.0f}, "
                    f"km_travelled={r['km_travelled']:,.0f}, readings={int(r['count'])}"
                )
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# 6. Deterministic audit
# ---------------------------------------------------------------------------

def run_audit(df: pd.DataFrame) -> list:
    """
    Run generic, deterministic audit checks. Returns a list of finding strings.
    Heuristics activate only when matching columns exist, so it stays general.
    """
    if df is None or df.empty:
        return ["No records to audit."]

    findings = []
    content_cols = [c for c in df.columns if not c.startswith("_")]
    num_cols = _numeric_columns(df)

    # Duplicate-copy markers
    if "_is_duplicate_copy" in df.columns:
        n = int((df["_is_duplicate_copy"] == True).sum())  # noqa: E712
        if n:
            findings.append(f"{n} record(s) explicitly marked as duplicate/customer copy.")

    # Content duplicates
    if "_is_duplicate" in df.columns:
        dups = df[df["_is_duplicate"] == True]  # noqa: E712
        if not dups.empty:
            findings.append(
                f"{len(dups)} duplicate record(s) detected across "
                f"{dups['_duplicate_group'].nunique()} group(s) (identical content)."
            )

    # Duplicate ID values + missing IDs (receipt/invoice numbers only —
    # vehicle numbers legitimately repeat and must not be flagged here)
    for col in content_cols:
        if _is_unique_id_col(col):
            vals = df[col].astype(str).str.strip().replace("", None)
            missing = int(vals.isna().sum())
            if missing:
                findings.append(f"{missing} record(s) with missing '{col}'.")
            dup_ids = vals.dropna().value_counts()
            dup_ids = dup_ids[dup_ids > 1]
            if not dup_ids.empty:
                findings.append(
                    f"Duplicate '{col}' value(s): "
                    + ", ".join(f"{v} (x{c})" for v, c in dup_ids.head(15).items())
                )

    # Missing dates / amounts / quantities
    for col in content_cols:
        if _name_matches(col, _DATE_PATTERNS) or col.lower() in (
            "amount", "total", "value", "price", "quantity", "qty", "liters", "litres", "volume"
        ):
            missing = int(df[col].astype(str).str.strip().replace("", None).isna().sum())
            if missing:
                findings.append(f"{missing} record(s) with missing '{col}'.")

    # amount ≈ quantity × unit_price consistency
    amount_col = next((c for c in content_cols if c.lower() in ("amount", "total", "total_amount", "value")), None)
    qty_col = next((c for c in content_cols if c.lower() in ("quantity", "qty", "liters", "litres", "volume")), None)
    price_col = next((c for c in content_cols if "price" in c.lower() or "rate" in c.lower() or "unit" in c.lower()), None)
    if amount_col and qty_col and price_col:
        a = _numeric_series(df[amount_col])
        q = _numeric_series(df[qty_col])
        p = _numeric_series(df[price_col])
        expected = q * p
        diff = (a - expected).abs()
        mask = diff.notna() & (diff > (a.abs() * 0.02 + 0.01))
        n_bad = int(mask.sum())
        if n_bad:
            findings.append(
                f"{n_bad} record(s) where {amount_col} != {qty_col} x {price_col} "
                f"(>2% mismatch) - possible calculation error."
            )

    # Outlier quantities
    if qty_col:
        q = _numeric_series(df[qty_col]).dropna()
        if len(q) >= 5:
            lo, hi = q.quantile(0.01), q.quantile(0.99)
            n_out = int(((q < lo) | (q > hi)).sum())
            if n_out:
                findings.append(f"{n_out} record(s) with unusually high/low '{qty_col}' (outliers).")

    # Decreasing odometer over page order, per vehicle
    veh_col = next((c for c in content_cols if _name_matches(c, _VEHICLE_PATTERNS)), None)
    odo_col = next((c for c in content_cols if _name_matches(c, _ODO_PATTERNS) and c in num_cols), None)
    if veh_col and odo_col and "_page" in df.columns:
        tmp = pd.DataFrame({
            "veh": df[veh_col].astype(str).str.strip().replace("", None),
            "odo": _numeric_series(df[odo_col]),
            "page": pd.to_numeric(df["_page"], errors="coerce"),
        }).dropna()
        bad_vehicles = []
        for veh, g in tmp.sort_values("page").groupby("veh"):
            if (g["odo"].diff() < 0).any():
                bad_vehicles.append(str(veh))
        if bad_vehicles:
            findings.append(
                f"Odometer reading decreases over time for vehicle(s): "
                + ", ".join(bad_vehicles[:15])
            )

    # Low-confidence / unclear fields
    if "_unclear_fields" in df.columns:
        n_unclear = int((df["_unclear_fields"].astype(str).str.strip() != "").sum())
        if n_unclear:
            findings.append(f"{n_unclear} record(s) contain low-confidence/unclear fields needing manual review.")

    # Missing signature
    if "_has_signature" in df.columns:
        n_nosig = int((df["_has_signature"] == False).sum())  # noqa: E712
        if n_nosig:
            findings.append(f"{n_nosig} record(s) with no detected signature.")

    return findings or ["No issues detected by automated checks."]


# ---------------------------------------------------------------------------
# 7. Export helpers
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")   # BOM → Excel reads UTF-8/Arabic


def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Records")
    return buf.getvalue()


def records_to_compact_json(df: pd.DataFrame, char_budget: int = 60000) -> tuple[str, bool]:
    """
    Compact JSON of all records for the LLM. Returns (json_str, truncated).
    If over budget, includes only the first N rows (aggregates remain authoritative).
    """
    if df is None or df.empty:
        return "[]", False
    records = df.to_dict(orient="records")
    full = json.dumps(records, ensure_ascii=False, default=str)
    if len(full) <= char_budget:
        return full, False
    truncated, n = [], 0
    for rec in records:
        s = json.dumps(rec, ensure_ascii=False, default=str)
        if n + len(s) > char_budget:
            break
        truncated.append(rec)
        n += len(s)
    return json.dumps(truncated, ensure_ascii=False, default=str), True
