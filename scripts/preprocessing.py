"""Phase 3 — Data validation and cleaning for the Olist dataset.

Design principles:
  * Never silently drop data. Every mutation is logged to a ledger.
  * Preserve both raw and processed versions of every table.
  * Every timestamp becomes a real datetime; every impossible value is flagged.
  * Outputs a single analysis-ready master table + a data-quality ledger.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.config import PROC_DIR, TAB_DIR, RAW_FILES
from scripts.data_loading import load_all

LEDGER: list[dict] = []


def _log(stage: str, table: str, problem: str, action: str,
         rows_before: int, rows_after: int, notes: str = "") -> None:
    LEDGER.append({
        "stage":         stage,
        "table":         table,
        "problem":       problem,
        "action":        action,
        "rows_before":   rows_before,
        "rows_after":    rows_after,
        "rows_delta":    rows_after - rows_before,
        "notes":         notes,
    })


# ---------------------------------------------------------------- 3.2.1 orders
ORDER_TS_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    n0 = len(orders)
    df = orders.copy()

    # (1) type coercion for timestamps
    for c in ORDER_TS_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # (2) deduplicate on order_id (defensive — should be a PK)
    dup = df.duplicated(subset=["order_id"]).sum()
    if dup:
        df = df.drop_duplicates(subset=["order_id"], keep="first")
        _log("clean_orders", "orders", "duplicate order_id",
             "drop_duplicates(keep=first)", n0, len(df), f"removed={dup}")

    # (3) flag impossible timestamps: carrier_date before purchase, etc.
    bad_seq = (
        (df["order_delivered_carrier_date"] < df["order_purchase_timestamp"]) |
        (df["order_delivered_customer_date"] < df["order_delivered_carrier_date"])
    )
    df["flag_timestamp_inversion"] = bad_seq.fillna(False)

    # (4) flag status codes not in the known set (schema validation)
    valid_status = {"delivered", "shipped", "canceled", "unavailable",
                    "invoiced", "processing", "created", "approved"}
    df["flag_unknown_status"] = ~df["order_status"].isin(valid_status)

    # (5) derived temporal features (only where computable)
    df["delivery_days"]   = (df["order_delivered_customer_date"] -
                             df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["estimated_days"]  = (df["order_estimated_delivery_date"] -
                             df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["delay_days"]      = df["delivery_days"] - df["estimated_days"]
    df["is_late"]         = (df["delay_days"] > 0).astype("Int64")   # nullable
    df.loc[df["order_delivered_customer_date"].isna(), "is_late"] = pd.NA

    _log("clean_orders", "orders", "string timestamps + impossible sequence",
         "coerce datetimes; flag inversions; derive delay_days/is_late",
         n0, len(df),
         f"flagged_inversion={int(df['flag_timestamp_inversion'].sum())}, "
         f"flagged_status={int(df['flag_unknown_status'].sum())}")

    return df


# ---------------------------------------------------------------- 3.2.2 reviews
def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    n0 = len(reviews)
    df = reviews.copy()

    for c in ("review_creation_date", "review_answer_timestamp"):
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    invalid_score = df["review_score"].notna() & ~df["review_score"].between(1, 5)
    df.loc[invalid_score, "review_score"] = np.nan

    # Multiple reviews per order are allowed in the raw data.
    # Keep the LATEST review per order_id, log how many rows were collapsed.
    before = len(df)
    df = (df.sort_values("review_answer_timestamp")
            .drop_duplicates(subset=["order_id"], keep="last"))
    _log("clean_reviews", "reviews",
         "duplicate reviews per order_id + invalid scores",
         "keep latest review per order; null out out-of-range scores",
         n0, len(df),
         f"collapsed={before - len(df)}, "
         f"invalid_scores_nulled={int(invalid_score.sum())}")

    df["has_comment"] = df["review_comment_message"].notna()
    return df


# ---------------------------------------------------------------- 3.2.3 products
def clean_products(products: pd.DataFrame,
                   category: pd.DataFrame) -> pd.DataFrame:
    n0 = len(products)
    df = products.merge(category, on="product_category_name", how="left")

    num_cols = ["product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # zero-weight / zero-dimension products are physically impossible
        df.loc[df[c] <= 0, c] = np.nan

    df["product_category_name_english"] = (
        df["product_category_name_english"].fillna("unknown")
    )
    df["product_category_name"] = (
        df["product_category_name"].fillna("unknown")
    )

    _log("clean_products", "products",
         "missing English category + zero/negative physical dimensions",
         "left-join translation; set non-positive dims to NaN; label unknown",
         n0, len(df),
         f"unknown_category_rows="
         f"{int((df['product_category_name_english'] == 'unknown').sum())}")
    return df


# ---------------------------------------------------------------- 3.2.4 geolocation
def clean_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    n0 = len(geo)
    df = geo.copy()
    for c in ("geolocation_lat", "geolocation_lng"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Brazil bounding box sanity (lat -34..6, lng -74..-34)
    in_brazil = (df["geolocation_lat"].between(-34, 6) &
                 df["geolocation_lng"].between(-74, -34))
    df = df.loc[in_brazil]

    agg = (df.groupby("geolocation_zip_code_prefix", as_index=False)
             .agg(geolocation_lat=("geolocation_lat", "median"),
                  geolocation_lng=("geolocation_lng", "median"),
                  geolocation_city=("geolocation_city", "first"),
                  geolocation_state=("geolocation_state", "first")))

    _log("clean_geolocation", "geolocation",
         "multiple rows per zip prefix + out-of-Brazil coordinates",
         "restrict to Brazil bbox; median-aggregate to one row per zip",
         n0, len(agg),
         f"out_of_bbox_removed={n0 - int(in_brazil.sum())}")
    return agg


# ---------------------------------------------------------------- 3.2.5 passthrough
def clean_passthrough(name: str, df: pd.DataFrame,
                      key: str | None = None) -> pd.DataFrame:
    n0 = len(df)
    out = df.copy()
    if key and key in out.columns:
        dup = out.duplicated(subset=[key]).sum()
        if dup:
            out = out.drop_duplicates(subset=[key], keep="first")
            _log(f"clean_{name}", name, f"duplicate {key}",
                 "drop_duplicates(keep=first)", n0, len(out),
                 f"removed={dup}")
    return out


# ---------------------------------------------------------------- 3.2.6 master build
def build_master(orders, items, payments, customers,
                 products, sellers, reviews) -> pd.DataFrame:
    """Left-join everything onto orders. Every join is logged."""
    n0 = len(orders)

    # Payments: one row per (order, seq) → aggregate to order level
    pay = (payments.groupby("order_id", as_index=False)
                   .agg(payment_value=("payment_value", "sum"),
                        payment_installments=("payment_installments", "max"),
                        payment_type=("payment_type", "first")))

    items_agg = (items.groupby("order_id", as_index=False)
                      .agg(n_items=("order_item_id", "count"),
                           price_total=("price", "sum"),
                           freight_total=("freight_value", "sum"),
                           product_id=("product_id", "first"),
                           seller_id=("seller_id", "first")))

    master = (orders
              .merge(customers, on="customer_id", how="left")
              .merge(items_agg, on="order_id", how="left")
              .merge(pay,       on="order_id", how="left")
              .merge(products,  on="product_id", how="left")
              .merge(sellers,   on="seller_id", how="left")
              .merge(reviews[["order_id", "review_score", "has_comment"]],
                     on="order_id", how="left"))

    master["freight_ratio"] = (master["freight_total"] /
                               master["price_total"].replace(0, np.nan))

    _log("build_master", "master", "multi-table join",
         "left-join on keys; aggregate payments & items first",
         n0, len(master), f"final_cols={master.shape[1]}")
    return master


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    frames = load_all()

    orders     = clean_orders(frames["orders"])
    reviews    = clean_reviews(frames["reviews"])
    products   = clean_products(frames["products"], frames["category"])
    geolocation= clean_geolocation(frames["geolocation"])
    customers  = clean_passthrough("customers",  frames["customers"], "customer_id")
    sellers    = clean_passthrough("sellers",    frames["sellers"],   "seller_id")
    items      = clean_passthrough("order_items",frames["order_items"])
    payments   = clean_passthrough("payments",   frames["payments"])

    master = build_master(orders, items, payments, customers,
                          products, sellers, reviews)

    # persist processed artefacts
       # Try parquet (preserves dtypes); fall back to CSV if no engine available.
    try:
        import pyarrow  # noqa: F401
        _ext, _writer = "parquet", lambda d, p: d.to_parquet(p, index=False)
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            _ext, _writer = "parquet", lambda d, p: d.to_parquet(p, index=False)
        except ImportError:
            print("[WARN] No parquet engine (pyarrow/fastparquet) found. "
                  "Falling back to CSV. Install pyarrow for dtype fidelity: "
                  "pip install pyarrow")
            _ext, _writer = "csv", lambda d, p: d.to_csv(p, index=False)

    for name, df in [
        ("orders_clean", orders), ("reviews_clean", reviews),
        ("products_clean", products), ("geolocation_clean", geolocation),
        ("customers_clean", customers), ("sellers_clean", sellers),
        ("order_items_clean", items), ("payments_clean", payments),
        ("master", master),
    ]:
        _writer(df, PROC_DIR / f"{name}.{_ext}") 

    ledger = pd.DataFrame(LEDGER)
    ledger.to_csv(TAB_DIR / "cleaning_ledger.csv", index=False)

    # before/after summary for the report
    summary = pd.DataFrame([{
        "table":        k,
        "rows_before":  len(frames[k]),
        "cols_before":  frames[k].shape[1],
    } for k in frames])
    after = {
        "orders": len(orders), "reviews": len(reviews),
        "products": len(products), "geolocation": len(geolocation),
        "customers": len(customers), "sellers": len(sellers),
        "order_items": len(items), "payments": len(payments),
        "category": len(frames["category"]),
    }
    summary["rows_after"] = summary["table"].map(after)
    summary.to_csv(TAB_DIR / "cleaning_before_after.csv", index=False)

    print("\n=== CLEANING LEDGER ===")
    print(ledger.to_string(index=False))
    print("\n=== BEFORE / AFTER ROW COUNTS ===")
    print(summary.to_string(index=False))
    print(f"\n[INFO] Master shape: {master.shape}")
    print(f"[INFO] Master columns: {list(master.columns)}")
    print(f"\n[INFO] Processed parquet files in {PROC_DIR}")
    print(f"[INFO] Ledger written to {TAB_DIR / 'cleaning_ledger.csv'}")