"""Central configuration for the Olist Big Data Analytics project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures"
TAB_DIR      = PROJECT_ROOT / "outputs" / "tables"
MODEL_DIR    = PROJECT_ROOT / "outputs" / "models"

for _d in (RAW_DIR, PROC_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Kaggle dataset slug — Olist Brazilian E-Commerce
KAGGLE_SLUG = "olistbr/brazilian-ecommerce"

# Expected raw files (exact names as published on Kaggle)
RAW_FILES = {
    "customers":  "olist_customers_dataset.csv",
    "geolocation":"olist_geolocation_dataset.csv",
    "order_items":"olist_order_items_dataset.csv",
    "payments":   "olist_order_payments_dataset.csv",
    "reviews":    "olist_order_reviews_dataset.csv",
    "orders":     "olist_orders_dataset.csv",
    "products":   "olist_products_dataset.csv",
    "sellers":    "olist_sellers_dataset.csv",
    "category":   "product_category_name_translation.csv",
}

RANDOM_STATE = 42