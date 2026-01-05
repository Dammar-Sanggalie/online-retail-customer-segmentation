# src/data_cleaning.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


# =========================
# Config (simple & portable)
# =========================
@dataclass(frozen=True)
class CleaningConfig:
    raw_path: str = "data/raw/online_retail_II.csv"      
    output_path: str = "data/interim/online_retail_clean.csv"
    datetime_col: str = "InvoiceDate"
    customer_id_col: str = "Customer ID"
    invoice_col: str = "Invoice"
    qty_col: str = "Quantity"
    price_col: str = "Price"
    desc_col: str = "Description"


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_raw(cfg: CleaningConfig) -> pd.DataFrame:
    _print_section("1) Loading raw data")
    path = Path(cfg.raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {cfg.raw_path}\n"
            f"Pastikan file CSV ada di folder data/raw/ dan namanya sesuai."
        )

    # Low-memory False biar dtype inference lebih stabil
    df = pd.read_csv(path, low_memory=False)

    print(f"Loaded shape: {df.shape}")
    print("Columns:", list(df.columns))
    return df


def clean_transactions(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    _print_section("2) Applying cleaning policy")

    df = df.copy()

    # --- Basic type cleanup ---
    # Customer ID sering kebaca float (13085.0). Kita simpan sebagai string/int-ish biar aman.
    # Tapi sebelum itu, kita drop null dulu.
    initial_rows = len(df)

    # Rule 1: drop Customer ID null
    null_cust = df[cfg.customer_id_col].isna().sum()
    df = df.dropna(subset=[cfg.customer_id_col])
    print(f"Drop CustomerID null: {null_cust} rows")

    # Rule 2: parse datetime
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    null_dt = df[cfg.datetime_col].isna().sum()
    if null_dt > 0:
        # kalau parsing gagal, row ini ga bisa dipakai buat recency & temporal analysis
        df = df.dropna(subset=[cfg.datetime_col])
        print(f"Drop invalid InvoiceDate (parse failed): {null_dt} rows")

    # Normalize Customer ID format
    # Contoh: 13085.0 -> "13085"
    df[cfg.customer_id_col] = (
        df[cfg.customer_id_col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Rule 3: remove cancellation invoice starting with "C"
    is_cancel = df[cfg.invoice_col].astype(str).str.startswith("C", na=False)
    cancel_rows = is_cancel.sum()
    df = df.loc[~is_cancel]
    print(f"Drop cancellation invoices (Invoice startswith 'C'): {cancel_rows} rows")

    # Rule 4: remove Quantity <= 0 (return/cancel/meaningless)
    bad_qty = (df[cfg.qty_col] <= 0).sum()
    df = df.loc[df[cfg.qty_col] > 0]
    print(f"Drop Quantity <= 0: {bad_qty} rows")

    # Rule 5: remove Price <= 0 (non-economic transactions / errors)
    bad_price = (df[cfg.price_col] <= 0).sum()
    df = df.loc[df[cfg.price_col] > 0]
    print(f"Drop Price <= 0: {bad_price} rows")

    # Optional: drop Description null (karena kita ga pakai NLP item)
    if cfg.desc_col in df.columns:
        bad_desc = df[cfg.desc_col].isna().sum()
        if bad_desc > 0:
            df = df.dropna(subset=[cfg.desc_col])
            print(f"Drop Description null: {bad_desc} rows")

    # Feature: Revenue
    df["Revenue"] = df[cfg.qty_col] * df[cfg.price_col]

    # Final cleanup: drop duplicates (optional, tapi aman)
    before_dup = len(df)
    df = df.drop_duplicates()
    dropped_dup = before_dup - len(df)
    if dropped_dup > 0:
        print(f"Drop duplicates: {dropped_dup} rows")

    # Sort for sanity & reproducibility
    df = df.sort_values([cfg.customer_id_col, cfg.datetime_col]).reset_index(drop=True)

    final_rows = len(df)
    print(f"\nRows: {initial_rows} -> {final_rows} (dropped {initial_rows - final_rows})")
    print(f"Cleaned shape: {df.shape}")

    return df


def save_clean(df: pd.DataFrame, cfg: CleaningConfig) -> None:
    _print_section("3) Saving cleaned data")

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved: {cfg.output_path}")
    print(f"File size: {out_path.stat().st_size / (1024**2):.2f} MB")


def quality_report(df: pd.DataFrame, cfg: CleaningConfig) -> None:
    _print_section("4) Quick quality report (post-cleaning)")

    print("Date range:")
    print(df[cfg.datetime_col].min(), "->", df[cfg.datetime_col].max())

    print("\nUnique customers:", df[cfg.customer_id_col].nunique())
    print("Unique invoices:", df[cfg.invoice_col].nunique())

    # Revenue sanity
    print("\nRevenue summary:")
    print(df["Revenue"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    # Missing check
    miss = df.isna().mean().sort_values(ascending=False)
    print("\nTop missing ratios (should be near 0):")
    print(miss.head(10))


def main() -> None:
    cfg = CleaningConfig()

    df_raw = load_raw(cfg)
    df_clean = clean_transactions(df_raw, cfg)
    save_clean(df_clean, cfg)
    quality_report(df_clean, cfg)


if __name__ == "__main__":
    main()
