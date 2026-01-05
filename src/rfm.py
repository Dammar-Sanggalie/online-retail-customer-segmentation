from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


# =========================
# Config
# =========================
@dataclass(frozen=True)
class RFMConfig:
    clean_path: str = "data/interim/online_retail_clean.csv"
    output_path: str = "data/processed/rfm_raw.csv"

    customer_id_col: str = "Customer ID"
    invoice_col: str = "Invoice"
    datetime_col: str = "InvoiceDate"
    revenue_col: str = "Revenue"

    # Snapshot date strategy: max(InvoiceDate) + 1 day
    snapshot_add_days: int = 1


def _project_root() -> Path:
    # project root = parent folder dari /src
    return Path(__file__).resolve().parents[1]


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_clean(cfg: RFMConfig) -> pd.DataFrame:
    _print_section("1) Loading cleaned transactions")

    path = (_project_root() / cfg.clean_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Clean file not found: {path}\n"
            f"Pastikan kamu sudah run src/data_cleaning.py dan file outputnya ada."
        )

    df = pd.read_csv(path, low_memory=False)

    # Ensure datetime
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    if df[cfg.datetime_col].isna().any():
        bad = df[cfg.datetime_col].isna().sum()
        raise ValueError(f"Found {bad} invalid InvoiceDate after loading clean data.")

    # Minimal sanity checks
    required = [cfg.customer_id_col, cfg.invoice_col, cfg.datetime_col, cfg.revenue_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in clean data: {missing_cols}")

    print(f"Loaded shape: {df.shape}")
    print("Date range:", df[cfg.datetime_col].min(), "->", df[cfg.datetime_col].max())
    print("Unique customers:", df[cfg.customer_id_col].nunique())
    print("Unique invoices:", df[cfg.invoice_col].nunique())
    return df


def build_rfm(df: pd.DataFrame, cfg: RFMConfig) -> tuple[pd.DataFrame, pd.Timestamp]:
    _print_section("2) Building RFM table")

    # Snapshot date = max invoice date + 1 day (biar recency >= 0)
    max_dt = df[cfg.datetime_col].max()
    snapshot_date = (max_dt.normalize() + pd.Timedelta(days=cfg.snapshot_add_days))
    print("Snapshot date:", snapshot_date)

    # --- RFM definition ---
    # Recency: days since last purchase (customer max invoice date)
    # Frequency: number of unique invoices
    # Monetary: sum of revenue
    agg = df.groupby(cfg.customer_id_col).agg(
        last_purchase=(cfg.datetime_col, "max"),
        Frequency=(cfg.invoice_col, pd.Series.nunique),
        Monetary=(cfg.revenue_col, "sum"),
    )

    agg["Recency"] = (snapshot_date - agg["last_purchase"]).dt.days

    # reorder + clean
    rfm = agg[["Recency", "Frequency", "Monetary"]].reset_index()
    rfm = rfm.rename(columns={cfg.customer_id_col: "CustomerID"})

    # basic sanity
    if (rfm["Recency"] < 0).any():
        n = (rfm["Recency"] < 0).sum()
        raise ValueError(f"Found {n} customers with negative recency. Check snapshot date logic.")

    print(f"RFM shape (customers): {rfm.shape}")
    return rfm, snapshot_date


def rfm_diagnostics(rfm: pd.DataFrame) -> None:
    _print_section("3) Quick RFM diagnostics (light EDA)")

    print("Head:")
    print(rfm.head())

    print("\nDescribe:")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    # Skew quick check (numeric skewness)
    skew = rfm[["Recency", "Frequency", "Monetary"]].skew(numeric_only=True).sort_values(ascending=False)
    print("\nSkewness (indikasi jomplang/outlier):")
    print(skew)

    # Outlier peek (top 10)
    print("\nTop 10 Monetary:")
    print(rfm.sort_values("Monetary", ascending=False).head(10)[["CustomerID", "Recency", "Frequency", "Monetary"]])

    print("\nTop 10 Frequency:")
    print(rfm.sort_values("Frequency", ascending=False).head(10)[["CustomerID", "Recency", "Frequency", "Monetary"]])

    print("\nTop 10 Recency (paling lama ga belanja):")
    print(rfm.sort_values("Recency", ascending=False).head(10)[["CustomerID", "Recency", "Frequency", "Monetary"]])


def save_rfm(rfm: pd.DataFrame, cfg: RFMConfig) -> None:
    _print_section("4) Saving RFM")

    out_path = (_project_root() / cfg.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rfm.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Rows(customers): {len(rfm)}")


def main() -> None:
    cfg = RFMConfig()
    df = load_clean(cfg)
    rfm, snapshot_date = build_rfm(df, cfg)
    rfm_diagnostics(rfm)
    save_rfm(rfm, cfg)


if __name__ == "__main__":
    main()
