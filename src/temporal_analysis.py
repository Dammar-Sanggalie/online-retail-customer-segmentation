# src/temporal_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


# =========================
# Config
# =========================
@dataclass(frozen=True)
class TemporalConfig:
    transactions_path: str = "data/interim/online_retail_clean.csv"
    clustered_path: str = "data/processed/rfm_clustered.csv"
    output_path: str = "data/processed/temporal_profile.csv"

    id_col: str = "CustomerID"
    datetime_col: str = "InvoiceDate"
    cluster_col: str = "Cluster"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data(cfg: TemporalConfig) -> pd.DataFrame:
    _print_section("1) Loading transactions & clusters")

    root = _project_root()
    trx = pd.read_csv((root / cfg.transactions_path).resolve(), low_memory=False)
    clusters = pd.read_csv((root / cfg.clustered_path).resolve(), low_memory=False)

    # Parse datetime
    trx[cfg.datetime_col] = pd.to_datetime(trx[cfg.datetime_col], errors="coerce")
    trx = trx.dropna(subset=[cfg.datetime_col])

    # Normalize ID column name
    trx = trx.rename(columns={"Customer ID": "CustomerID"})
    clusters = clusters.rename(columns={"Customer ID": "CustomerID"})

    # Normalize ID dtype/format
    trx["CustomerID"] = trx["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    clusters["CustomerID"] = clusters["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # Debug quick checks
    print("trx cols:", trx.columns.tolist())
    print("clusters cols:", clusters.columns.tolist())
    print("trx shape:", trx.shape, "| clusters shape:", clusters.shape)

    # Keep only required columns from clusters
    clusters = clusters[["CustomerID", cfg.cluster_col]]

    df = trx.merge(clusters, on="CustomerID", how="inner")

    print(f"Merged shape: {df.shape}")
    print("Cluster distribution:")
    print(df[cfg.cluster_col].value_counts().sort_index())

    return df


def enrich_time_features(df: pd.DataFrame, cfg: TemporalConfig) -> pd.DataFrame:
    _print_section("2) Creating temporal features")

    df = df.copy()

    df["hour"] = df[cfg.datetime_col].dt.hour
    df["day_of_week"] = df[cfg.datetime_col].dt.day_name()
    df["dow_num"] = df[cfg.datetime_col].dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["dow_num"] >= 5

    df["day_of_month"] = df[cfg.datetime_col].dt.day
    df["month"] = df[cfg.datetime_col].dt.month

    df["month_period"] = pd.cut(
        df["day_of_month"],
        bins=[0, 10, 20, 31],
        labels=["Early", "Mid", "Late"]
    )

    return df


def top_preferences(df: pd.DataFrame, cluster_col: str, col: str, top_n: int = 3) -> pd.DataFrame:
    # count
    counts = df.groupby([cluster_col, col]).size().reset_index(name="transactions")

    # percent within cluster
    totals = counts.groupby(cluster_col)["transactions"].transform("sum")
    counts["pct_within_cluster"] = counts["transactions"] / totals

    # top N per cluster
    top = (
        counts.sort_values([cluster_col, "pct_within_cluster"], ascending=[True, False])
              .groupby(cluster_col)
              .head(top_n)
              .reset_index(drop=True)
    )
    return top


def top_unique_customers(df: pd.DataFrame, cluster_col: str, time_col: str, id_col: str, top_n: int = 3) -> pd.DataFrame:
    """
    Hitung top time slot berdasarkan UNIQUE customers active (% within cluster).
    """
    # count unique customers per slot
    counts = (
        df.groupby([cluster_col, time_col])[id_col]
          .nunique()
          .reset_index(name="unique_customers")
    )

    # percent within cluster
    totals = counts.groupby(cluster_col)["unique_customers"].transform("sum")
    counts["pct_unique_within_cluster"] = counts["unique_customers"] / totals

    # top N per cluster
    top = (
        counts.sort_values([cluster_col, "pct_unique_within_cluster"], ascending=[True, False])
              .groupby(cluster_col)
              .head(top_n)
              .reset_index(drop=True)
    )
    return top


def temporal_summary(df: pd.DataFrame, cfg: TemporalConfig) -> None:
    _print_section("3) Temporal summary by cluster")

    # Day of week
    print("\n--- Day of Week Preference (count) ---")
    dow = (
        df.groupby([cfg.cluster_col, "day_of_week"])
        .size()
        .reset_index(name="transactions")
        .sort_values([cfg.cluster_col, "transactions"], ascending=[True, False])
    )
    print(dow.head(15))

    # Hour of day
    print("\n--- Hour of Day Preference (count) ---")
    hour = (
        df.groupby([cfg.cluster_col, "hour"])
        .size()
        .reset_index(name="transactions")
        .sort_values([cfg.cluster_col, "transactions"], ascending=[True, False])
    )
    print(hour.head(15))

    # Month period
    print("\n--- Month Period Preference (count) ---")
    period = (
        df.groupby([cfg.cluster_col, "month_period"], observed=True)
        .size()
        .reset_index(name="transactions")
        .sort_values([cfg.cluster_col, "transactions"], ascending=[True, False])
    )
    print(period)

    # Top preferences with percentages
    print("\n--- TOP Day of Week by % within cluster ---")
    print(top_preferences(df, cfg.cluster_col, "day_of_week", top_n=3))

    print("\n--- TOP Hour of Day by % within cluster ---")
    print(top_preferences(df, cfg.cluster_col, "hour", top_n=5))

    print("\n--- TOP Month Period by % within cluster ---")
    print(top_preferences(df, cfg.cluster_col, "month_period", top_n=3))

    # Top time slots by unique customers
    print("\n--- TOP Day of Week by UNIQUE customers (% within cluster) ---")
    print(top_unique_customers(df, cfg.cluster_col, "day_of_week", "CustomerID", top_n=3))

    print("\n--- TOP Hour of Day by UNIQUE customers (% within cluster) ---")
    print(top_unique_customers(df, cfg.cluster_col, "hour", "CustomerID", top_n=5))

    print("\n--- TOP Month Period by UNIQUE customers (% within cluster) ---")
    print(top_unique_customers(df, cfg.cluster_col, "month_period", "CustomerID", top_n=3))


def save_temporal_profile(df: pd.DataFrame, cfg: TemporalConfig) -> None:
    _print_section("4) Saving aggregated temporal profile")

    # 1) transactions count
    trx_profile = (
        df.groupby([cfg.cluster_col, "day_of_week", "hour", "month_period"], observed=True)
        .size()
        .reset_index(name="transactions")
    )

    # 2) unique customers
    uniq_profile = (
        df.groupby([cfg.cluster_col, "day_of_week", "hour", "month_period"], observed=True)["CustomerID"]
        .nunique()
        .reset_index(name="unique_customers")
    )

    # merge them
    profile = trx_profile.merge(
        uniq_profile,
        on=[cfg.cluster_col, "day_of_week", "hour", "month_period"],
        how="inner"
    )

    out_path = (_project_root() / cfg.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Columns:", profile.columns.tolist())


def main() -> None:
    cfg = TemporalConfig()
    df = load_data(cfg)
    df = enrich_time_features(df, cfg)
    temporal_summary(df, cfg)
    save_temporal_profile(df, cfg)


if __name__ == "__main__":
    main()
