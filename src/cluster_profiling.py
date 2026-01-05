from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


# =========================
# Config
# =========================
@dataclass(frozen=True)
class ProfilingConfig:
    rfm_raw_path: str = "data/processed/rfm_raw.csv"
    rfm_clustered_path: str = "data/processed/rfm_clustered.csv"
    output_path: str = "data/processed/cluster_profile.csv"

    id_col: str = "CustomerID"
    cluster_col: str = "Cluster"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data(cfg: ProfilingConfig) -> pd.DataFrame:
    _print_section("1) Loading RFM & cluster data")

    root = _project_root()
    rfm_raw = pd.read_csv((root / cfg.rfm_raw_path).resolve())
    rfm_clustered = pd.read_csv((root / cfg.rfm_clustered_path).resolve())

    df = rfm_raw.merge(
        rfm_clustered[[cfg.id_col, cfg.cluster_col]],
        on=cfg.id_col,
        how="inner"
    )

    print(f"Merged shape: {df.shape}")
    print("Cluster distribution:")
    print(df[cfg.cluster_col].value_counts().sort_index())

    return df


def profile_clusters(df: pd.DataFrame, cfg: ProfilingConfig) -> pd.DataFrame:
    _print_section("2) Profiling clusters (mean & median)")

    profile = (
        df
        .groupby(cfg.cluster_col)
        .agg(
            Customers=(cfg.id_col, "count"),
            Recency_mean=("Recency", "mean"),
            Recency_median=("Recency", "median"),
            Frequency_mean=("Frequency", "mean"),
            Frequency_median=("Frequency", "median"),
            Monetary_mean=("Monetary", "mean"),
            Monetary_median=("Monetary", "median"),
        )
        .reset_index()
        .sort_values("Monetary_mean", ascending=False)
    )

    print(profile)

    return profile


def save_profile(profile: pd.DataFrame, cfg: ProfilingConfig) -> None:
    _print_section("3) Saving cluster profile")

    out_path = (_project_root() / cfg.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    cfg = ProfilingConfig()
    df = load_data(cfg)
    profile = profile_clusters(df, cfg)
    save_profile(profile, cfg)


if __name__ == "__main__":
    main()