from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# =========================
# Config
# =========================
@dataclass(frozen=True)
class FeatureConfig:
    rfm_input: str = "data/processed/rfm_raw.csv"
    output_path: str = "data/processed/rfm_scaled.csv"

    id_col: str = "CustomerID"
    r_col: str = "Recency"
    f_col: str = "Frequency"
    m_col: str = "Monetary"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_rfm(cfg: FeatureConfig) -> pd.DataFrame:
    _print_section("1) Loading RFM data")

    path = (_project_root() / cfg.rfm_input).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"RFM file not found: {path}\n"
            f"Pastikan kamu sudah run src/rfm.py"
        )

    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")
    return df


def log_transform(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    _print_section("2) Log transformation (log1p)")

    df = df.copy()

    df["R_log"] = np.log1p(df[cfg.r_col])
    df["F_log"] = np.log1p(df[cfg.f_col])
    df["M_log"] = np.log1p(df[cfg.m_col])

    print("Skewness after log transform:")
    print(
        df[["R_log", "F_log", "M_log"]]
        .skew()
        .sort_values(ascending=False)
    )

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    _print_section("3) Feature scaling (StandardScaler)")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["R_log", "F_log", "M_log"]])

    df_scaled = df.copy()
    df_scaled[["R_scaled", "F_scaled", "M_scaled"]] = scaled

    print("Scaled feature summary (mean ~0, std ~1):")
    print(df_scaled[["R_scaled", "F_scaled", "M_scaled"]].describe())

    return df_scaled


def save_features(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    _print_section("4) Saving scaled features")

    out_path = (_project_root() / cfg.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        cfg.id_col,
        "R_log", "F_log", "M_log",
        "R_scaled", "F_scaled", "M_scaled",
    ]

    df[cols].to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows(customers): {len(df)}")


def main() -> None:
    cfg = FeatureConfig()
    df = load_rfm(cfg)
    df = log_transform(df, cfg)
    df = scale_features(df)
    save_features(df, cfg)


if __name__ == "__main__":
    main()