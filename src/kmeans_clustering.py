from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================
# Config
# =========================
@dataclass(frozen=True)
class KMeansConfig:
    input_path: str = "data/processed/rfm_scaled.csv"
    output_path: str = "data/processed/rfm_clustered.csv"

    features: tuple[str, ...] = ("R_scaled", "F_scaled", "M_scaled")
    cluster_range: tuple[int, int] = (2, 8)   # inclusive
    final_k: int = 3
    random_state: int = 42
    n_init: int = 20


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_features(cfg: KMeansConfig) -> pd.DataFrame:
    _print_section("1) Loading scaled RFM features")

    path = (_project_root() / cfg.input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")
    return df


def evaluate_kmeans(df: pd.DataFrame, cfg: KMeansConfig) -> pd.DataFrame:
    _print_section("2) Evaluating KMeans (Elbow & Silhouette)")

    X = df[list(cfg.features)]

    results = []
    for k in range(cfg.cluster_range[0], cfg.cluster_range[1] + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=cfg.random_state,
            n_init=cfg.n_init
        )
        labels = kmeans.fit_predict(X)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette
        })

        print(f"k={k} | inertia={inertia:.2f} | silhouette={silhouette:.4f}")

    return pd.DataFrame(results)


def fit_final_model(df: pd.DataFrame, cfg: KMeansConfig) -> pd.DataFrame:
    _print_section(f"3) Fitting final KMeans (k={cfg.final_k})")

    X = df[list(cfg.features)]

    kmeans = KMeans(
        n_clusters=cfg.final_k,
        random_state=cfg.random_state,
        n_init=cfg.n_init
    )

    df["Cluster"] = kmeans.fit_predict(X)

    print("Cluster counts:")
    print(df["Cluster"].value_counts().sort_index())

    return df


def save_clustered(df: pd.DataFrame, cfg: KMeansConfig) -> None:
    _print_section("4) Saving clustered data")

    out_path = (_project_root() / cfg.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    cfg = KMeansConfig()
    df = load_features(cfg)

    eval_df = evaluate_kmeans(df, cfg)

    print("\nSummary (for decision making):")
    print(eval_df)

    df_clustered = fit_final_model(df, cfg)
    save_clustered(df_clustered, cfg)


if __name__ == "__main__":
    main()
