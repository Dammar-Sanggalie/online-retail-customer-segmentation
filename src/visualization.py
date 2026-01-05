from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_data():
    root = _project_root()
    rfm = pd.read_csv(root / "data/processed/rfm_raw.csv")
    clusters = pd.read_csv(root / "data/processed/rfm_clustered.csv")
    temporal = pd.read_csv(root / "data/processed/temporal_profile.csv")

    df = rfm.merge(clusters[["CustomerID", "Cluster"]], on="CustomerID")
    return df, temporal


def plot_cluster_size(df, outdir):
    counts = df["Cluster"].value_counts().sort_index()

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Customer Distribution by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(outdir / "cluster_size.png")
    plt.close()


def plot_rfm_profile(df, outdir):
    profile = (
        df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .median()
    )

    profile.plot(kind="bar")
    plt.title("Median RFM Profile by Cluster")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(outdir / "rfm_profile.png")
    plt.close()


def plot_hour_heatmap(temporal, outdir):
    # aggregate unique customers
    pivot = (
        temporal
        .groupby(["Cluster", "hour"])["unique_customers"]
        .sum()
        .reset_index()
        .pivot(index="Cluster", columns="hour", values="unique_customers")
        .fillna(0)
    )

    # normalize to %
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 3))
    plt.imshow(pivot_pct, aspect="auto")
    plt.colorbar(label="Share of Unique Customers")
    plt.yticks(range(len(pivot_pct.index)), pivot_pct.index)
    plt.xticks(range(len(pivot_pct.columns)), pivot_pct.columns)
    plt.xlabel("Hour of Day")
    plt.ylabel("Cluster")
    plt.title("Unique Customers Activity by Hour (Normalized)")
    plt.tight_layout()
    plt.savefig(outdir / "hour_heatmap.png")
    plt.close()


def plot_top_dow(temporal, outdir):
    top = (
        temporal
        .groupby(["Cluster", "day_of_week"])["unique_customers"]
        .sum()
        .reset_index()
    )

    for cluster in sorted(top["Cluster"].unique()):
        sub = top[top["Cluster"] == cluster]
        sub = sub.sort_values("unique_customers", ascending=False)

        plt.figure()
        plt.bar(sub["day_of_week"], sub["unique_customers"])
        plt.title(f"Top Days by Unique Customers (Cluster {cluster})")
        plt.ylabel("Unique Customers")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(outdir / f"dow_cluster_{cluster}.png")
        plt.close()


def plot_dow_heatmap(temporal, outdir):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    dow_agg = (
        temporal.groupby(["Cluster", "day_of_week"])["unique_customers"]
        .sum()
        .reset_index()
    )

    dow_agg["day_of_week"] = pd.Categorical(dow_agg["day_of_week"], categories=order, ordered=True)
    pivot = (
        dow_agg.pivot(index="Cluster", columns="day_of_week", values="unique_customers")
        .fillna(0)
        .sort_index()
    )

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 3))
    plt.imshow(pivot_pct.values, aspect="auto")
    plt.colorbar(label="Share of Unique Customers")

    plt.yticks(range(len(pivot_pct.index)), pivot_pct.index)
    plt.xticks(range(len(pivot_pct.columns)), pivot_pct.columns, rotation=30, ha="right")
    plt.xlabel("Day of Week")
    plt.ylabel("Cluster")
    plt.title("Unique Customers Activity by Day of Week (Normalized within Cluster)")
    plt.tight_layout()
    plt.savefig(outdir / "dow_heatmap.png")
    plt.close()


def plot_hour_by_dow_heatmap(temporal, outdir):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    agg = (
        temporal.groupby(["day_of_week", "hour"])["unique_customers"]
        .sum()
        .reset_index()
    )
    agg["day_of_week"] = pd.Categorical(agg["day_of_week"], categories=order, ordered=True)

    pivot = (
        agg.pivot(index="day_of_week", columns="hour", values="unique_customers")
        .fillna(0)
        .sort_index()
    )

    # normalize within each day (biar keliatan peak jam per hari)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 5))
    plt.imshow(pivot_pct.values, aspect="auto")
    plt.colorbar(label="Share of Unique Customers within Day")

    plt.yticks(range(len(pivot_pct.index)), pivot_pct.index)
    plt.xticks(range(len(pivot_pct.columns)), pivot_pct.columns)
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.title("Unique Customers Activity Heatmap: Day of Week x Hour (Normalized per Day)")
    plt.tight_layout()
    plt.savefig(outdir / "dow_hour_heatmap.png")
    plt.close()


def plot_month_period(df_temporal, outdir):
    period_agg = (
        df_temporal.groupby(["Cluster", "month_period"])["unique_customers"]
        .sum()
        .reset_index()
    )

    pivot = (
        period_agg.pivot(index="Cluster", columns="month_period", values="unique_customers")
        .fillna(0)
        .sort_index()
    )

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    pivot_pct.plot(kind="bar")
    plt.title("Unique Customers by Month Period (Normalized within Cluster)")
    plt.xlabel("Cluster")
    plt.ylabel("Share of Unique Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "month_period_cluster.png")
    plt.close()


def main():
    df, temporal = load_data()
    outdir = _project_root() / "reports/figures"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_cluster_size(df, outdir)
    plot_rfm_profile(df, outdir)
    plot_hour_heatmap(temporal, outdir)
    plot_top_dow(temporal, outdir)

    # NEW
    plot_dow_heatmap(temporal, outdir)
    plot_hour_by_dow_heatmap(temporal, outdir)
    plot_month_period(temporal, outdir)

    print("All figures saved to reports/figures/")


if __name__ == "__main__":
    main()