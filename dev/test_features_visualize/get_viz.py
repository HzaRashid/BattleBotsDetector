#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    ks_2samp,
    expon, norm, lognorm, weibull_min,
    gamma, invgauss, fisk, pareto
)

def load_session_data(file_path):
    print(f"    ▶️ Loading session data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        session = json.load(f)

    posts_df = pd.DataFrame(session["posts"])
    posts_df["created_at"] = pd.to_datetime(posts_df["created_at"], errors="coerce")

    if "users" not in session:
        raise KeyError("'users' attribute not found in session data")
    users_df = pd.DataFrame(session["users"])
    if "user_id" not in users_df.columns or "is_bot" not in users_df.columns:
        raise KeyError("Users must contain 'user_id' and 'is_bot'")
    users_df = users_df.rename(columns={"user_id": "author_id"})[["author_id", "is_bot"]]

    posts_df = posts_df.merge(users_df, on="author_id", how="left")
    if posts_df["is_bot"].isnull().any():
        raise ValueError("Missing is_bot after merging!")
    posts_df["is_bot"] = posts_df["is_bot"].astype(bool)

    return posts_df

def compute_inter_arrival_times(df):
    df = df.sort_values("created_at")
    diffs = df["created_at"].diff().dt.total_seconds().dropna()
    return diffs

def ks_test_distributions(time_diffs):
    """
    Fit several distributions to time_diffs and return
    a dict: {dist_name: (D_statistic, p_value)}.
    """
    dist_names = [
        "exponential", "normal", "lognormal", "weibull",
        "gamma", "invgauss", "loglogistic", "pareto"
    ]
    if len(time_diffs) < 2:
        return {d: (0.0, 1.0) for d in dist_names}

    stats = {}

    # 1) Exponential
    params = expon.fit(time_diffs)
    stats["exponential"] = ks_2samp(
        time_diffs, expon.rvs(*params, size=min(len(time_diffs), 1000))
    )

    # 2) Normal
    params = norm.fit(time_diffs)
    stats["normal"] = ks_2samp(
        time_diffs, norm.rvs(*params, size=min(len(time_diffs), 1000))
    )

    # 3) Lognormal
    params = lognorm.fit(time_diffs)
    stats["lognormal"] = ks_2samp(
        time_diffs, lognorm.rvs(*params, size=min(len(time_diffs), 1000))
    )

    # 4) Weibull
    params = weibull_min.fit(time_diffs)
    stats["weibull"] = ks_2samp(
        time_diffs, weibull_min.rvs(*params, size=min(len(time_diffs), 1000))
    )

    # 5) Gamma
    params = gamma.fit(time_diffs)
    stats["gamma"] = ks_2samp(
        time_diffs, gamma.rvs(*params, size=min(len(time_diffs), 1000))
    )

    # 6) Inverse Gaussian
    params = invgauss.fit(time_diffs)
    stats["invgauss"] = ks_2samp(
        time_diffs, invgauss.rvs(*params, size=min(len(time_diffs), 1000))
    )

    # 7) Log‑Logistic (Fisk)
    params = fisk.fit(time_diffs)
    stats["loglogistic"] = ks_2samp(
        time_diffs, fisk.rvs(*params, size=min(len(time_diffs), 1000))
    )

    # 8) Pareto
    params = pareto.fit(time_diffs)
    stats["pareto"] = ks_2samp(
        time_diffs, pareto.rvs(*params, size=min(len(time_diffs), 1000))
    )

    return {dist: (stat, pval) for dist, (stat, pval) in stats.items()}

def compute_ks_features(posts_df):
    rows = []
    for user_id, group in posts_df.groupby("author_id"):
        diffs = compute_inter_arrival_times(group)
        dist_res = ks_test_distributions(diffs)
        row = {
            "user_id": user_id,
            "is_bot": group["is_bot"].iloc[0],
        }
        # add ks_ and p_ features
        for d, (stat, pval) in dist_res.items():
            row[f"ks_{d}"] = stat
            row[f"p_{d}"]  = pval
        rows.append(row)
    return pd.DataFrame(rows)

def save_ks_figures(df, output_dir="figures", bins=20):
    os.makedirs(output_dir, exist_ok=True)
    distributions = [
        "ks_exponential", "ks_normal", "ks_lognormal", "ks_weibull",
        "ks_gamma",       "ks_invgauss", "ks_loglogistic", "ks_pareto"
    ]

    for dist in distributions:
        bots   = df[df["is_bot"]][dist]
        humans = df[~df["is_bot"]][dist]
        print(f"    Plotting {dist}: bots={len(bots)}, humans={len(humans)}")

        plt.figure(figsize=(6, 4))
        plt.hist(bots,   bins=bins, alpha=0.6, label="Bots",   density=True)
        plt.hist(humans, bins=bins, alpha=0.6, label="Humans", density=True)
        plt.title(f"KS Test: {dist.replace('ks_','').replace('_',' ').title()}")
        plt.xlabel("KS Statistic")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{dist}.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"    ✅ Saved figure: {out_path}")

if __name__ == "__main__":
    print("▶️  Starting KS‐analysis script")
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "../data")
    print(f"Looking for session files in: {data_dir}")
    try:
        print("Directory contents:", os.listdir(data_dir))
    except FileNotFoundError:
        print("❌ Data directory not found—please check the path.")
        exit(1)

    sessions = range(15, 19)
    all_dfs = []

    for s in sessions:
        fname = f"session_{s}_results.json"
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            print(f"  ✔️ Found {fname}")
            posts = load_session_data(fpath)
            ks_df = compute_ks_features(posts)
            print(f"    → Computed KS for {len(ks_df)} users "
                  f"(bots={ks_df['is_bot'].sum()}, humans={len(ks_df)-ks_df['is_bot'].sum()})")
            all_dfs.append(ks_df)
        else:
            print(f"  ⚠️  Missing {fname}, skipping")

    if not all_dfs:
        print("❌ No session files processed. Exiting.")
        exit(1)

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"▶️  Total users aggregated: {len(merged)}")
    save_ks_figures(merged)
    print("🏁  Done.")
