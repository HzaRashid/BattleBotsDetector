import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from scipy.stats import ks_2samp, expon, norm, lognorm, weibull_min
from statsmodels.tsa.stattools import acf
import os


# Load session data from JSON file
def load_session_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        session_data = json.load(file)
    posts_df = pd.DataFrame(session_data["posts"])
    posts_df["created_at"] = pd.to_datetime(posts_df["created_at"], errors="coerce")
    return posts_df

# Compute inter-arrival times
def compute_inter_arrival_times(df):
    df = df.sort_values("created_at")
    time_diffs = df["created_at"].diff().dt.total_seconds()
    return time_diffs.dropna()

# Compute entropy
# def compute_entropy(time_diffs):
#     if time_diffs.empty:
#         return np.nan
#     hist, _ = np.histogram(time_diffs, bins=20, density=True)
#     return entropy(hist)

# Compute autocorrelation
def compute_autocorrelation(time_diffs, lag=5):
    if len(time_diffs) < 2:
        return np.nan
    return np.mean(acf(time_diffs, nlags=lag, fft=True))

# Perform KS test against multiple distributions
def ks_test_distributions(time_diffs):
    results = {}
    if len(time_diffs) < 2:
        return {dist: (0, 1) for dist in ["exponential", "normal", "lognormal", "weibull"]}
    
    # Exponential distribution
    exp_fit = expon.fit(time_diffs)
    results["exponential"] = ks_2samp(time_diffs, expon.rvs(*exp_fit, size=min(len(time_diffs), 1000)))
    
    # Normal distribution
    norm_fit = norm.fit(time_diffs)
    results["normal"] = ks_2samp(time_diffs, norm.rvs(*norm_fit, size=min(len(time_diffs), 1000)))
    
    # Log-normal distribution
    lognorm_fit = lognorm.fit(time_diffs)
    results["lognormal"] = ks_2samp(time_diffs, lognorm.rvs(*lognorm_fit, size=min(len(time_diffs), 1000)))
    
    # Weibull distribution
    weibull_fit = weibull_min.fit(time_diffs)
    results["weibull"] = ks_2samp(time_diffs, weibull_min.rvs(*weibull_fit, size=min(len(time_diffs), 1000)))
    
    return {dist: (stat, pval) for dist, (stat, pval) in results.items()}

# Compute KS tests for each user
def compute_ks_features(posts_df):
    ks_test_results = []
    for user_id, group in posts_df.groupby("author_id"):
        time_diffs = compute_inter_arrival_times(group)
        ks_results = ks_test_distributions(time_diffs)
        ks_test_results.append({
            "user_id": user_id,
            "ks_exponential": ks_results["exponential"][0], "p_exponential": ks_results["exponential"][1],
            "ks_normal": ks_results["normal"][0], "p_normal": ks_results["normal"][1],
            "ks_lognormal": ks_results["lognormal"][0], "p_lognormal": ks_results["lognormal"][1],
            "ks_weibull": ks_results["weibull"][0], "p_weibull": ks_results["weibull"][1]
        })
    return pd.DataFrame(ks_test_results)

# Visualize KS statistic distributions
def plot_ks_histograms(ks_test_results_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("KS Statistics Distribution (Bots vs. Humans)")
    
    distributions = ["ks_exponential", "ks_normal", "ks_lognormal", "ks_weibull"]
    ks_test_results_df["is_bot"] = ks_test_results_df["user_id"].astype(str).str.contains("[a-zA-Z]")
    
    for i, dist in enumerate(distributions):
        ax = axes[i // 2, i % 2]
        bots = ks_test_results_df[ks_test_results_df["is_bot"]][dist]
        humans = ks_test_results_df[~ks_test_results_df["is_bot"]][dist]
        
        ax.hist(bots, bins=20, alpha=0.6, label="Bots", density=True)
        ax.hist(humans, bins=20, alpha=0.6, label="Humans", density=True)
        ax.set_title(f"KS Test: {dist.replace('_', ' ').title()}")
        ax.set_xlabel("KS Statistic")
        ax.set_ylabel("Density")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Main execution
def main(file_path):
    posts_df = load_session_data(file_path)
    ks_results_df = compute_ks_features(posts_df)
    plot_ks_histograms(ks_results_df)
    return ks_results_df

# Example usage
data_dir = os.path.join(os.path.dirname(__file__), "../data")
ks_results_df = main(os.path.join(data_dir, f"session_13_results.json"))
