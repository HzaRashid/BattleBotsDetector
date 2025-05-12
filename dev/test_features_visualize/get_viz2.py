import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import os
from scipy.stats import variation, entropy
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
data_dir = os.path.join(os.path.dirname(__file__), "../data")
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

# Compute ARIMA residuals
def fit_arima(time_diffs):
    if len(time_diffs) < 5:
        return np.nan
    model = ARIMA(time_diffs, order=(1, 1, 1))
    result = model.fit()
    return np.mean(np.abs(result.resid))  # Mean absolute residual

# Compute Coefficient of Variation (CV)
def compute_cv(time_diffs):
    if len(time_diffs) < 2:
        return np.nan
    return variation(time_diffs)

# Compute Fourier peak power
def compute_fourier_peak(time_diffs):
    if len(time_diffs) < 2:
        return np.nan
    freqs, power = welch(time_diffs, nperseg=min(len(time_diffs), 256))
    return np.max(power)

# Compute Burstiness Index
def compute_burstiness_index(time_diffs):
    if len(time_diffs) < 2:
        return np.nan
    sigma = np.std(time_diffs)
    mu = np.mean(time_diffs)
    if sigma + mu == 0:
        return np.nan
    return (sigma - mu) / (sigma + mu)

# Compute Time-of-Day entropy and peak activity hour
def compute_time_of_day_features(df):
    if df.empty:
        return np.nan, np.nan
    hours = df["created_at"].dt.hour
    hour_counts = hours.value_counts(normalize=True).sort_index()
    time_entropy = entropy(hour_counts) if len(hour_counts) > 1 else np.nan
    peak_hour = hour_counts.idxmax() if not hour_counts.empty else np.nan
    return time_entropy, peak_hour

# Compute Session Length features
def compute_session_features(df, session_gap=600):
    if df.empty:
        return np.nan, np.nan
    time_diffs = df["created_at"].diff().dt.total_seconds()
    session_ids = (time_diffs > session_gap).cumsum()
    num_sessions = session_ids.nunique()
    avg_session_length = len(df) / num_sessions if num_sessions > 0 else np.nan
    return num_sessions, avg_session_length

# Load dataset
file_path = os.path.join(data_dir, f"session_13_results.json")
posts_df = load_session_data(file_path)

# Compute features for each user
features_list = []
for user_id, group in posts_df.groupby("author_id"):
    time_diffs = compute_inter_arrival_times(group)
    time_entropy, peak_hour = compute_time_of_day_features(group)
    num_sessions, avg_session_length = compute_session_features(group)
    features_list.append({
        "user_id": user_id,
        "arima_resid": fit_arima(time_diffs),
        "cv": compute_cv(time_diffs),
        "fourier_peak": compute_fourier_peak(time_diffs),
        "burstiness_index": compute_burstiness_index(time_diffs),
        "time_entropy": time_entropy,
        "peak_hour": peak_hour,
        "num_sessions": num_sessions,
        "avg_session_length": avg_session_length,
        "is_bot": bool(re.search(r"[a-zA-Z]", str(user_id)))
    })

# Convert to DataFrame
features_df = pd.DataFrame(features_list).dropna()

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df[["cv", "time_entropy", "avg_session_length", "burstiness_index"]])

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
features_df["cluster"] = kmeans.fit_predict(scaled_features)

# Visualizations
plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_df["cv"], y=features_df["time_entropy"], hue=features_df["cluster"], palette="viridis", alpha=0.7)
plt.xlabel("Coefficient of Variation (CV)")
plt.ylabel("Time Entropy")
plt.title("Clustering of Users (CV vs. Time Entropy)")
plt.legend(title="Cluster")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_df["avg_session_length"], y=features_df["time_entropy"], hue=features_df["cluster"], palette="viridis", alpha=0.7)
plt.xlabel("Average Tweets per Session")
plt.ylabel("Time Entropy")
plt.title("Clustering of Users (Tweets per Session vs. Time Entropy)")
plt.legend(title="Cluster")
plt.show()


# Scatter plot of CV vs. Time Entropy (Humans vs. Bots)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=features_df["cv"], 
    y=features_df["time_entropy"], 
    hue=features_df["is_bot"], 
    palette={True: "red", False: "blue"},
    alpha=0.7
)
plt.xlabel("Coefficient of Variation (CV)")
plt.ylabel("Time Entropy")
plt.title("Humans vs. Bots: CV vs. Time Entropy")
plt.legend(title="User Type", labels=["Human", "Bot"])
plt.show()

# Scatter plot of Tweets per Session vs. Time Entropy (Humans vs. Bots)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=features_df["avg_session_length"], 
    y=features_df["time_entropy"], 
    hue=features_df["is_bot"], 
    palette={True: "red", False: "blue"},
    alpha=0.7
)
plt.xlabel("Average Tweets per Session")
plt.ylabel("Time Entropy")
plt.title("Humans vs. Bots: Tweets per Session vs. Time Entropy")
plt.legend(title="User Type", labels=["Human", "Bot"])
plt.show()
