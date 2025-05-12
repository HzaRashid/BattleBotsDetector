import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, entropy, expon
from statsmodels.tsa.stattools import acf

def compute_inter_arrival_times(df):
    df['time_diff'] = df['created_at'].diff().dt.total_seconds()
    return df['time_diff'].dropna()

def check_exponential_distribution(time_diffs):
    exp_fit = expon.fit(time_diffs)
    ks_stat, p_value = ks_2samp(time_diffs, expon.rvs(*exp_fit, size=len(time_diffs)))
    return ks_stat, p_value

def compute_autocorrelation(time_diffs, lag=10):
    return acf(time_diffs, nlags=lag)

def compute_entropy(time_diffs):
    hist, bin_edges = np.histogram(time_diffs, bins=20, density=True)
    return entropy(hist)

def extract_temporal_features(group):
    group['created_at'] = pd.to_datetime(group['created_at'], errors='coerce', infer_datetime_format=True)
    time_diffs = compute_inter_arrival_times(group)
    if time_diffs.empty:
        return pd.Series({
            'ks_stat': np.nan,
            'ks_p_value': np.nan,
            'autocorr_mean': np.nan,
            'entropy': np.nan,
        })
    
    ks_stat, p_value = check_exponential_distribution(time_diffs)
    autocorr_values = compute_autocorrelation(time_diffs)
    autocorr_mean = np.mean(autocorr_values)
    ent = compute_entropy(time_diffs)
    
    return pd.Series({
        'ks_stat': ks_stat,
        'ks_p_value': p_value,
        'autocorr_mean': autocorr_mean,
        'entropy': ent,
    })

def aggregate_user_features(data_df):
    data_df = data_df.copy()
    data_df['created_at'] = pd.to_datetime(data_df['created_at'], errors='coerce', infer_datetime_format=True)
    user_features = data_df.groupby('user_id').apply(extract_temporal_features).reset_index()
    return user_features
