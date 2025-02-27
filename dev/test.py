import pandas as pd
import numpy as np
import json
import re
import os
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle

# -------------------------
# Preprocessing and Data Loading
# -------------------------
def preprocess_text(text):
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    # text = re.sub(r"#\w+", "<HashtagMention>", text)
    return text

def process_data():
    datasets = []
    data_dir = os.path.join(os.path.dirname(__file__), "./data")
    # Adjust sessions as needed
    sessions = [
        "session_13", 
        ]
    for session in sessions:
        with open(os.path.join(data_dir, f"{session}_results.json"), "r", encoding="utf-8") as file:
            datasets.append(json.load(file))
    users_df = pd.concat([pd.DataFrame(ds["users"]) for ds in datasets])
    posts_df = pd.concat([pd.DataFrame(ds["posts"]) for ds in datasets])
    posts_df = posts_df[['author_id', 'text', 'created_at']]
    users_df = users_df[['user_id', 'is_bot']]
    combined_df = users_df.merge(posts_df, left_on="user_id", right_on="author_id", how="left")
    return combined_df

# -------------------------
# Feature Engineering Functions
# -------------------------

# -------------------------
# Topic Distribution Transformation Functions (unchanged)
# -------------------------
def compute_topic_distribution_stats(topics, num_topics):
    """
    Given a list of topic indices and the total number of topics,
    compute robust summary statistics that are invariant to the specific topic labels.
    
    Returns:
        distribution (np.array): The normalized topic histogram.
        weighted_mean (float): The weighted mean of topic indices.
        weighted_variance (float): The variance over topic indices.
        weighted_std (float): Standard deviation.
        entropy (float): Entropy of the distribution.
        dominance_ratio (float): Proportion of the most common topic.
    """
    valid_topics = [t for t in topics if t >= 0]
    if not valid_topics:
        # Return zeros or NaNs as appropriate when no valid topics are present.
        return (np.zeros(num_topics), 0, 0, 0, 0, 0)
    
    counts = np.bincount(valid_topics, minlength=num_topics)
    total_count = counts.sum()
    distribution = counts / total_count
    
    # Weighted mean and variance
    indices = np.arange(num_topics)
    weighted_mean = np.sum(indices * distribution)
    weighted_variance = np.sum(distribution * (indices - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_variance)
    
    # Entropy: measures the uncertainty in the distribution.
    entropy = -np.sum(distribution * np.log(distribution + 1e-10))
    
    # Dominance ratio: the proportion of the most frequent topic.
    dominance_ratio = distribution.max()
    
    return distribution, weighted_mean, weighted_variance, weighted_std, entropy, dominance_ratio

def pad_or_trim(vec, expected_dim):
    vec = np.array(vec)
    if len(vec) < expected_dim:
        return np.concatenate([vec, np.zeros(expected_dim - len(vec))])
    elif len(vec) > expected_dim:
        return vec[:expected_dim]
    else:
        return vec

def compute_bucket_duplicate_counts(times, num_buckets=10, tolerance=1):
    if times.empty:
        return np.zeros(num_buckets)
    
    times_array = times.values
    buckets = np.array_split(times_array, num_buckets)
    bucket_counts = []
    
    for bucket in buckets:
        if len(bucket) == 0:
            bucket_counts.append(-1)
        else:
            # Convert each time to seconds (as float)
            times_seconds = sorted([pd.Timestamp(t).timestamp() for t in bucket])
            dup_sum = 0
            group_count = 1
            last_time = times_seconds[0]
            for t in times_seconds[1:]:
                if t - last_time <= tolerance:
                    group_count += 1
                else:
                    dup_sum += (group_count - 1)
                    group_count = 1
                last_time = t
            dup_sum += (group_count - 1)
            bucket_counts.append(dup_sum)
    
    return np.array(bucket_counts)


def compute_intra_user_similarity_by_topic(embeddings, topics):
    """
    Given tweet embeddings and their corresponding topic labels for a single user,
    compute pairwise cosine similarity within each topic (only for topics with at least 2 tweets),
    and then average the similarity mean, median, and std across topics.

    Args:
        embeddings (np.array): Array of shape (n_tweets, embedding_dim).
        topics (list or array): List of topic labels for each tweet.

    Returns:
        tuple: (overall_mean, overall_median, overall_std) aggregated over topics,
               or (np.nan, np.nan, np.nan) if no topic has enough tweets.
    """
    unique_topics = set(topics)
    topic_stats = []  # List to store (mean, median, std) for each topic

    for topic in unique_topics:
        # Skip topics with less than 2 tweets
        indices = [i for i, t in enumerate(topics) if t == topic]
        if len(indices) < 2:
            continue
        
        topic_embeddings = embeddings[indices, :]  # subset for this topic
        sim_matrix = cosine_similarity(topic_embeddings)
        # Get upper-triangular values (excluding the diagonal) to avoid redundancy
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        sim_values = sim_matrix[triu_indices]

        topic_mean = sim_values.mean()
        topic_median = np.median(sim_values)
        topic_std = sim_values.std()

        topic_stats.append((topic_mean, topic_median, topic_std))

    if not topic_stats:
        return np.nan, np.nan, np.nan

    # Average the stats across topics
    means = [stat[0] for stat in topic_stats]
    medians = [stat[1] for stat in topic_stats]
    stds = [stat[2] for stat in topic_stats]

    overall_mean = np.mean(means)
    overall_median = np.mean(medians)
    overall_std = np.mean(stds)
    
    return overall_mean, overall_median, overall_std

def compute_token_ratio(text, marker):
    """Compute the ratio of tokens in the tweet equal to the given marker."""
    tokens = text.split()
    if not tokens:
        return 0
    return sum(1 for token in tokens if token == marker) / len(tokens)

def user_agg(group):
    # Convert created_at to datetime and sort
    group['created_at'] = pd.to_datetime(group['created_at'], errors='coerce', infer_datetime_format=True)
    times = group['created_at'].dropna().tolist()
    times_sorted = sorted(times)
    
    # Compute inter-tweet intervals (if possible)
    if len(times_sorted) < 2:
        tweet_time_mean = np.nan
        tweet_time_std = np.nan
        tweet_time_var = np.nan
    else:
        timestamps = [t.timestamp() for t in times_sorted]
        intervals = np.diff(timestamps)
        tweet_time_mean = intervals.mean()
        tweet_time_std = intervals.std()
        tweet_time_var = intervals.var()

    # Compute duplicates as before
    bucketed_duplicates = compute_bucket_duplicate_counts(pd.Series(times_sorted), num_buckets=10)
    
    # Process embeddings
    embeddings = np.vstack(group['embedding'])
    embedding_mean = np.mean(embeddings, axis=0)
    mean_norm = np.linalg.norm(embedding_mean)
    if mean_norm == 0: 
        mean_norm = 1
    embedding_mean = embedding_mean / mean_norm
    
    # Topic distribution stats
    topics = group['topic'].tolist()
    if (group['topic'] >= 0).any():
        num_topics = int(group.loc[group['topic'] >= 0, 'topic'].max()) + 1
    else:
        num_topics = 0
    topic_distribution, topic_mean, topic_variance, topic_std, topic_entropy, topic_dominance = compute_topic_distribution_stats(topics, num_topics)
    sim_mean, sim_median, sim_std = compute_intra_user_similarity_by_topic(embeddings, topics)
    
    is_bot = group['is_bot'].iloc[0]
    
    return pd.Series({
        'embedding_mean': embedding_mean,
        'topic_distribution': topic_distribution,
        'topic_mean': topic_mean,
        'topic_variance': topic_variance,
        'topic_std': topic_std,
        'topic_entropy': topic_entropy,
        'topic_dominance': topic_dominance,
        'tweet_time_mean': tweet_time_mean,
        'tweet_time_std': tweet_time_std,
        'tweet_time_var': tweet_time_var,
        'time_bucket_duplicates': bucketed_duplicates,
        'similarity_mean': sim_mean,
        'similarity_median': sim_median,
        'similarity_std': sim_std,
        'is_bot': is_bot
    })

def aggregate_user_features(data_df):
    data_df = data_df.copy()
    data_df['created_at'] = pd.to_datetime(data_df['created_at'], errors='coerce', infer_datetime_format=True)
    user_features = data_df.groupby('user_id').apply(user_agg).reset_index()
    return user_features

# -------------------------
# Main Pipeline
# -------------------------
def main():
    model_dir = os.path.join(os.path.dirname(__file__), "../DetectorTemplate/DetectorCode/models")
    data_df = process_data()
    
    # Split users into training and testing sets
    unique_users = data_df[['user_id', 'is_bot']].drop_duplicates()

    test_df = data_df[data_df['user_id'].isin(unique_users['user_id'])].reset_index(drop=True)
    
    # -------------------------
    # Process training data (text and embedding)
    # -------------------------
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    hdbscan_model = HDBSCAN(cluster_selection_method='eom', prediction_data=True)
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=model)

    
    # -------------------------
    # Process test data using the fitted BERTopic model
    # -------------------------
    test_texts = test_df["text"].apply(preprocess_text).fillna("").tolist()
    test_embeddings = model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True)
    test_topics, _ = topic_model.fit_transform(test_texts)
    
    test_df['embedding'] = list(test_embeddings)
    test_df['topic'] = test_topics
    
    test_user_features = aggregate_user_features(test_df)
    
    # -------------------------
    # Fix topic distribution dimensions (if needed for fixed-size input)
    # -------------------------
    FIXED_TOPIC_DIM = 361
    test_user_features['topic_distribution'] = test_user_features['topic_distribution'].apply(
        lambda vec: pad_or_trim(vec, FIXED_TOPIC_DIM)
    )

    
    # -------------------------
    # Prepare features for classification
    # -------------------------

    X_test = np.hstack((
        np.vstack(test_user_features['embedding_mean']),
        test_user_features[['topic_mean', 'topic_variance', 'topic_std', 
                            'topic_entropy', 'topic_dominance']].values,
        test_user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_time_var']].values,
        np.vstack(test_user_features['time_bucket_duplicates']),
        test_user_features[['similarity_mean', 'similarity_median', 'similarity_std']].values,
    ))
    y_test = test_user_features['is_bot'].values
    
    # -------------------------
    # Train and evaluate classifier
    # -------------------------
    
    model_dir = os.path.join(os.path.dirname(__file__), '../DetectorTemplate/DetectorCode/models')
    
    # Load the pre-trained Random Forest classifier (trained with our new features)
    with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as model_file:
        clf = pickle.load(model_file)
    
    prediction_probs = clf.predict_proba(X_test)[:, 1]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    normalized_confidences = scaler.fit_transform(prediction_probs.reshape(-1, 1)).flatten()
    y_pred = [int(normalized_confidences[i]) >= 50 for i in range(len(prediction_probs))]

    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))
    for i in range(len(y_pred)):
        if y_pred[i] == False and y_test[i] == True:
            print('--------- missed this one --------')
            user_id = test_user_features['user_id'].iloc[i]
            print("user id:", user_id, '|', 'confidence:', normalized_confidences[i])
            # bot_data = data_df[data_df['user_id']==user_id]
            # print(bot_data)
            print()


if __name__ == "__main__":
    main()
