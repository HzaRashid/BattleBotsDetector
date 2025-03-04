from abc_classes import ADetector
from teams_classes import DetectionMark
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# -------------------------
# Preprocessing Functions
# -------------------------
def preprocess_text(text):
    # Replace URLs and mentions as before, but now capture hashtag content.
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    text = re.sub(r"#(\w+)", lambda m: f"<HashtagMention:{m.group(1)}>", text)
    return text

def pad_or_trim(vec, expected):
    vec = np.array(vec)
    if len(vec) < expected:
        return np.concatenate([vec, np.zeros(expected - len(vec))])
    else:
        return vec[:expected]

# -------------------------
# Topic & Similarity Functions
# -------------------------
def compute_topic_distribution_stats(topics, num_topics):
    """
    Compute a normalized histogram of topics as well as summary statistics:
    weighted mean, variance, std, entropy, and dominance ratio.
    """
    valid_topics = [t for t in topics if t >= 0]
    if not valid_topics:
        return (np.zeros(num_topics), 0, 0, 0, 0, 0)
    
    counts = np.bincount(valid_topics, minlength=num_topics)
    total_count = counts.sum()
    distribution = counts / total_count
    
    indices = np.arange(num_topics)
    weighted_mean = np.sum(indices * distribution)
    weighted_variance = np.sum(distribution * (indices - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_variance)
    
    # Compute entropy with a small constant for numerical stability.
    entropy = -np.sum(distribution * np.log(distribution + 1e-10))
    dominance_ratio = distribution.max()
    
    return distribution, weighted_mean, weighted_variance, weighted_std, entropy, dominance_ratio

def compute_bucket_duplicate_counts(times, num_buckets=10, tolerance=1):
    """
    Partition tweet timestamps into buckets and count duplicates within each bucket.
    Uses a tolerance (in seconds) to determine if two posts are duplicates.
    """
    if times.empty:
        return np.zeros(num_buckets)
    times_array = times.values
    buckets = np.array_split(times_array, num_buckets)
    bucket_counts = []
    for bucket in buckets:
        if len(bucket) == 0:
            bucket_counts.append(-1)
        else:
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
    For each topic with at least 2 tweets, compute pairwise cosine similarities
    and return the average of means, medians, and standard deviations across topics.
    """
    unique_topics = set(topics)
    topic_stats = []
    for topic in unique_topics:
        indices = [i for i, t in enumerate(topics) if t == topic]
        if len(indices) < 2:
            continue
        topic_embeddings = embeddings[indices, :]
        sim_matrix = cosine_similarity(topic_embeddings)
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        sim_values = sim_matrix[triu_indices]
        topic_mean = sim_values.mean()
        topic_median = np.median(sim_values)
        topic_std = sim_values.std()
        topic_stats.append((topic_mean, topic_median, topic_std))
    if not topic_stats:
        return np.nan, np.nan, np.nan
    means = [stat[0] for stat in topic_stats]
    medians = [stat[1] for stat in topic_stats]
    stds = [stat[2] for stat in topic_stats]
    overall_mean = np.mean(means)
    overall_median = np.mean(medians)
    overall_std = np.mean(stds)
    return overall_mean, overall_median, overall_std

# -------------------------
# Detector Class
# -------------------------
class Detector(ADetector):
    def __init__(self):
        # Load the SentenceTransformer model (using the same normalization as in training)
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        # Load the pre-trained Random Forest classifier
        with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as model_file:
            self.clf = pickle.load(model_file)
        
        # Expected fixed dimension for bucket duplicate counts (10 buckets)
        self.expected_time_bucket_dim = 10

    def detect_bot(self, session_data):
        feature_vectors = self.process_data(session_data)
        prediction_probs = self.clf.predict_proba(feature_vectors)[:, 1]
        
        scaler = MinMaxScaler(feature_range=(0, 100))
        normalized_confidences = scaler.fit_transform(prediction_probs.reshape(-1, 1)).flatten()

        marked_accounts = []
        for i, user in enumerate(session_data.users):
            predicted_class = int(normalized_confidences[i]) >= 50
            marked_accounts.append(DetectionMark(user_id=user['id'],
                                                 confidence=int(normalized_confidences[i]),
                                                 bot=predicted_class))
        return marked_accounts

    def process_data(self, session_data):
        # Build DataFrames for users and posts.
        users_df = pd.DataFrame(session_data.users)
        posts_df = pd.DataFrame(session_data.posts)

        posts_df = posts_df[['author_id', 'text', 'created_at']].rename(columns={"author_id": "user_id"})
        users_df = users_df.rename(columns={"id": "user_id"})

        posts_df['cleaned_text'] = posts_df['text'].apply(preprocess_text)
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'], errors='coerce')

        # Compute embeddings with normalization (for consistency with training)
        texts = posts_df['cleaned_text'].tolist()
        embeddings = self.sentence_transformer.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Fit a new BERTopic model on the current batch (alternatively, use a fixed model)
        hdbscan_model = HDBSCAN(cluster_selection_method='eom', prediction_data=True)
        topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=self.sentence_transformer)
        topics, _ = topic_model.fit_transform(texts, embeddings)

        posts_df['embedding'] = list(embeddings)
        posts_df['topic'] = topics

        # Aggregate tweet-level features to user-level.
        user_features = self.aggregate_user_features(posts_df)
        users_df = users_df.drop(columns=['tweet_count'], errors='ignore')
        user_features = users_df.merge(user_features, on="user_id", how="left")

        # Ensure embedding_mean has the correct dimension.
        embedding_dim = embeddings.shape[1]
        user_features['embedding_mean'] = user_features['embedding_mean'].apply(
            lambda x: x if isinstance(x, np.ndarray) else np.zeros(embedding_dim)
        )

        # Pad/trim the bucket duplicates vector.
        if 'time_bucket_duplicates' in user_features.columns:
            user_features['time_bucket_duplicates'] = user_features['time_bucket_duplicates'].apply(
                lambda vec: pad_or_trim(vec, self.expected_time_bucket_dim)
            )
        else:
            user_features['time_bucket_duplicates'] = user_features.apply(
                lambda _: np.zeros(self.expected_time_bucket_dim), axis=1
            )

        # Fill missing tweet time stats with zeros.
        user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_time_var']] = user_features[
            ['tweet_time_mean', 'tweet_time_std', 'tweet_time_var']
        ].fillna(0)

        # Build the feature vector by concatenating:
        # [embedding_mean || topic_mean, topic_variance, topic_std, topic_entropy, topic_dominance ||
        #  tweet_time_mean, tweet_time_std, tweet_time_var || time_bucket_duplicates ||
        #  similarity_mean, similarity_median, similarity_std]
        X = np.hstack((
            np.vstack(user_features['embedding_mean']),
            user_features[['topic_mean', 'topic_variance', 'topic_std', 'topic_entropy', 'topic_dominance']].values,
            user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_time_var']].values,
            np.vstack(user_features['time_bucket_duplicates']),
            user_features[['similarity_mean', 'similarity_median', 'similarity_std']].values,
        ))
        return X

    def aggregate_user_features(self, data_df):
        data_df = data_df.copy()
        data_df['created_at'] = pd.to_datetime(data_df['created_at'], errors='coerce', infer_datetime_format=True)

        def user_agg(group):
            # Process tweet times.
            times = group['created_at'].dropna().tolist()
            times_sorted = sorted(times)
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

            # Compute duplicate counts with tolerance.
            bucketed_duplicates = compute_bucket_duplicate_counts(pd.Series(times_sorted), num_buckets=10, tolerance=1)

            # Process embeddings: compute mean and then normalize.
            embeddings = np.vstack(group['embedding'])
            embedding_mean = np.mean(embeddings, axis=0)
            mean_norm = np.linalg.norm(embedding_mean)
            if mean_norm == 0:
                mean_norm = 1
            embedding_mean = embedding_mean / mean_norm

            # Compute topic summary statistics.
            topics = group['topic'].tolist()
            if (group['topic'] >= 0).any():
                num_topics = int(group.loc[group['topic'] >= 0, 'topic'].max()) + 1
            else:
                num_topics = 0
            topic_distribution, topic_mean, topic_variance, topic_std, topic_entropy, topic_dominance = compute_topic_distribution_stats(topics, num_topics)

            # Compute intra-user similarity metrics by topic.
            sim_mean, sim_median, sim_std = compute_intra_user_similarity_by_topic(embeddings, topics)

            return pd.Series({
                'embedding_mean': embedding_mean,
                'topic_distribution': topic_distribution,  # (retained for debugging, not used in classification)
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
            })

        user_features = data_df.groupby('user_id').apply(user_agg).reset_index()
        return user_features
