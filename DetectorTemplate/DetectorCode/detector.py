from abc_classes import ADetector
from teams_classes import DetectionMark
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def compute_topic_distribution(topics, num_topics):
    """
    Given an iterable of topic IDs, filter out unassigned topics (-1) and return
    a normalized histogram vector of length num_topics.
    """
    valid_topics = [t for t in topics if t >= 0]
    if not valid_topics:
        return np.zeros(num_topics)
    counts = np.bincount(valid_topics, minlength=num_topics)
    return counts / counts.sum()

def pad_or_trim(vec, expected):
    """
    Ensure that vec is a NumPy array of length 'expected'. If it's shorter,
    pad with zeros; if it's longer, trim it.
    """
    vec = np.array(vec)
    if len(vec) < expected:
        return np.concatenate([vec, np.zeros(expected - len(vec))])
    else:
        return vec[:expected]

def compute_bucket_duplicate_counts(times, num_buckets=10):
    """
    Given a sorted pandas Series of tweet times, partition the times into
    `num_buckets` buckets. For each bucket, compute the sum of duplicate posts,
    where for each timestamp that occurs more than once, we add (count - 1).
    If all timestamps in a bucket are unique, the bucket's value will be 0.
    """
    if times.empty:
        return np.zeros(num_buckets)
    
    # Convert to numpy array and partition into num_buckets buckets
    times_array = times.values
    buckets = np.array_split(times_array, num_buckets)
    bucket_counts = []
    for bucket in buckets:
        if len(bucket) == 0:
            bucket_counts.append(-1)
        else:
            bucket_series = pd.Series(bucket)
            counts = bucket_series.value_counts()
            dup_sum = int((counts[counts > 1] - 1).sum())
            bucket_counts.append(dup_sum)
    return np.array(bucket_counts)

class Detector(ADetector):
    def __init__(self):
        # Load the pre-trained SentenceTransformer model
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Load the pre-trained Random Forest classifier (trained with our new features)
        with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as model_file:
            self.clf = pickle.load(model_file)
        
        # Expected dimensions from training
        self.expected_topic_dim = 361
        self.expected_time_bucket_dim = 10

    def detect_bot(self, session_data):
        feature_vectors = self.process_data(session_data)
        prediction_probs = self.clf.predict_proba(feature_vectors)[:, 1]
        
        scaler = MinMaxScaler(feature_range=(0, 100))
        normalized_confidences = scaler.fit_transform(prediction_probs.reshape(-1, 1)).flatten()

        marked_accounts = []
        for i, user in enumerate(session_data.users):
            predicted_class = int(normalized_confidences[i]) >= 50
            # print(predicted_class, int(normalized_confidences[i]), user['id'])
            marked_accounts.append(DetectionMark(user_id=user['id'],
                                                 confidence=int(normalized_confidences[i]),
                                                 bot=predicted_class))
        return marked_accounts

    def process_data(self, session_data):
        # Build dataframes for users and posts
        users_df = pd.DataFrame(session_data.users)
        posts_df = pd.DataFrame(session_data.posts)

        posts_df = posts_df[['author_id', 'text', 'created_at']].rename(columns={"author_id": "user_id"})
        users_df = users_df.rename(columns={"id": "user_id"})

        posts_df['cleaned_text'] = posts_df['text'].apply(self.preprocess_text)
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'], errors='coerce')

        # Compute embeddings and topics
        texts = posts_df['cleaned_text'].tolist()
        embeddings = self.sentence_transformer.encode(texts, convert_to_numpy=True)

        # Fit a new BERTopic model with HDBSCAN for this batch
        hdbscan_model = HDBSCAN(cluster_selection_method='eom', prediction_data=True)
        topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=self.sentence_transformer)
        topics, _ = topic_model.fit_transform(texts, embeddings)

        posts_df['embedding'] = list(embeddings)
        posts_df['topic'] = topics

        # Aggregate tweet-level features to user-level, including topic distribution and duplicate counts
        user_features = self.aggregate_user_features(posts_df)
        users_df = users_df.drop(columns=['tweet_count'], errors='ignore')
        user_features = users_df.merge(user_features, on="user_id", how="left")

        # Ensure that embedding_mean is an array of the proper dimension
        embedding_dim = embeddings.shape[1]
        user_features['embedding_mean'] = user_features['embedding_mean'].apply(
            lambda x: x if isinstance(x, np.ndarray) else np.zeros(embedding_dim)
        )

        # Pad/trim topic_distribution to match the expected dimension
        if 'topic_distribution' in user_features.columns:
            user_features['topic_distribution'] = user_features['topic_distribution'].apply(
                lambda vec: pad_or_trim(vec, self.expected_topic_dim)
            )
        else:
            user_features['topic_distribution'] = user_features.apply(
                lambda _: np.zeros(self.expected_topic_dim), axis=1
            )

        # Fill missing time features with zeros
        user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_count']] = user_features[
            ['tweet_time_mean', 'tweet_time_std', 'tweet_count']
        ].fillna(0)

        # Pad/trim time_bucket_duplicates to the expected length
        if 'time_bucket_duplicates' in user_features.columns:
            user_features['time_bucket_duplicates'] = user_features['time_bucket_duplicates'].apply(
                lambda vec: pad_or_trim(vec, self.expected_time_bucket_dim)
            )
        else:
            user_features['time_bucket_duplicates'] = user_features.apply(
                lambda _: np.zeros(self.expected_time_bucket_dim), axis=1
            )

        # Build feature vector by concatenating all features:
        # [embedding_mean || topic_distribution || tweet_time_mean, tweet_time_std, tweet_count || time_bucket_duplicates]
        X = np.hstack((
            np.vstack(user_features['embedding_mean']),
            np.vstack(user_features['topic_distribution']),
            user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_count']].values,
            np.vstack(user_features['time_bucket_duplicates'])
        ))
        return X

    def aggregate_user_features(self, data_df):
        data_df = data_df.copy()
        data_df['created_at'] = pd.to_datetime(data_df['created_at'],
                                                errors='coerce',
                                                infer_datetime_format=True)

        # Determine number of topics (ignoring -1) from this batch
        if (data_df['topic'] >= 0).any():
            num_topics = int(data_df.loc[data_df['topic'] >= 0, 'topic'].max()) + 1
        else:
            num_topics = 0

        def user_agg(group):
            # Compute mean of the tweet embeddings
            embedding_mean = np.mean(np.vstack(group['embedding']), axis=0)
            
            topics = group['topic'].tolist()
            # Determine the most common topic (ignoring -1)
            if any(t >= 0 for t in topics):
                most_common_topic = np.bincount([t for t in topics if t >= 0]).argmax()
            else:
                most_common_topic = -1
            
            # Compute topic distribution
            topic_distribution = compute_topic_distribution(topics, num_topics)
            
            # Sort the tweet times and compute inter-post intervals (in seconds)
            times = group['created_at'].dropna().sort_values()
            if len(times) < 2:
                tweet_time_mean = np.nan
                tweet_time_std = np.nan
            else:
                # Compute differences in nanoseconds, convert to minutes
                intervals = times.diff().dropna().values.astype('int64') / 1e9
                tweet_time_mean = intervals.mean()
                tweet_time_std = intervals.std()
            
            tweet_count = group['created_at'].count()
            
            # Compute bucketed duplicate counts feature
            time_bucket_duplicates = compute_bucket_duplicate_counts(times, num_buckets=10)
            
            return pd.Series({
                'embedding_mean': embedding_mean,
                'most_common_topic': most_common_topic,  # retained for reference/debugging
                'topic_distribution': topic_distribution,
                'tweet_time_mean': tweet_time_mean,
                'tweet_time_std': tweet_time_std,
                'tweet_count': tweet_count,
                'time_bucket_duplicates': time_bucket_duplicates
            })

        user_features = data_df.groupby('user_id').apply(user_agg).reset_index()
        return user_features

    def preprocess_text(self, text):
        text = re.sub(r"https?://\S+", "<URLURL>", text)
        text = re.sub(r"@\w+", "<UsernameMention>", text)
        text = re.sub(r"#\w+", "<HashtagMention>", text)
        return text
