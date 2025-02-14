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


class Detector(ADetector):
    def __init__(self):
        # Load the pre-trained SentenceTransformer model
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Load the pre-trained Random Forest classifier (trained with our new features)
        with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as model_file:
            self.clf = pickle.load(model_file)
        
        # Set the expected topic distribution dimension from training.
        # For example, if training produced topic distributions of length 355:
        self.expected_topic_dim = 361

    """ -> 
    DETECT BOT 
    <- """
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

        # Aggregate tweet-level features to user-level, including topic distribution
        user_features = self.aggregate_user_features(posts_df)
        # Merge with users_df to ensure all users are present
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

        # Build feature vector: [embedding_mean || topic_distribution || tweet_time_mean, tweet_time_std, tweet_count]
        X = np.hstack((
            np.vstack(user_features['embedding_mean']),
            np.vstack(user_features['topic_distribution']),
            user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_count']].values
        ))
        return X

    def aggregate_user_features(self, data_df):
        data_df = data_df.copy()
        data_df['created_at'] = pd.to_datetime(data_df['created_at'],
                                                errors='coerce',
                                                infer_datetime_format=True)

        # Determine number of topics (ignoring -1) from this batch
        # (We will later force the vector to self.expected_topic_dim)
        if (data_df['topic'] >= 0).any():
            num_topics = int(data_df.loc[data_df['topic'] >= 0, 'topic'].max()) + 1
        else:
            num_topics = 0

        def user_agg(group):
            embedding_mean = np.mean(np.vstack(group['embedding']), axis=0)
            topics = group['topic'].tolist()
            # most_common_topic is computed here (retained for debugging if needed)
            if any(t >= 0 for t in topics):
                most_common_topic = np.bincount([t for t in topics if t >= 0]).argmax()
            else:
                most_common_topic = -1
            topic_distribution = compute_topic_distribution(topics, num_topics)
            tweet_times = group['created_at'].dropna().astype('int64')
            tweet_time_mean = tweet_times.mean() if not tweet_times.empty else np.nan
            tweet_time_std = tweet_times.std() if not tweet_times.empty else np.nan
            tweet_count = group['created_at'].count()
            return pd.Series({
                'embedding_mean': embedding_mean,
                'most_common_topic': most_common_topic,  # retained for reference
                'topic_distribution': topic_distribution,
                'tweet_time_mean': tweet_time_mean,
                'tweet_time_std': tweet_time_std,
                'tweet_count': tweet_count,
            })

        user_features = data_df.groupby('user_id').apply(user_agg).reset_index()
        return user_features

    def preprocess_text(self, text):
        text = re.sub(r"https?://\S+", "<URLURL>", text)
        text = re.sub(r"@\w+", "<UsernameMention>", text)
        text = re.sub(r"#\w+", "<HashtagMention>", text)
        return text
