from abc_classes import ADetector
from teams_classes import DetectionMark
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.preprocessing import MinMaxScaler

class Detector(ADetector):
    def __init__(self):
        
        # Load the pre-trained SentenceTransformer model
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Load the pre-trained Random Forest classifier (ensure this was trained with our new features)
        with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as model_file:
            self.clf = pickle.load(model_file)

        # Load the pre-fitted BERTopic model from training
        with open(os.path.join(model_dir, "bertopic_model.pkl"), "rb") as topic_file:
            self.topic_model = pickle.load(topic_file)
    
    def detect_bot(self, session_data):
        # Process new session data to extract features in the same way as training
        feature_vectors = self.process_data(session_data)
        prediction_probs = self.clf.predict_proba(feature_vectors)[:, 1]
        
        # Normalize confidence scores between 0 and 100
        if len(prediction_probs) > 1:
            scaler = MinMaxScaler(feature_range=(0, 100))
            normalized_confidences = scaler.fit_transform(prediction_probs.reshape(-1, 1)).flatten()
        else:
            normalized_confidences = prediction_probs * 100
        
        marked_accounts = []
        for i, user in enumerate(session_data.users):
            predicted_class = int(normalized_confidences[i]) >= 50  # classify as bot if confidence is 50% or higher
            print(predicted_class, int(normalized_confidences[i]), user['id'])
            marked_accounts.append(DetectionMark(user_id=user['id'], confidence=int(normalized_confidences[i]), bot=predicted_class))
        
        return marked_accounts

    def process_data(self, session_data):
        """
        Processes new session data to match the training pipeline:
         - Builds DataFrames for users and posts.
         - Cleans tweet text.
         - Computes embeddings and topics using the pre-fitted models.
         - Aggregates tweet-level features to user-level.
         - Constructs the final feature matrix.
        """
        # Build DataFrames from session data
        users_df = pd.DataFrame(session_data.users)
        posts_df = pd.DataFrame(session_data.posts)
        print(users_df.columns)
        # Ensure posts_df has only the needed columns and rename for consistency
        posts_df = posts_df[['author_id', 'text', 'created_at']].rename(columns={"author_id": "user_id"})
        users_df = users_df.rename(columns={"id": "user_id"})
        
        # Clean text in posts and convert created_at to datetime
        posts_df["cleaned_text"] = posts_df["text"].apply(self.preprocess_text)
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'], errors='coerce')
        
        # Compute sentence embeddings for each tweet
        texts = posts_df["cleaned_text"].fillna("").tolist()
        embeddings = self.sentence_transformer.encode(texts, convert_to_numpy=True)
        posts_df['embedding'] = list(embeddings)
        
        # Obtain topics using the pre-fitted BERTopic model
        topics, _ = self.topic_model.transform(texts)
        posts_df['topic'] = topics
        
        # Aggregate tweet-level features to user-level
        user_features = self.aggregate_user_features(posts_df)

        # Merge aggregated features with users_df (to ensure every user is included)
        # Drop tweet_count from users_df if it's not needed
        users_df = users_df.drop(columns=['tweet_count'], errors='ignore')
        user_features = users_df.merge(user_features, on="user_id", how="left")
        
        # For users with no tweets, fill missing features with default values
        embedding_dim = embeddings.shape[1] if len(embeddings) > 0 else 384
        user_features['embedding_mean'] = user_features['embedding_mean'].apply(
            lambda x: x if isinstance(x, np.ndarray) else np.zeros(embedding_dim)
        )
        user_features[['most_common_topic', 'tweet_time_mean', 'tweet_time_std', 'tweet_count']] = user_features[
            ['most_common_topic', 'tweet_time_mean', 'tweet_time_std', 'tweet_count']
        ].fillna(0)
        
        # Create the final feature matrix
        X = np.hstack((
            np.vstack(user_features['embedding_mean']),
            user_features[['most_common_topic', 'tweet_time_mean', 'tweet_time_std', 'tweet_count']].values
        ))
        
        return X

    def aggregate_user_features(self, data_df):
        """
        Aggregates tweet-level features to user-level.
         - Computes the mean embedding for each user.
         - Determines the most common (non-noise) topic.
         - Computes mean and std of tweet timestamps and tweet count.
        """
        data_df = data_df.copy()
        data_df['created_at'] = pd.to_datetime(data_df['created_at'], errors='coerce')
        
        user_features = data_df.groupby('user_id').agg({
            'embedding': lambda x: np.mean(np.vstack(x), axis=0),
            'topic': lambda x: np.bincount([i for i in x if i >= 0]).argmax() if any(i >= 0 for i in x) else -1,
            'created_at': [
                lambda x: x.dropna().astype('int64').mean(),  # tweet_time_mean
                lambda x: x.dropna().astype('int64').std(),   # tweet_time_std
                'count'                                       # tweet_count
            ]
        })
        # Flatten the MultiIndex columns and rename
        user_features.columns = ['embedding_mean', 'most_common_topic', 'tweet_time_mean', 'tweet_time_std', 'tweet_count']
        return user_features.reset_index()

    def preprocess_text(self, text):
        text = re.sub(r"https?://\S+", "<URLURL>", text)
        text = re.sub(r"@\w+", "<UsernameMention>", text)
        text = re.sub(r"#\w+", "<HashtagMention>", text)
        return text
