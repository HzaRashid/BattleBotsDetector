import pandas as pd
import numpy as np
import json
import re
import os
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

# Set a fixed dimension for the topic distribution vector
FIXED_TOPIC_DIM = 361

model_dir = f'{os.path.dirname(__file__)}/../DetectorTemplate/DetectorCode/models'

def preprocess_text(text):
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    text = re.sub(r"#\w+", "<HashtagMention>", text)
    return text

def process_data():
    datasets = []
    for session in ["session_10", "session_11", "session_4"]:
        with open(f"./data/{session}_results.json", "r", encoding="utf-8") as file:
            datasets.append(json.load(file))

    users_df = pd.concat([pd.DataFrame(ds["users"]) for ds in datasets])
    posts_df = pd.concat([pd.DataFrame(ds["posts"]) for ds in datasets])

    posts_df = posts_df[['author_id', 'text', 'created_at']]
    users_df = users_df[['user_id', 'is_bot']]

    combined_df = users_df.merge(posts_df, left_on="user_id", right_on="author_id", how="left")
    return combined_df

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

def pad_or_trim(vec, expected_dim):
    """
    Ensure that 'vec' is a NumPy array of length 'expected_dim'. 
    If it's shorter, pad with zeros; if it's longer, trim it.
    """
    vec = np.array(vec)
    if len(vec) < expected_dim:
        return np.concatenate([vec, np.zeros(expected_dim - len(vec))])
    elif len(vec) > expected_dim:
        return vec[:expected_dim]
    else:
        return vec

def aggregate_user_features(data_df):
    data_df = data_df.copy()
    data_df['created_at'] = pd.to_datetime(
        data_df['created_at'], errors='coerce', infer_datetime_format=True
    )

    # Here we still compute num_topics dynamically but will force the vector to FIXED_TOPIC_DIM later.
    if (data_df['topic'] >= 0).any():
        num_topics = int(data_df.loc[data_df['topic'] >= 0, 'topic'].max()) + 1
    else:
        num_topics = 0

    def user_agg(group):
        embedding_mean = np.mean(np.vstack(group['embedding']), axis=0)
        topics = group['topic'].tolist()
        if any(t >= 0 for t in topics):
            most_common_topic = np.bincount([t for t in topics if t >= 0]).argmax()
        else:
            most_common_topic = -1
        # Compute topic distribution with dynamic length then later pad/trim
        topic_distribution = compute_topic_distribution(topics, num_topics)
        tweet_times = group['created_at'].dropna().astype('int64')
        tweet_time_mean = tweet_times.mean() if not tweet_times.empty else np.nan
        tweet_time_std = tweet_times.std() if not tweet_times.empty else np.nan
        tweet_count = group['created_at'].count()
        is_bot = group['is_bot'].iloc[0]
        return pd.Series({
            'embedding_mean': embedding_mean,
            'most_common_topic': most_common_topic,
            'topic_distribution': topic_distribution,
            'tweet_time_mean': tweet_time_mean,
            'tweet_time_std': tweet_time_std,
            'tweet_count': tweet_count,
            'is_bot': is_bot
        })

    user_features = data_df.groupby('user_id').apply(user_agg).reset_index()
    return user_features

def main():
    # Load the combined data
    data_df = process_data()

    # Split by user to ensure no user appears in both training and test sets
    unique_users = data_df[['user_id', 'is_bot']].drop_duplicates()
    train_users, test_users = train_test_split(
        unique_users,
        test_size=0.2,
        random_state=42,
        stratify=unique_users['is_bot']
    )

    train_df = data_df[data_df['user_id'].isin(train_users['user_id'])]
    test_df = data_df[data_df['user_id'].isin(test_users['user_id'])]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # -------------------------
    # Process training data only
    # -------------------------
    train_df_text = train_df["text"].apply(preprocess_text).fillna("").tolist()
    
    # Initialize the pre-trained embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    train_embeddings = model.encode(train_df_text, convert_to_numpy=True)
    
    # Fit BERTopic (and HDBSCAN) on the training data
    hdbscan_model = HDBSCAN(cluster_selection_method='eom', prediction_data=True)
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=model)
    train_topics, _ = topic_model.fit_transform(train_df_text)

    # Store features in the training dataframe
    train_df['embedding'] = list(train_embeddings)
    train_df['topic'] = train_topics
    
    # Aggregate tweet-level features to user-level for training data
    train_user_features = aggregate_user_features(train_df)

    # -------------------------
    # Process test data using the fitted BERTopic model
    # -------------------------
    test_df_text = test_df["text"].apply(preprocess_text).fillna("").tolist()
    test_embeddings = model.encode(test_df_text, convert_to_numpy=True)
    
    # Transform test texts with the already fitted BERTopic model
    test_topics, _ = topic_model.transform(test_df_text)
    test_df['embedding'] = list(test_embeddings)
    test_df['topic'] = test_topics

    # Aggregate tweet-level features to user-level for test data
    test_user_features = aggregate_user_features(test_df)
    
    # -------------------------
    # Force topic distribution to have a fixed dimension using pad_or_trim
    # -------------------------
    train_user_features['topic_distribution'] = train_user_features['topic_distribution'].apply(
        lambda vec: pad_or_trim(vec, FIXED_TOPIC_DIM)
    )
    test_user_features['topic_distribution'] = test_user_features['topic_distribution'].apply(
        lambda vec: pad_or_trim(vec, FIXED_TOPIC_DIM)
    )

    # -------------------------
    # Prepare features and labels for classification
    # -------------------------
    X_train = np.hstack((
        np.vstack(train_user_features['embedding_mean']),
        np.vstack(train_user_features['topic_distribution']),
        train_user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_count']].values
    ))
    y_train = train_user_features['is_bot'].values

    X_test = np.hstack((
         np.vstack(test_user_features['embedding_mean']),
         np.vstack(test_user_features['topic_distribution']),
         test_user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_count']].values
    ))
    y_test = test_user_features['is_bot'].values
    
    # -------------------------
    # Train and evaluate the classifier
    # -------------------------
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))

    # -------------------------
    # Save models
    # -------------------------
    with open(os.path.join(model_dir, "random_forest.pkl"), "wb") as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    main()
