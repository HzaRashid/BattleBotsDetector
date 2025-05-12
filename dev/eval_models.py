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
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Set a fixed dimension for the topic distribution vector
FIXED_TOPIC_DIM = 361

# -------------------------
# Preprocessing and Data Loading
# -------------------------
def preprocess_text(text):
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    # Retain hashtag contents by capturing the word after '#' and formatting it
    text = re.sub(r"#(\w+)", lambda m: f"<HashtagMention:{m.group(1)}>", text)
    return text
# print(preprocess_text("Can't believe how packed the #arena #is tonight! Support for #our team #is unreal! #HomeC#ourtAdvantage https://t.co/twitter_link @mention @mention"))
def process_data():
    datasets = []
    data_dir = os.path.join(os.path.dirname(__file__), "./data")
    sessions = ["session_10", "session_11", "session_12", "session_4", "session_13"]
    for session in sessions[:4]:
        with open(os.path.join(data_dir, f"{session}_results.json"), "r", encoding="utf-8") as file:
            datasets.append(json.load(file))
    users_df = pd.concat([pd.DataFrame(ds["users"]) for ds in datasets])
    posts_df = pd.concat([pd.DataFrame(ds["posts"]) for ds in datasets])
    posts_df = posts_df[['author_id', 'text', 'created_at']]
    users_df = users_df[['user_id', 'is_bot']]
    combined_df = users_df.merge(posts_df, left_on="user_id", right_on="author_id", how="left")
    return combined_df


# -------------------------
# Topic Distribution Transformation Functions (unchanged)
# -------------------------
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
    vec = np.array(vec)
    if len(vec) < expected_dim:
        return np.concatenate([vec, np.zeros(expected_dim - len(vec))])
    elif len(vec) > expected_dim:
        return vec[:expected_dim]
    else:
        return vec

def compute_burst_count(times, num_buckets=10, tolerance=1):
    if times.empty:
        return np.zeros(num_buckets)
    times_array = times.values
    buckets = np.array_split(times_array, num_buckets)
    bucket_counts = []
    for bucket in buckets:
        if len(bucket) == 0:
            bucket_counts.append(0)
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
    return sum(bucket_counts)

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
    burst_count = compute_burst_count(pd.Series(times_sorted), num_buckets=10)
    
    # Process embeddings
    embeddings = np.vstack(group['embedding'])
    embedding_mean = np.mean(embeddings, axis=0)
    embedding_mean = embedding_mean / (1 if not embedding_mean.any() \
                                       else np.linalg.norm(embedding_mean))
    
    # Topic distribution stats
    topics = group['topic'].tolist()
    if (group['topic'] >= 0).any():
        num_topics = int(group.loc[group['topic'] >= 0, 'topic'].max()) + 1
    else:
        num_topics = 0
    topic_distribution = compute_topic_distribution(topics, num_topics)
    
    
    is_bot = group['is_bot'].iloc[0]
    tweet_count = group['created_at'].count()

    return pd.Series({
        'embedding_mean': embedding_mean,
        'topic_distribution': topic_distribution,
        'tweet_time_mean': tweet_time_mean,
        'tweet_time_std': tweet_time_std,
        'tweet_time_var': tweet_time_var,
        'burst_count': burst_count,
        'tweet_count': tweet_count,
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
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42, stratify=unique_users['is_bot'])
    
    train_df = data_df[data_df['user_id'].isin(train_users['user_id'])].reset_index(drop=True)
    test_df = data_df[data_df['user_id'].isin(test_users['user_id'])].reset_index(drop=True)
    
    # -------------------------
    # Process training data (text and embedding)
    # -------------------------
    train_texts = train_df["text"].apply(preprocess_text).fillna("").tolist()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    train_embeddings = model.encode(train_texts, convert_to_numpy=True)
    
    hdbscan_model = HDBSCAN(cluster_selection_method='eom', prediction_data=True)
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=model)
    train_topics, _ = topic_model.fit_transform(train_texts)
    
    train_df['embedding'] = list(train_embeddings)
    train_df['topic'] = train_topics
    
    train_user_features = aggregate_user_features(train_df)
    
    # print(
    #     train_user_features.iloc[0]['burst_count']
    # )
    
    # -------------------------
    # Process test data using the fitted BERTopic model
    # -------------------------
    test_texts = test_df["text"].apply(preprocess_text).fillna("").tolist()
    test_embeddings = model.encode(test_texts, convert_to_numpy=True)
    
    test_topics, _ = topic_model.transform(test_texts)
    
    test_df['embedding'] = list(test_embeddings)
    test_df['topic'] = test_topics
    
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
    # Combine features for classification:
    # We include the mean embeddings, topic summary statistics, tweet time statistics,
    # and time bucket duplicate counts.
    # -------------------------
    X_train = np.hstack((
        np.vstack(train_user_features['embedding_mean']),
        np.vstack(train_user_features['topic_distribution']),
        train_user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_time_var', 'tweet_count']].values,
        np.vstack(train_user_features['burst_count']),
    ))
    y_train = train_user_features['is_bot'].values

    X_test = np.hstack((
        np.vstack(test_user_features['embedding_mean']),
        np.vstack(test_user_features['topic_distribution']),
        test_user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_time_var', 'tweet_count']].values,
        np.vstack(test_user_features['burst_count']),
    ))
    y_test = test_user_features['is_bot'].values
    
    # -------------------------
    # Train and evaluate classifier
    # -------------------------
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))
    
    # -------------------------
    # Save the trained model
    # -------------------------
    with open(os.path.join(model_dir, "random_forest.pkl"), "wb") as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    main()
