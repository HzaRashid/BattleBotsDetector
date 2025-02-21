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

data_dir = f'{os.path.dirname(__file__)}/./data'
# os.path.join(data_dir, f'{session}_results.json')
def preprocess_text(text):
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    text = re.sub(r"#\w+", "<HashtagMention>", text)
    return text


def process_data():
    datasets = []
    for session in ["session_10", 
                    "session_11", 
                    "session_12", 
                    "session_4"
                    ]:
        with open(os.path.join(data_dir, f'{session}_results.json'), "r", encoding="utf-8") as file:
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
            # Count occurrences of each timestamp in the bucket
            bucket_series = pd.Series(bucket)
            counts = bucket_series.value_counts()
            # For timestamps with duplicates, add (count - 1)
            dup_sum = int((counts[counts > 1] - 1).sum())
            bucket_counts.append(dup_sum)
    return np.array(bucket_counts)


def user_agg(group):
    # Compute the mean of the tweet embeddings
    embedding_mean = np.mean(np.vstack(group['embedding']), axis=0)
    
    # Determine the most common topic (ignoring -1 values)
    topics = group['topic'].tolist()
    if any(t >= 0 for t in topics):
        most_common_topic = np.bincount([t for t in topics if t >= 0]).argmax()
    else:
        most_common_topic = -1

    # Compute the topic distribution (with dynamic length, to be padded later)
    if (group['topic'] >= 0).any():
        num_topics = int(group.loc[group['topic'] >= 0, 'topic'].max()) + 1
    else:
        num_topics = 0
    topic_distribution = compute_topic_distribution(topics, num_topics)
    
    # Sort the tweet times
    times = group['created_at'].dropna().sort_values()
    
    # Compute inter-post intervals (in seconds)
    if len(times) < 2:
        tweet_time_mean = np.nan
        tweet_time_std = np.nan
    else:
        # Compute differences in nanoseconds, convert to seconds
        intervals = times.diff().dropna().values.astype('int64') / 1e9
        tweet_time_mean = intervals.mean()
        tweet_time_std = intervals.std()
        tweet_time_var = tweet_time_std**2
    
    # Compute the new bucketed duplicate counts feature
    bucketed_duplicates = compute_bucket_duplicate_counts(times, num_buckets=10)
    
    tweet_count = group['created_at'].count()
    is_bot = group['is_bot'].iloc[0]
    
    return pd.Series({
        'embedding_mean': embedding_mean,
        'most_common_topic': most_common_topic,
        'topic_distribution': topic_distribution,
        'tweet_time_mean': tweet_time_mean,         # average interval (in seconds) between posts
        'tweet_time_std': tweet_time_std,           # std of intervals (in seconds) between posts
        'time_bucket_duplicates': bucketed_duplicates,  # new feature: vector of duplicate counts per bucket
        'tweet_count': tweet_count,
        'is_bot': is_bot
    })


def aggregate_user_features(data_df):
    data_df = data_df.copy()
    data_df['created_at'] = pd.to_datetime(
        data_df['created_at'], errors='coerce', infer_datetime_format=True
    )

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
    # train_user_features.to_csv(path_or_buf=os.path.join(os.path.dirname(__file__), 'foo/check_features.csv'))
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
        train_user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_count']].values,
        np.vstack(train_user_features['time_bucket_duplicates'])
    ))
    y_train = train_user_features['is_bot'].values

    X_test = np.hstack((
         np.vstack(test_user_features['embedding_mean']),
         np.vstack(test_user_features['topic_distribution']),
         test_user_features[['tweet_time_mean', 'tweet_time_std', 'tweet_count']].values,
         np.vstack(test_user_features['time_bucket_duplicates'])
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