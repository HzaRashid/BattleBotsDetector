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

def extract_features_from_tweets(data_df):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(data_df['cleaned_text'].tolist(), convert_to_numpy=True)

    hdbscan_model = HDBSCAN(cluster_selection_method='eom', prediction_data=True)
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=model)
    topics, _ = topic_model.fit_transform(data_df['cleaned_text'].tolist())

    data_df['embedding'] = list(embeddings)
    data_df['topic'] = topics

    return data_df, topic_model


def aggregate_user_features(data_df):
    data_df = data_df.copy()
    data_df['created_at'] = pd.to_datetime(data_df['created_at'], errors='coerce')

    user_features = data_df.groupby('user_id').agg({
        'embedding': lambda x: np.mean(np.vstack(x), axis=0),
        'topic': lambda x: np.bincount([i for i in x if i >= 0]).argmax() if any(i >= 0 for i in x) else -1,

        'created_at': [
            lambda x: x.dropna().astype('int64').mean(),  # tweet_time_mean
            lambda x: x.dropna().astype('int64').std(),   # tweet_time_std
            'count'                                       # tweet_count
        ],
        'is_bot': 'first'  # preserve the is_bot label for each user
    })

    # Flatten the MultiIndex columns and rename them
    user_features.columns = [
        'embedding_mean', 
        'most_common_topic', 
        'tweet_time_mean', 
        'tweet_time_std', 
        'tweet_count',
        'is_bot'
    ]
    return user_features.reset_index()


def main():
    # Load the combined data
    data_df = process_data()

    # Split by user to ensure no user appears in both training and test sets
    unique_users = data_df[['user_id', 'is_bot']].drop_duplicates()
    # print(data_df)
    train_users, test_users = train_test_split(
        unique_users,
        test_size=0.2,
        random_state=42,
        stratify=unique_users['is_bot']
    )
    # print('foo')
    train_df = data_df[data_df['user_id'].isin(train_users['user_id'])]
    test_df = data_df[data_df['user_id'].isin(test_users['user_id'])]

    # print('bar')
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
    # Prepare features and labels for classification
    # -------------------------
    X_train = np.hstack((
         np.vstack(train_user_features['embedding_mean']),
         train_user_features[['most_common_topic', 'tweet_time_mean', 'tweet_time_std', 'tweet_count']].values
    ))
    y_train = train_user_features['is_bot'].values

    X_test = np.hstack((
         np.vstack(test_user_features['embedding_mean']),
         test_user_features[['most_common_topic', 'tweet_time_mean', 'tweet_time_std', 'tweet_count']].values
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
    
    # Save the Random Forest classifier
    with open(os.path.join(model_dir, "random_forest.pkl"), "wb") as f:
        pickle.dump(clf, f)
        
    # Save the BERTopic model
    with open(os.path.join(model_dir, "bertopic_model.pkl"), "wb") as f:
        pickle.dump(topic_model, f)


if __name__ == "__main__":
    main()
