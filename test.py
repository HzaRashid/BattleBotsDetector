import pandas as pd
import json
import re
import os
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

# Load spaCy's English model (tokenizer, stopwords, lemmatizer)
nlp = spacy.load("en_core_web_sm", enable=['tokenizer', 'stopwords','lemmatizer'])

model_dir = f'{os.path.dirname(__file__)}/DetectorTemplate/DetectorCode/models'
print(model_dir)

def preprocess_text(text):
    """
    Preprocesses a tweet by:
    1. Tokenizing it with spaCy
    2. Replacing URLs, mentions, and hashtags with special tokens
    3. Removing stopwords
    4. Removing punctuation
    5. Lowercasing all words
    6. Applying lemmatization
    """
    # Replace URLs
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    
    # Replace mentions
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    
    # Replace hashtags
    text = re.sub(r"#\w+", "<HashtagMention>", text)

    # Tokenize with spaCy
    doc = nlp(text)

    # Extract lemmatized tokens, removing stopwords and punctuation, and converting to lowercase
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(tokens)

def process_data():
    # Load dataset
    with open("./data/session_10_results.json", "r", encoding="utf-8") as file:
        dataset1 = json.load(file)

    with open("./data/session_3_results.json", "r", encoding="utf-8") as file:
        dataset2 = json.load(file)

    with open("./data/session_4_results.json", "r", encoding="utf-8") as file:
        dataset3 = json.load(file)

    # Convert JSON to DataFrames
    users_df = pd.DataFrame(dataset1["users"] + dataset2["users"] + dataset3["users"])
    posts_df = pd.DataFrame(dataset1["posts"] + dataset2["posts"] + dataset3["posts"])

    # Keep only relevant columns
    posts_df = posts_df[['author_id', 'text', 'created_at']]
    users_df = users_df[['user_id', 'is_bot']]

    # Preprocess tweets
    posts_df["cleaned_text"] = posts_df["text"].apply(preprocess_text)

    # Combine all tweets for each user into a single document
    user_tweets = posts_df.groupby("author_id")["cleaned_text"].apply(lambda x: " ".join(x)).reset_index()

    # Merge processed tweets back to users
    users_df = users_df.merge(user_tweets, left_on="user_id", right_on="author_id", how="left")

    # Drop redundant column
    users_df.drop(columns=["author_id"], inplace=True)

    # Separate features and target variable
    X = users_df[['cleaned_text']]
    y = users_df['is_bot']

    # Train-test split (80% train, 20% test)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X['cleaned_text'], y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply TF-IDF only on the training data
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_text.fillna(""))
    X_test_tfidf = vectorizer.transform(X_test_text.fillna(""))

    # Convert TF-IDF matrices to DataFrames
    feature_names = vectorizer.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
    X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)

    # Train a svm classifier as a baseline
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    clf.fit(X_train_df, y_train)

    # Predictions for svm
    y_pred = clf.predict(X_test_df)

    # Evaluate SVM performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    (print(clf))
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:\n', report)


    # Save the trained model
    model_path = os.path.join(model_dir, "random_forest.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(clf, model_file)

    # Save the TF-IDF vectorizer
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    # print(final_df.head())


if __name__ == "__main__":
    process_data()
