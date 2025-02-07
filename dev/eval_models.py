from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
import spacy
import json
import re
import os


# Load spaCy's English model
nlp = spacy.load("en_core_web_sm", enable=['tokenizer', 'stopwords', 'lemmatizer'])

model_dir = f'{os.path.dirname(__file__)}../DetectorTemplate/DetectorCode/models'
print(model_dir)

# "all-MiniLM-L6-v2" is a fairly lightweight model (~82MB) 
# that generates 384-dimensional sentence embeddings.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def preprocess_text(text):
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    text = re.sub(r"#\w+", "<HashtagMention>", text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def process_data():
    with open("./data/session_10_results.json", "r", encoding="utf-8") as file:
        dataset1 = json.load(file)
    # with open("./data/session_3_results.json", "r", encoding="utf-8") as file:
    #     dataset2 = json.load(file)
    # with open("./data/session_4_results.json", "r", encoding="utf-8") as file:
    #     dataset3 = json.load(file)

    users_df = pd.DataFrame(dataset1["users"] \
                            # + dataset2["users"] + dataset3["users"]
                            )
    posts_df = pd.DataFrame(dataset1["posts"] \
                            #  + dataset2["posts"] + dataset3["posts"]
                             )

    posts_df = posts_df[['author_id', 'text', 'created_at']]
    users_df = users_df[['user_id', 'is_bot']]

    posts_df["cleaned_text"] = posts_df["text"].apply(preprocess_text)
    user_tweets = posts_df.groupby("author_id")["cleaned_text"].apply(lambda x: " ".join(x)).reset_index()
    users_df = users_df.merge(user_tweets, left_on="user_id", right_on="author_id", how="left")
    users_df.drop(columns=["author_id"], inplace=True)

    return users_df


def train_and_test(users_df):
    X = users_df[['cleaned_text']]
    y = users_df['is_bot']

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X['cleaned_text'], y, test_size=0.2, 
        random_state=42, stratify=y, shuffle=True
    )

    X_train_text = X_train_text.fillna("").reset_index(drop=True)
    X_test_text = X_test_text.fillna("").reset_index(drop=True)
    # print(X_train_text.reset_index(drop=True)[4])
    X_train_embeddings= model.encode(X_train_text)
    X_test_embeddings = model.encode(X_test_text)

    # LDA Topic Modeling
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X_train_counts = count_vectorizer.fit_transform(X_train_text.fillna(""))
    X_test_counts = count_vectorizer.transform(X_test_text.fillna(""))

    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    X_train_lda = lda.fit_transform(X_train_counts)
    X_test_lda = lda.transform(X_test_counts)

    # Combine TF-IDF and LDA features
    X_train_combined = np.hstack((X_train_embeddings, X_train_lda))
    X_test_combined = np.hstack((X_test_embeddings, X_test_lda))



    # Train Random Forest Classifier
    clf  = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_combined, y_train)

    y_pred = clf.predict(X_test_combined)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:\n', report)

    # save_models_and_vectorizers(clf, tfidf_vectorizer, lda, count_vectorizer)


def save_models_and_vectorizers(clf, tfidf_vectorizer, lda, count_vectorizer):
    # Save the trained model and vectorizers
    model_path = os.path.join(model_dir, "random_forest_with_lda.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(clf, model_file)

    tfidf_vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    with open(tfidf_vectorizer_path, "wb") as vec_file:
        pickle.dump(tfidf_vectorizer, vec_file)

    lda_vectorizer_path = os.path.join(model_dir, "lda_vectorizer.pkl")
    with open(lda_vectorizer_path, "wb") as lda_file:
        pickle.dump(lda, lda_file)

    count_vectorizer_path = os.path.join(model_dir, "count_vectorizer.pkl")
    with open(count_vectorizer_path, "wb") as count_file:
        pickle.dump(count_vectorizer, count_file)


def main():
    users_df = process_data()
    train_and_test(users_df)

if __name__ == "__main__":
    main()
