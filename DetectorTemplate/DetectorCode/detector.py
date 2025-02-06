from abc_classes import ADetector
from teams_classes import DetectionMark
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import spacy
import pickle
import re
import os

class Detector(ADetector):
    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm", enable=['tokenizer', 'stopwords','lemmatizer'])
        
        model_dir = f'{os.path.dirname(__file__)}/models'
        with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as model_file:
            self.clf = pickle.load(model_file)

        with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "rb") as vec_file:
            self.vectorizer = pickle.load(vec_file)
            

    def detect_bot(self, session_data):
        feature_vectors = self.process_data(session_data)
        prediction_probs = self.clf.predict_proba(feature_vectors)
        # print(prediction_probs)
        
        # todo logic    
        # Example:
        marked_account = []
        
        for i, user in enumerate(session_data.users):
            bot_confidence = int(prediction_probs[i][1] * 100)  # Highest probability
            predicted_class = bool(prediction_probs[i].argmax())   # Class with the highest probability
            # print(bot_confidence, predicted_class)
            marked_account.append(DetectionMark(user_id=user['id'], confidence=bot_confidence, bot=predicted_class))

        return marked_account
    

    def process_data(self, session_data):
        """
        session_data = {
        session_id: int,
        lang: str,
        metadata: None,
        users: ...
        posts: ...
        }
        """
        users_df = pd.DataFrame(session_data.users)
        posts_df = pd.DataFrame(session_data.posts)

        # Keep only relevant columns
        posts_df = posts_df[['author_id', 'text', 'created_at']]
        users_df = users_df[['id']]

        posts_df["cleaned_text"] = posts_df["text"].apply(self.preprocess_text)

        # Combine all tweets for each user into a single document
        user_tweets = posts_df.groupby("author_id")["cleaned_text"].apply(lambda x: " ".join(x)).reset_index()

        # Merge processed tweets back to users
        users_df = users_df.merge(user_tweets, left_on="id", right_on="author_id", how="left")
        
        # Apply TF-IDF to get user-specific word importance
        tfidf_matrix = self.vectorizer.transform(users_df["cleaned_text"].fillna(""))

        # Convert TF-IDF matrix to a DataFrame
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        return tfidf_df


    def preprocess_text(self, text):
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
        doc = self.nlp(text)

        # Extract lemmatized tokens, removing stopwords and punctuation, and converting to lowercase
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        
        return " ".join(tokens)