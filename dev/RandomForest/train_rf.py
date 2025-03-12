import os
import re
import json
import math
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# -------------------------
# Digital DNA Functions (same as before)
# -------------------------
emoji_pattern = re.compile("[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U0001F700-\U0001F77F"  # alchemical symbols
    u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA00-\U0001FA6F"  # Chess Symbols, etc.
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+", flags=re.UNICODE)

def get_content_dna_symbol(tweet_text):
    url_present = bool(re.search(r"https?://\S+", tweet_text))
    hashtag_present = bool(re.search(r"#\w+", tweet_text))
    mention_present = bool(re.search(r"@\w+", tweet_text))
    
    if tweet_text.strip().startswith("@"):
        return "R"
    
    emoji_present = bool(emoji_pattern.search(tweet_text))
    entity_types = sum([url_present, hashtag_present, mention_present, emoji_present])
    
    if entity_types == 0:
        return "N"
    elif entity_types == 1:
        if url_present:
            return "U"
        elif hashtag_present:
            return "H"
        elif mention_present:
            return "M"
        elif emoji_present:
            return "E"
    else:
        return "X"

def generate_content_dna(tweets):
    # Sort tweets by creation time
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = "".join(get_content_dna_symbol(tweet["text"]) for tweet in tweets)
    return dna

# -------------------------
# DNA Feature Extraction Function
# -------------------------
def extract_dna_features(dna, symbols=["N", "U", "H", "M", "X", "R", "E"]):
    """
    Returns a normalized count for each symbol in the DNA string.
    """
    total = len(dna)
    if total == 0:
        # Return zeros if no dna was generated
        return np.zeros(len(symbols))
    counts = [dna.count(s) for s in symbols]
    # Normalize by total length
    counts = [c/total for c in counts]
    return np.array(counts)

# -------------------------
# Data Loading Function
# -------------------------
def load_data(data_dir, session_numbers, st_model):
    """
    Loads users and posts from session JSON files.
    Returns lists of features and labels.
    """
    user_info = {}
    user_posts = {}

    for num in session_numbers:
        json_file = os.path.join(data_dir, f"session_{num}_results.json")
        with open(json_file, "r") as f:
            data = json.load(f)
        
        for user in data.get("users", []):
            uid = user["user_id"]
            if uid not in user_info:
                user_info[uid] = {
                    "description": user.get("description", ""),
                    "is_bot": user.get("is_bot", 0)
                }
        for post in data.get("posts", []):
            uid = post["author_id"]
            if uid not in user_posts:
                user_posts[uid] = []
            user_posts[uid].append(post)

    X = []
    y = []
    symbols = ["N", "U", "H", "M", "X", "R", "E"]

    # Process only users that have both user info and posts
    for uid in user_info:
        if uid not in user_posts:
            continue
        description = user_info[uid]["description"]
        # Get the description embedding
        desc_emb = st_model.encode(description)
        # Generate DNA from posts
        dna = generate_content_dna(user_posts[uid])
        dna_features = extract_dna_features(dna, symbols=symbols)
        # Concatenate description embedding and dna features
        features = np.concatenate([desc_emb, dna_features])
        X.append(features)
        y.append(int(user_info[uid]["is_bot"]))

    return np.array(X), np.array(y)

# -------------------------
# Main Pipeline for Random Forest Classifier
# -------------------------
if __name__ == "__main__":
    # Paths and session configuration
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    session_numbers = [12, 13]  # Adjust session numbers as needed
    
    # Initialize SentenceTransformer model
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load data and create features
    X, y = load_data(data_dir, session_numbers, st_model)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_clf.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Optionally, save the trained model using joblib
    # from joblib import dump
    # dump(rf_clf, os.path.join(cur_dir, "rf_classifier.joblib"))
