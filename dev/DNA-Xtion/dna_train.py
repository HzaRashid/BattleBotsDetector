import os
import re
import json
import math
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

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
    # Sort tweets chronologically
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    # Concatenate the DNA symbols in order
    dna = "".join(get_content_dna_symbol(tweet["text"]) for tweet in tweets)
    return dna

# -------------------------
# N-gram Feature Extraction from DNA
# -------------------------
def extract_ngram_features(dna, n=2, symbols=["N", "U", "H", "M", "X", "R", "E"]):
    """
    Extracts normalized n-gram frequency features from the DNA sequence.
    For example, with bigrams (n=2), there are len(symbols)^2 possible n-grams.
    """
    # Build dictionary for all possible n-grams
    from itertools import product
    possible_ngrams = [''.join(p) for p in product(symbols, repeat=n)]
    ngram_counts = {ngram: 0 for ngram in possible_ngrams}
    
    # Count n-grams in the DNA string (preserving order)
    total = 0
    for i in range(len(dna) - n + 1):
        ngram = dna[i:i+n]
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
            total += 1
    
    # Normalize counts: if no n-gram found, return zeros.
    if total == 0:
        return np.zeros(len(possible_ngrams))
    
    features = np.array([ngram_counts[ng] / total for ng in possible_ngrams])
    return features

# -------------------------
# DNA Mutation Function (if needed for SMOTE minority augmentation)
# -------------------------
def mutate_dna(dna, mutation_rate=0.1, deletion_rate=0.05, splicing_rate=0.2):
    if not dna: 
        return dna
    rand_state = np.random.RandomState(seed=42)
    symbols = ["N", "U", "H", "M", "X", "R", "E"]
    mutated_chars = []
    for char in dna:
        if rand_state.rand() < deletion_rate:
            continue  # deletion
        if rand_state.rand() < mutation_rate:
            possible = [s for s in symbols if s != char]
            new_char = rand_state.choice(possible)
            mutated_chars.append(new_char)
        else:
            mutated_chars.append(char)
    mutated_dna = "".join(mutated_chars)
    if len(mutated_dna) > 3 and np.random.rand() < splicing_rate:
        mutated_dna = mutated_dna[3:] + mutated_dna[:3]
    return mutated_dna

# -------------------------
# Data Loading Function
# -------------------------
def load_data(data_dir, session_numbers, st_model):
    """
    Loads user description and posts from JSON files.
    Returns:
        emb_list: Array of description embeddings.
        dna_list: List of digital DNA strings (ordered chronologically).
        labels_list: Array of labels.
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

    emb_list = []
    dna_list = []
    labels_list = []

    for uid in user_info:
        if uid not in user_posts:
            continue
        description = user_info[uid]["description"]
        desc_emb = st_model.encode(description)
        dna = generate_content_dna(user_posts[uid])
        emb_list.append(desc_emb)
        dna_list.append(dna)
        labels_list.append(int(user_info[uid]["is_bot"]))
    return np.array(emb_list), dna_list, np.array(labels_list)

# -------------------------
# SMOTE and Mutation Integration on Embeddings
# -------------------------
def oversample_embeddings(embeddings, dna_list, labels, n=2):
    """
    Normalizes embeddings, applies SMOTE on the normalized embeddings,
    and for each synthetic minority sample, generates mutated DNA and
    extracts n-gram features.
    Returns:
        new_embeddings: Array of embeddings (back in original scale)
        new_dna_features: Array of n-gram DNA features for all samples
        new_labels: Array of labels after oversampling.
    """
    scaler = StandardScaler()
    embeddings_norm = scaler.fit_transform(embeddings)
    
    # Apply SMOTE on normalized embeddings
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(embeddings_norm, labels)
    
    new_embeddings = scaler.inverse_transform(X_res)
    
    # Prepare pools for DNA strings based on class
    minority_dna = [dna_list[i] for i, lab in enumerate(labels) if lab == 1]
    majority_dna = [dna_list[i] for i, lab in enumerate(labels) if lab == 0]
    
    new_dna_features = []
    for lab in y_res:
        if lab == 1 and minority_dna:
            orig_dna = np.random.choice(minority_dna)
            mutated_dna = mutate_dna(orig_dna)
            ngram_feat = extract_ngram_features(mutated_dna, n=n)
        elif lab == 0 and majority_dna:
            orig_dna = np.random.choice(majority_dna)
            ngram_feat = extract_ngram_features(orig_dna, n=n)
        else:
            ngram_feat = np.zeros(len(["".join(p) for p in __import__('itertools').product(["N", "U", "H", "M", "X", "R", "E"], repeat=n)]))
        new_dna_features.append(ngram_feat)
    new_dna_features = np.array(new_dna_features)
    
    return new_embeddings, new_dna_features, y_res

# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    session_numbers = [12, 13]  # Adjust as needed

    # Initialize SentenceTransformer model
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load data: description embeddings, raw DNA strings (in order), and labels.
    embeddings, dna_list, labels = load_data(data_dir, session_numbers, st_model)

    # Split into training and test sets
    emb_train, emb_test, dna_train, dna_test, y_train, y_test = train_test_split(
        embeddings, dna_list, labels, test_size=0.2, random_state=42)

    # Apply SMOTE (with normalization) on embeddings and generate mutated n-gram features for training.
    train_emb, train_dna_features, y_train_res = oversample_embeddings(emb_train, dna_train, y_train, n=2)
    
    # For test set, simply extract bigram features from the original DNA sequences.
    test_dna_features = np.array([extract_ngram_features(dna, n=2) for dna in dna_test])

    # Combine embeddings and DNA features
    X_train = np.concatenate([train_emb, train_dna_features], axis=1)
    X_test = np.concatenate([emb_test, test_dna_features], axis=1)

    # Optionally compute scale_pos_weight for XGBoost
    neg, pos = np.bincount(y_train_res)
    scale_pos_weight = neg / pos

    # Train XGBoost classifier
    xgb_clf = XGBClassifier(n_estimators=100,
                            learning_rate=0.01,
                            random_state=42,
                            scale_pos_weight=scale_pos_weight,
                            use_label_encoder=False,
                            eval_metric='logloss')
    xgb_clf.fit(X_train, y_train_res)

    # Evaluate the model
    y_pred = xgb_clf.predict(X_test)
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))
