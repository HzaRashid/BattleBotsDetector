import re
import ijson
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sentence_transformers import SentenceTransformer, models
from huggingface_hub import hf_hub_download

# ------------* custom model and utils *------------
from .hybridv3 import MOEAttention   # updated: use MOEAttention for this version
from .time_utils import generate_time_dna, encode_time_dna_batch_cnn
from .content_utils import generate_content_dna, encode_content_dna_batch_cnn  # updated for consistency with training
from .tweet_embs import sample_evenly_tweets, compute_mean_embeddings  # new tweet embedding utilities

# -------------------------
# Set Seed for Reproducibility
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 32  # batch size is only used for encoding individual user texts

# -------------------------
# Load Description Embedding Model (SentenceTransformer)
# -------------------------
transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

# -------------------------
# Detector Class Using the Trained PyTorch Model
# -------------------------
from abc_classes import ADetector
from teams_classes import DetectionMark
<<<<<<< HEAD
import pandas as pd
import spacy
import pickle
import re
import os
import warnings

class Detector(ADetector):
    def __init__(self):
        warnings.filterwarnings("ignore", message=r"\[W108\].*lemmatizer.*")
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
=======

class Detector(ADetector):
    def __init__(self):
        # Use the same description embedding model.
        self.st_model = st_model

        # Load the trained PyTorch model from Hugging Face.
        model_path = hf_hub_download(repo_id="hzarashid/ForensiX", filename="hybridv3_weights.bin")
        self.model = MOEAttention()  # updated: use MOEAttention as in training
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        
    def detect_bot(self, session_data):
        """
        For each user in the session, compute the feature vectors from four modalities:
         - Description embedding (768-dim)
         - Mean tweet embedding (via compute_mean_embeddings)
         - Content DNA embedding (384-dim, via CNN)
         - Time DNA embedding (384-dim)
        Then, run these features through the trained model to predict the bot probability.
        Processes one user at a time to reduce memory usage.
        Returns a list of DetectionMark objects.
        """
        # Group posts by user_id.
        user_posts = {}
        for post in session_data.posts:
            uid = post.get("user_id") or post.get("author_id")
            if uid is not None:
                user_posts.setdefault(uid, []).append(post)
                
        marked_accounts = []
        
        # Process each user individually to avoid memory issues
        for user in session_data.users:
            uid = user.get("id")
            if uid is None or uid not in user_posts:
                continue

            # Description embedding.
            description = user.get("description", "")
            # Encode description (single sample encoding: returns a list, extract first element)
            desc_emb = self.st_model.encode([description], batch_size=1)[0]
            
            posts = user_posts[uid]
            # Sort posts by creation time.
            posts_sorted = sorted(
                posts,
                key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00"))
            )
            texts = [post.get("text", "") for post in posts_sorted]
            # Sample tweets evenly (e.g., 5 tweets per user).
            sampled_tweets = sample_evenly_tweets(texts, n=5)
            # Compute mean tweet embedding (again, returns a list)
            tweet_emb = compute_mean_embeddings([sampled_tweets], self.st_model, batch_size=1)[0]
            
            # Generate DNA sequences for content and time modalities.
            content_dna = generate_content_dna(posts)
            time_dna = generate_time_dna(posts)
            # Encode DNA sequences (each returns a list with one element)
            dna_emb = encode_content_dna_batch_cnn([content_dna])[0]
            time_emb = encode_time_dna_batch_cnn([time_dna])[0]
            
            # Convert each modality to a tensor and add a batch dimension.
            desc_tensor = torch.tensor(desc_emb, dtype=torch.float32).unsqueeze(0)
            tweet_tensor = torch.tensor(tweet_emb, dtype=torch.float32).unsqueeze(0)
            dna_tensor = torch.tensor(dna_emb, dtype=torch.float32).unsqueeze(0)
            time_tensor = torch.tensor(time_emb, dtype=torch.float32).unsqueeze(0)
            
            # Run through the trained model for the current user.
            with torch.no_grad():
                logits, _ = self.model(desc_tensor, tweet_tensor, dna_tensor, time_tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = probabilities.argmax(dim=1).item()
                confidence = int(probabilities[0, 1].item() * 100)
                is_bot = (pred_class == 1)  # assuming class '1' represents bots
            
            marked_accounts.append(DetectionMark(user_id=uid, confidence=confidence, bot=is_bot))
        
        print(marked_accounts)
        return marked_accounts
>>>>>>> 1cafa0c6
