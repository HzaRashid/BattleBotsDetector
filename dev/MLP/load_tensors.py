import os
import ijson
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from content_utils import generate_content_dna, encode_content_dna_batch_cnn
from time_utils import generate_time_dna, encode_time_dna_batch_cnn
random.seed(42)
# -------------------------
# Data Loading Function (returns separate modalities)
# -------------------------
def load_data(data_dir, session_numbers=[], st_model=None, xnums=[]):
    user_info_list = []
    user_posts_dict = {}
    datasets = []
    
    if session_numbers:
        datasets += [f"session_{num}_results.json" for num in session_numbers]
    if xnums:
        datasets += [f"twibot22/processed/tweet_{num}_processed.json" for num in xnums]

    for fname in datasets:
        json_file = os.path.join(data_dir, fname)
        with open(json_file, "r") as f:
            for user in ijson.items(f, "users.item"):
                user_info_list.append(user)
        with open(json_file, "r") as f:
            for post in ijson.items(f, "posts.item"):
                uid = post.get("user_id") or post.get("author_id")
                if uid is not None:
                    user_posts_dict.setdefault(uid, []).append(post)

    # stratified train/test split
    user_info_df = pd.DataFrame(user_info_list)[['user_id', 'is_bot']].drop_duplicates()
    train_users, test_users = train_test_split(
        user_info_df, test_size=0.2, random_state=42, stratify=user_info_df['is_bot']
    )
    train_user_ids = set(train_users['user_id'])
    test_user_ids = set(test_users['user_id'])

    train_texts, test_texts = [], []
    train_dna_list, test_dna_list = [], []
    train_time_list, test_time_list = [], []
    train_labels, test_labels = [], []

    for user in user_info_list:
        uid = user.get("user_id")
        if uid is None or uid not in user_posts_dict:
            continue
        description = user.get("description", "")
        # Get text embedding.
        desc_emb = st_model.encode(description)  # 768-dim
        # tweet_embs = generate_text_embs(tweets=user_posts_dict[uid], st_model=st_model)
        # generate the DNA sequences
        content_dna = generate_content_dna(user_posts_dict[uid])
        time_dna = generate_time_dna(user_posts_dict[uid])

       
        # user's true label
        label = int(user.get("is_bot", 0))

        if uid in train_user_ids:
            train_texts.append(desc_emb)
            train_dna_list.append(content_dna)
            train_time_list.append(time_dna)
            train_labels.append(label)
        elif uid in test_user_ids:
            test_texts.append(desc_emb)
            test_dna_list.append(content_dna)
            test_time_list.append(time_dna)
            test_labels.append(label)
    
    # generate the CNN embeddings of the DNA sequences
    train_dna_embs = encode_content_dna_batch_cnn(train_dna_list)
    train_time_embs = encode_time_dna_batch_cnn(train_time_list)

    test_dna_embs = encode_content_dna_batch_cnn(test_dna_list)
    test_time_embs = encode_time_dna_batch_cnn(test_time_list)

    # return split with seperable modalities
    train = (np.array(x) 
             for x in [
                 train_texts,
                 train_dna_embs,
                 train_time_embs,
                 train_labels
                 ])
    test = (np.array(x) 
            for x in [
                test_texts,
                test_dna_embs,
                test_time_embs,
                test_labels
                ])

    return (*train, *test)


def generate_text_embs(tweets, st_model):
    tweets_sample = random.sample(population=tweets, k=min(len(tweets), 10))
    return st_model.encode([tweet['text'] for tweet in tweets_sample])


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, models 
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])
    load_data(data_dir=data_dir, session_numbers=[], xnums=[0], st_model=st_model)

    # ----- optional class weights if using BCE loss ------
    # from sklearn.utils.class_weight import compute_class_weight
    # y_train_np = y_train_tensor.numpy()
    # classes = np.unique(y_train_np)
    # class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_np)
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # --------------------------
    # -------------------------
    # Helper function to downsample negatives in a given set of indices.
    # The target ratio is positive:negative = 1:10.
    # -------------------------
    # def downsample_indices(indices, labels, target_ratio=10):
    #     indices = np.array(indices)
    #     pos_idx = indices[labels[indices] == 1]
    #     neg_idx = indices[labels[indices] == 0]
    #     desired_neg_count = min(len(neg_idx), target_ratio * len(pos_idx))
    #     neg_idx_downsampled = np.random.choice(neg_idx, size=desired_neg_count, replace=False)
    #     combined = np.concatenate([pos_idx, neg_idx_downsampled])
    #     np.random.shuffle(combined)
    #     return combined
    # ----- optional downsampling for entire training set -----
    # train_idx_downsampled = downsample_indices(train_idx, y_train_tensor.numpy(), target_ratio=8)
    # ---------------------------------

    # ----- optional downsampling for optuna -----
    # train_idx_downsampled = downsample_indices(train_idx, y_train_tensor.numpy(), target_ratio=8)
    # rejected_idx = np.setdiff1d(train_idx, train_idx_downsampled)
    # val_idx_updated = np.concatenate([val_idx, rejected_idx])
    # -------------------------------