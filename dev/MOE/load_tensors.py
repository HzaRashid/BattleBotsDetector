import os
import ijson
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from time_utils import generate_time_dna, encode_time_dna_batch_cnn
from content_utils import generate_content_dna, encode_content_dna_batch_cnn
from tweet_embs import sample_evenly_tweets, compute_mean_embeddings
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 64

class BotSessionDataset(Dataset):
    def __init__(self, user_info_list, user_posts_dict, st_model, sample_n=5):
        # 1) filter out users without posts
        valid_users = [u for u in user_info_list if u['user_id'] in user_posts_dict]
        desc_texts, tweet_lists, content_dnas, time_dnas, labels = [], [], [], [], []
        for u in valid_users:
            uid = u['user_id']
            desc_texts.append(u.get('description', ''))
            # sort & sample
            posts = sorted(
                user_posts_dict[uid],
                key=lambda x: datetime.fromisoformat(x['created_at'].replace('Z', '+00:00'))
            )
            texts = [p['text'] for p in posts]
            tweet_lists.append(sample_evenly_tweets(texts, sample_n))
            content_dnas.append(generate_content_dna(posts))
            time_dnas.append(generate_time_dna(posts))
            labels.append(int(u.get('is_bot', 0)))
        # 3) precompute embeddings once
        self.desc_embs    = st_model.encode(desc_texts, batch_size=BATCH_SIZE)
        self.tweet_embs   = compute_mean_embeddings(tweet_lists, st_model, batch_size=BATCH_SIZE)
        self.content_embs = encode_content_dna_batch_cnn(content_dnas)
        self.time_embs    = encode_time_dna_batch_cnn(time_dnas)
        self.labels       = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.desc_embs[idx],    dtype=torch.float32),
            torch.tensor(self.tweet_embs[idx],   dtype=torch.float32),
            torch.tensor(self.content_embs[idx], dtype=torch.float32),
            torch.tensor(self.time_embs[idx],    dtype=torch.float32),
            torch.tensor(self.labels[idx],       dtype=torch.long),
        )


def load_user_and_post_json(data_dir, session_numbers=None, xnums=None):
    if session_numbers is None:
        session_numbers = []
    if xnums is None:
        xnums = []

    user_info_list = []
    user_posts_dict = {}
    filenames = [f"session_{n}_results.json" for n in session_numbers] + \
                [f"twibot22/processed/tweet_{n}_processed.json" for n in xnums]

    for fname in filenames:
        path = os.path.join(data_dir, fname)
        with open(path, "r") as f:
            for key, items in ijson.kvitems(f, ""):
                if key == "users":
                    user_info_list.extend(items)
                elif key == "posts":
                    for post in items:
                        uid = post.get("user_id") or post.get("author_id")
                        if uid is None:
                            continue
                        user_posts_dict.setdefault(uid, []).append(post)

    # Deduplicate user entries: ensure each user_id appears only once
    # unique = {}
    # for u in user_info_list:
    #     uid = u.get('user_id') or u.get('author_id')
    #     unique.setdefault(uid, u)
    # user_info_list = list(unique.values())

    # # Deduplicate posts for each user by post 'id' or 'post_id'
    # for uid, posts in user_posts_dict.items():
    #     seen = {}
    #     for p in posts:
    #         pid = p.get('id') or p.get('post_id')
    #         if pid is None:
    #             # if no id field, fallback to full object hash
    #             pid = hash(frozenset(p.items()))
    #         if pid not in seen:
    #             seen[pid] = p
    #     user_posts_dict[uid] = list(seen.values())

    return user_info_list, user_posts_dict


def build_datasets(data_dir, session_numbers=None, xnums=None, st_model=None,
                   train_ratio=0.75, val_ratio=0.05, test_ratio=0.20):
    """
    Splits the data into train/validation/test according to the provided ratios.
    """
    # Load and dedupe data
    user_info_list, user_posts_dict = load_user_and_post_json(data_dir, session_numbers, xnums)

    # Create DataFrame for stratification
    df = pd.DataFrame(user_info_list)[['user_id', 'is_bot']].drop_duplicates()

    # First split: train vs. temp (val+test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df['is_bot'],
        random_state=42
    )

    # Second split: validation vs. test
    # test_size relative to temp set = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / temp_ratio,
        stratify=temp_df['is_bot'],
        random_state=42
    )

    # Extract user_id sets
    train_uids = set(train_df['user_id'])
    val_uids   = set(val_df['user_id'])
    test_uids  = set(test_df['user_id'])

    # Partition user lists
    train_users = [u for u in user_info_list if u['user_id'] in train_uids]
    val_users   = [u for u in user_info_list if u['user_id'] in val_uids]
    test_users  = [u for u in user_info_list if u['user_id'] in test_uids]

    # Create datasets
    train_ds = BotSessionDataset(train_users, {u: user_posts_dict[u] for u in train_uids}, st_model)
    val_ds   = BotSessionDataset(val_users,   {u: user_posts_dict[u] for u in val_uids},   st_model)
    test_ds  = BotSessionDataset(test_users,  {u: user_posts_dict[u] for u in test_uids},  st_model)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, models
    data_dir = Path(__file__).parent / "../data"

    transformer_model = models.Transformer(
        "Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'}
    )
    pooling_model = models.Pooling(
        transformer_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

    train_ds, val_ds, test_ds = build_datasets(
        data_dir, session_numbers=[10], xnums=[], st_model=st_model
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
