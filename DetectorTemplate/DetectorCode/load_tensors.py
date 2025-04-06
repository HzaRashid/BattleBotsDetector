import os
import ijson
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from time_utils import generate_time_dna, encode_time_dna_batch_cnn
from content_utils import generate_content_dna, encode_content_dna_batch_cnn
from tweet_embs import sample_evenly_tweets, compute_mean_embeddings


BATCH_SIZE=64

def load_data(data_dir, session_numbers=[], st_model=None, xnums=[]):
    """
    Loads user and post data from JSON files in a memory-efficient way using ijson.kvitems,
    extracts features with batch processing for text embeddings, and computes a mean pooled
    embedding of each user's tweets using evenly distributed temporal stratification.
    
    Returns modalities for train and test sets in the following order:
        description embeddings, mean tweet embeddings, DNA embeddings, time embeddings, labels.
    """
    user_info_list = []
    user_posts_dict = {}
    datasets = []

    if session_numbers:
        datasets += [f"session_{num}_results.json" for num in session_numbers]
    if xnums:
        datasets += [f"twibot22/processed/tweet_{num}_processed.json" for num in xnums]

    # Stream the JSON file once to extract both users and posts.
    for fname in datasets:
        json_file = os.path.join(data_dir, fname)
        with open(json_file, "r") as f:
            # Iterate over top-level keys ("users" and "posts")
            for key, items in ijson.kvitems(f, ""):
                if key == "users":
                    for user in items:
                        user_info_list.append(user)
                elif key == "posts":
                    for post in items:
                        uid = post.get("user_id") or post.get("author_id")
                        if uid is not None:
                            user_posts_dict.setdefault(uid, []).append(post)

    # Stratified train/test split using user metadata.
    user_info_df = pd.DataFrame(user_info_list)[['user_id', 'is_bot']].drop_duplicates()
    train_users, test_users = train_test_split(
        user_info_df, test_size=0.2, random_state=42, stratify=user_info_df['is_bot']
    )
    train_user_ids = set(train_users['user_id'])
    test_user_ids = set(test_users['user_id'])

    # Prepare lists for different modalities.
    train_desc_texts, test_desc_texts = [], []
    train_dna_list, test_dna_list = [], []
    train_time_list, test_time_list = [], []
    train_labels, test_labels = [], []
    # Each entry in these lists will be a list of the sampled tweets.
    train_tweets, test_tweets = [], []

    for user in user_info_list:
        uid = user.get("user_id")
        # Skip users with no associated posts.
        if uid is None or uid not in user_posts_dict:
            continue
        description = user.get("description", "")
        # Sample tweets evenly from the user's tweet texts.
        time_sorted_posts = sorted(user_posts_dict[uid], key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
        user_texts = [tweetitem['text'] for tweetitem in time_sorted_posts]
                      
        sampled_tweets = sample_evenly_tweets(user_texts, n=5)
        # Generate DNA sequences for content and time modalities.
        content_dna = generate_content_dna(user_posts_dict[uid])
        time_dna = generate_time_dna(user_posts_dict[uid])
        label = int(user.get("is_bot", 0))

        if uid in train_user_ids:
            train_desc_texts.append(description)
            train_dna_list.append(content_dna)
            train_time_list.append(time_dna)
            train_tweets.append(sampled_tweets)
            train_labels.append(label)
        elif uid in test_user_ids:
            test_desc_texts.append(description)
            test_dna_list.append(content_dna)
            test_time_list.append(time_dna)
            test_tweets.append(sampled_tweets)
            test_labels.append(label)

    # Batch encode user descriptions.
    train_desc_embs = st_model.encode(train_desc_texts, batch_size=BATCH_SIZE)
    test_desc_embs = st_model.encode(test_desc_texts, batch_size=BATCH_SIZE)

    # Compute mean tweet embeddings for train set.
    train_tweet_embs = compute_mean_embeddings(train_tweets, st_model, batch_size=BATCH_SIZE)
    test_tweet_embs = compute_mean_embeddings(test_tweets, st_model, batch_size=BATCH_SIZE)

    # Generate CNN embeddings for DNA sequences.
    train_dna_embs = encode_content_dna_batch_cnn(train_dna_list)
    train_time_embs = encode_time_dna_batch_cnn(train_time_list)
    test_dna_embs = encode_content_dna_batch_cnn(test_dna_list)
    test_time_embs = encode_time_dna_batch_cnn(test_time_list)

    # Return data as separate modalities for train and test sets.
    # Order: description, mean tweet, DNA, time, labels.
    train = [np.array(x) for x in [train_desc_embs, 
                                   train_tweet_embs,
                                   train_dna_embs, 
                                   train_time_embs, 
                                   train_labels]]
    test = [np.array(x) for x in [test_desc_embs, 
                                  test_tweet_embs,
                                  test_dna_embs, 
                                  test_time_embs, 
                                  test_labels]]
    return (train, test)

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, models
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")

    # Initialize the transformer and pooling models.
    transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

    # Load data using the updated function.
    (train, test) = load_data(
         data_dir=data_dir, session_numbers=[10], xnums=[], st_model=st_model
     )

    # Display the shapes of the extracted embeddings and labels.
    print("Train description embeddings shape:", train[0].shape)
    print("Train mean tweet embeddings shape:", train[1].shape)
    print("Train DNA embeddings shape:", train[2].shape)
    print("Train time embeddings shape:", train[3].shape)
    print("Train labels shape:", train[4].shape)

    print("Test description embeddings shape:", test[0].shape)
    print("Test mean tweet embeddings shape:", test[1].shape)
    print("Test DNA embeddings shape:", test[2].shape)
    print("Test time embeddings shape:", test[3].shape)
    print("Test labels shape:", test[4].shape)


    print(train[1])
