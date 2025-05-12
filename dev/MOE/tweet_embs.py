import numpy as np

def sample_evenly_tweets(tweets, n=8):
    """
    Samples n tweets evenly distributed across the timeline.
    
    Args:
        tweets (list of str): List of tweet texts.
        n (int): Number of tweets to sample.
    
    Returns:
        list of str: Sampled tweets.
    """
    if not tweets:
        return []
    if len(tweets) <= n:
        return tweets
    indices = np.linspace(0, len(tweets) - 1, num=n)
    indices = [int(round(idx)) for idx in indices]
    return [tweets[i] for i in indices]

def compute_mean_embeddings(tweet_groups, st_model, batch_size=32):
    """
    Compute mean embeddings for a list of tweet groups in a vectorized manner.
    
    Parameters:
        tweet_groups (List[List[str]]): List where each element is a list of tweet texts for a user.
        st_model: The sentence transformer model with an encode method supporting batch processing.
        batch_size (int): Batch size for encoding.
        
    Returns:
        np.ndarray: Array of mean embeddings (one per tweet group).
    """
    # Flatten tweet groups into one list.
    all_tweets = [tweet for group in tweet_groups for tweet in group]
    
    # Batch encode all tweet texts.
    all_embeddings = st_model.encode(all_tweets, batch_size=batch_size)
    all_embeddings = np.array(all_embeddings)
    
    mean_embeddings = []
    start = 0
    # For each tweet group, compute the mean embedding.
    for group in tweet_groups:
        group_size = len(group)
        group_embeddings = all_embeddings[start:start+group_size]
        mean_embeddings.append(group_embeddings.mean(axis=0))
        start += group_size
        
    return np.array(mean_embeddings)



# Example usage:
if __name__ == "__main__":
    # Sample tweets for a user (assumed to be ordered chronologically).
    tweet_groups = [
        ["Started my day with a morning run and coffee",
        "Loving the new features in our app update"],
        ["Exploring some machine learning algorithms today",
        "Had a productive meeting with the team",
        "Learning about temporal stratification in data sampling"],
        ["Wrapping up the day with a bit of reading",
        "Reflecting on the day's achievements and challenges"]
    ]
    
    from sentence_transformers import SentenceTransformer, models
    
    # Initialize the transformer and pooling models.
    transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])
    
    # Get the meta tweet embedding using evenly distributed sampling.
    meta_emb = compute_mean_embeddings(tweet_groups, st_model, 32)
    if meta_emb is not None:
        print("Meta tweet embedding (mean of evenly sampled tweets):", meta_emb)
    else:
        print("Could not compute meta tweet embedding due to empty input.")
