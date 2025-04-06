import re
import string
import numpy as np
from datetime import datetime
from scipy.stats import entropy

# -------------------------------
# Existing Feature Functions
# -------------------------------

def extract_text_features(tweets):
    tweet_lengths = []
    hashtag_counts = []
    url_counts = []
    mention_counts = []
    all_words = []
    texts = []

    for tweet in tweets:
        text = tweet.get('text', "")
        texts.append(text)
        tweet_length = len(text)
        tweet_lengths.append(tweet_length)
        
        hashtags = re.findall(r"#\w+", text)
        hashtag_counts.append(len(hashtags))
        
        urls = re.findall(r"http[s]?://\S+", text)
        url_counts.append(len(urls))
        
        mentions = re.findall(r"@\w+", text)
        mention_counts.append(len(mentions))
        
        words = text.split()
        all_words.extend(words)
    
    avg_length = np.mean(tweet_lengths) if tweet_lengths else 0
    avg_hashtags = np.mean(hashtag_counts) if hashtag_counts else 0
    avg_urls = np.mean(url_counts) if url_counts else 0
    avg_mentions = np.mean(mention_counts) if mention_counts else 0
    
    hashtag_ratio = avg_hashtags / avg_length if avg_length > 0 else 0
    url_ratio = avg_urls / avg_length if avg_length > 0 else 0
    mention_ratio = avg_mentions / avg_length if avg_length > 0 else 0

    lexical_diversity = len(set(all_words)) / len(all_words) if all_words else 0
    duplicate_ratio = 1 - (len(set(texts)) / len(texts)) if texts else 0

    return {
        'avg_tweet_length': avg_length,
        'avg_hashtags': avg_hashtags,
        'hashtag_ratio': hashtag_ratio,
        'avg_urls': avg_urls,
        'url_ratio': url_ratio,
        'avg_mentions': avg_mentions,
        'mention_ratio': mention_ratio,
        'lexical_diversity': lexical_diversity,
        'duplicate_ratio': duplicate_ratio,
    }

def extract_time_features(tweets):
    times = []
    for tweet in tweets:
        created_at = tweet.get('created_at')
        if created_at:
            try:
                # Replace trailing "Z" with "+00:00" for ISO compliance.
                if created_at.endswith("Z"):
                    created_at = created_at[:-1] + "+00:00"
                dt = datetime.fromisoformat(created_at)
                times.append(dt)
            except ValueError:
                continue

    if not times:
        return {
            'tweets_per_hour': 0,
            'time_entropy': 0,
            'inter_tweet_interval_std': 0
        }

    times.sort()
    time_span_hours = (times[-1] - times[0]).total_seconds() / 3600.0
    tweets_per_hour = len(times) / time_span_hours if time_span_hours > 0 else len(times)
    hours = [t.hour for t in times]
    counts = np.bincount(hours, minlength=24)
    probs = counts / counts.sum()
    time_entropy = entropy(probs)
    intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    inter_tweet_interval_std = np.std(intervals) if intervals else 0

    return {
        'tweets_per_hour': tweets_per_hour,
        'time_entropy': time_entropy,
        'inter_tweet_interval_std': inter_tweet_interval_std,
    }

def extract_emoji_and_punctuation_features(tweets):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    
    punct_ratios = []
    emoji_ratios = []
    
    for tweet in tweets:
        text = tweet.get("text", "")
        if not text:
            continue
        tweet_len = len(text)
        punct_count = sum(1 for char in text if char in string.punctuation)
        emojis = emoji_pattern.findall(text)
        emoji_count = len(emojis)
        
        punct_ratios.append(punct_count / tweet_len if tweet_len > 0 else 0)
        emoji_ratios.append(emoji_count / tweet_len if tweet_len > 0 else 0)
    
    avg_punctuation_ratio = np.mean(punct_ratios) if punct_ratios else 0
    avg_emoji_ratio = np.mean(emoji_ratios) if emoji_ratios else 0
    
    return {
        'avg_punctuation_ratio': avg_punctuation_ratio,
        'avg_emoji_ratio': avg_emoji_ratio,
    }

def extract_engagement_features(tweets):
    retweet_count = 0
    reply_count = 0
    total_tweets = 0
    
    for tweet in tweets:
        text = tweet.get("text", "")
        if not text:
            continue
        total_tweets += 1
        if text.startswith("RT"):
            retweet_count += 1
        elif text.startswith("@"):
            reply_count += 1
    
    original_count = total_tweets - retweet_count - reply_count
    retweet_ratio = retweet_count / total_tweets if total_tweets > 0 else 0
    reply_ratio = reply_count / total_tweets if total_tweets > 0 else 0
    original_ratio = original_count / total_tweets if total_tweets > 0 else 0
    
    return {
        'retweet_ratio': retweet_ratio,
        'reply_ratio': reply_ratio,
        'original_tweet_ratio': original_ratio,
    }

# -------------------------------
# Additional Linguistic Features
# -------------------------------

# 1. Readability Features (Flesch-Kincaid Grade Level)
def count_syllables(word):
    word = word.lower()
    syllable_count = 0
    vowels = "aeiouy"
    if word and word[0] in vowels:
        syllable_count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            syllable_count += 1
    # Adjust for silent 'e'
    if word.endswith("e"):
        syllable_count -= 1
    if syllable_count <= 0:
        syllable_count = 1
    return syllable_count

def compute_fk_grade(text):
    # Split text into sentences using punctuation.
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    total_sentences = len(sentences)
    words = text.split()
    total_words = len(words)
    total_syllables = sum(count_syllables(word) for word in words)
    if total_sentences == 0 or total_words == 0:
        return 0
    # Flesch-Kincaid Grade Level formula.
    fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return fk_grade

def extract_readability_features(tweets):
    fk_grades = []
    for tweet in tweets:
        text = tweet.get("text", "")
        if not text:
            continue
        fk = compute_fk_grade(text)
        fk_grades.append(fk)
    avg_fk_grade = np.mean(fk_grades) if fk_grades else 0
    return {'avg_fk_grade': avg_fk_grade}

# 2. N-gram Diversity Features
def extract_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def extract_ngram_features(tweets):
    all_tokens = []
    for tweet in tweets:
        text = tweet.get("text", "")
        if not text:
            continue
        tokens = text.split()
        all_tokens.extend(tokens)
    
    bigrams = extract_ngrams(all_tokens, 2)
    trigrams = extract_ngrams(all_tokens, 3)
    bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
    trigram_diversity = len(set(trigrams)) / len(trigrams) if trigrams else 0
    
    return {
        'bigram_diversity': bigram_diversity,
        'trigram_diversity': trigram_diversity,
    }

# -------------------------------
# Aggregating All Features
# -------------------------------
def extract_all_features(tweets):
    features = {}
    # Basic tweet and activity features.
    for fn in (
        extract_text_features,
        extract_time_features,
        extract_emoji_and_punctuation_features,
        extract_engagement_features,
    ):
        features.update(fn(tweets))
    # Additional linguistic features.
    for fn in (
        extract_readability_features,
        extract_ngram_features,
    ):
        features.update(fn(tweets))
    return features

def features_to_vector(features, feature_order=None):
    if feature_order is None:
        feature_order = sorted(features.keys())
    vector = np.array([features[key] for key in feature_order], dtype=np.float32)
    return vector


def naive_user_features(tweets):
    all_features = extract_all_features(tweets)
    feature_vec = features_to_vector(all_features)
    return feature_vec

# -------------------------------
# Sample Execution
# -------------------------------
if __name__ == "__main__":
    tweets = [
        {"text": "RT Check out our website! http://example.com #promo 😊", "created_at": "2025-03-31T12:34:56.000Z"},
        {"text": "@user Good morning! How are you today? 😊", "created_at": "2025-03-31T13:00:00.000Z"},
        {"text": "Just another day in paradise! #life", "created_at": "2025-03-31T13:30:00.000Z"},
        {"text": "Wow!!! Amazing performance!!!", "created_at": "2025-03-31T14:00:00.000Z"}
    ]
    
    all_features = extract_all_features(tweets)
    feature_vec = naive_user_features(tweets)
    
    print("Extracted Features:")
    for key, value in all_features.items():
        print(f"{key}: {value:.4f}")

    print("\nFeature Vector:")
    print(feature_vec)
