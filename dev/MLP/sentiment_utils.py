from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
import torch.nn as nn

# Initialize the sentiment analysis pipeline.
model_path = "clapAI/modernBERT-base-multilingual-sentiment"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=-1, use_fast=True)
label_map = {
    'negative': "-",
    "neutral": "|",
    "positive": "+"
    }

def get_sentiment_symbol(tweet_text):
    """
    Uses the CardiffNLP sentiment model to classify tweet text into one of three sentiment categories.
    Maps:
      - negative  ->   '-'
      - neutral   ->   '|'
      - positive  ->   '+'
    """
    result = sentiment_analyzer(tweet_text)[0]
    label = result["label"]
    return label_map.get(label, "|")

def generate_sentiment_dna(tweets):
    """
    Generates a sentiment DNA string for a list of tweets.
    It sorts tweets chronologically and assigns each tweet a sentiment symbol.
    """
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = "".join(get_sentiment_symbol(tweet["text"]) for tweet in tweets)
    return dna


class SentimentDNAEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(SentimentDNAEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    # Debugging snippet to check sentiment label mapping across multiple languages.
    sample_tweets = [
        "I love this!",                      # expected positive (English)
        "This is terrible...",               # expected negative (English)
        "I am not sure about this.",         # expected neutral (English)
        "Absolutely fantastic work!",        # expected positive (English)
        "I hate waiting.",                   # expected negative (English)
        "J'adore ce produit!",               # expected positive (French: "I love this product!")
        "C'est vraiment nul.",               # expected negative (French: "This is really bad.")
        "No estoy seguro sobre esto.",       # expected neutral (Spanish: "I'm not sure about this.")
        "Das ist fantastisch!"               # expected positive (German: "That is fantastic!")
    ]

    print("Debugging Sentiment Mapping:")
    for tweet in sample_tweets:
        # Get the model's output.
        result = sentiment_analyzer(tweet)[0]
        # Map the result to our symbol.
        mapped_symbol = get_sentiment_symbol(tweet)
        print(f"Tweet: {tweet}")
        print(f"  Model output: {result}")  # includes label and score.
        print(f"  Mapped symbol: {mapped_symbol}\n")