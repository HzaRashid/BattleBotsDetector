from gliner import GLiNER

# Load the GLiNER model
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Sample tweets
tweets = [
    "Elon Musk just announced that Tesla is working on a new AI chip!",
    "The Oscars 2024 will take place in Los Angeles on March 10.",
    "Cristiano Ronaldo scored a hat-trick for Al Nassr in the Saudi Pro League.",
    "Apple has unveiled the latest iPhone 15 Pro with groundbreaking features.",
    "NASA's Artemis program aims to land humans on the Moon again by 2026."
]

# Define entity labels of interest
labels = ["person", "organization", "date", "event", "teams", "product", "location"]

# Extract entities from each tweet
for tweet in tweets:
    entities = model.predict_entities(tweet, labels)
    print(f"Tweet: {tweet}")
    for entity in entities:
        print(f"  {entity['text']} => {entity['label']}")
    print("-" * 50)
