import os
import json
import re
import math
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms

# -------------------------
# Digital DNA Extraction (Content-based)
# -------------------------
def get_content_dna_symbol(tweet_text):
    """
    Returns a symbol based on the tweet content:
      - "R": if tweet appears to be a reply (starts with '@')
      - "U": if tweet contains URL(s) only.
      - "H": if tweet contains hashtag(s) only.
      - "M": if tweet contains mention(s) only (if not a reply).
      - "E": if tweet contains emoji(s) only.
      - "X": if tweet contains more than one type of entity.
      - "N": if tweet contains none of the above.
    """
    # Check for URLs, hashtags, and mentions using regex
    url_present = bool(re.search(r"https?://\S+", tweet_text))
    hashtag_present = bool(re.search(r"#\w+", tweet_text))
    mention_present = bool(re.search(r"@\w+", tweet_text))
    
    # If tweet starts with '@', consider it a reply
    if tweet_text.strip().startswith("@"):
        return "R"
    
    # Define a regex for emoji detection
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols and others
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "]+", flags=re.UNICODE)
    emoji_present = bool(emoji_pattern.search(tweet_text))
    
    # Count the number of entity types present (excluding reply which is handled above)
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
    """
    Given a list of tweets (each with 'text' and 'created_at'),
    sorts them chronologically and builds a DNA sequence based on tweet content.
    """
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = ""
    for tweet in tweets:
        symbol = get_content_dna_symbol(tweet["text"])
        dna += symbol
    return dna

# -------------------------
# Convert DNA Sequence to Image Tensor
# -------------------------
def dna_to_tensor(dna, mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255, "R": 32, "E": 32},
                  desired_size=256):
    """
    Converts a DNA sequence into a grayscale image tensor.
    Each symbol is mapped to a pixel value according to 'mapping'.
    The sequence is padded to form a square image and resized.
    """
    values = [mapping[symbol] for symbol in dna]
    length = len(values)
    n = math.ceil(math.sqrt(length))
    total = n * n
    # Pad with the value for "N" (assuming neutral)
    values += [mapping["N"]] * (total - length)
    arr = np.array(values, dtype=np.uint8).reshape((n, n))
    img = Image.fromarray(arr, mode="L")
    img = img.resize((desired_size, desired_size), Image.NEAREST)
    # Convert grayscale image to RGB and then to tensor
    img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # scales pixels to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img)
    return tensor

# -------------------------
# Example: Process a JSON File and Extract DNA Features
# -------------------------
def process_json_for_dna(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Build a dictionary of user labels (for demonstration)
    user_labels = {
        user['user_id']: user['is_bot'] for user in data['users']
    }
    # Group tweets by author_id
    users = {}
    for post in data["posts"]:
        author_id = post["author_id"]
        if author_id not in users:
            users[author_id] = []
        users[author_id].append(post)
    
    # For each user, generate the content-based digital DNA sequence and corresponding image tensor.
    results = {}
    for uid, tweets in users.items():
        dna = generate_content_dna(tweets)
        img_tensor = dna_to_tensor(dna)
        results[uid] = {
            "dna_sequence": dna,
            "image_tensor_shape": img_tensor.shape  # For verification
        }
        print(f"User {user_labels[uid]}: DNA = {dna}")
    return results

if __name__ == "__main__":
    # Replace with your JSON file path (e.g., "session_13_results.json")
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    json_path = os.path.join(data_dir, "session_13_results.json")
    process_json_for_dna(json_path)
