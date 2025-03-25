import os
import re
import json
import emoji
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from datetime import datetime
from torchvision import transforms, models as tv_models  # for CNN encoder


# Load your pre-built emoji lookup table (assumed to be saved as JSON).
with open(os.path.join(
    os.path.dirname(__file__), '../emoji_lookup.json'), "r", encoding="utf-8") as f: 
    emoji_lookup = json.load(f)

# -------------------------
# Digital DNA Functions for Emojis
# -------------------------
def get_emoji_dna_symbol(tweet_text):
    # Extract all emojis from the tweet using the emoji package.
    # The emoji_list function returns a list of dicts with the key 'emoji'
    extracted_emojis = [entry['emoji'] for entry in emoji.emoji_list(tweet_text)]
    
    # Look up the category for each emoji using the emoji_lookup dictionary.
    # If an emoji is not in the lookup, it will be skipped.
    categories = {emoji_lookup.get(e, None) for e in extracted_emojis if e in emoji_lookup}
    
    if len(categories) == 0:
        return "N"  # No emoji present.
    elif len(categories) == 1:
        return categories.pop()  # Exactly one emoji category present.
    else:
        return "X"  # Multiple distinct emoji categories present.

def generate_emoji_dna(tweets):
    # Sort tweets chronologically.
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    # Concatenate the emoji DNA symbol for each tweet.
    dna = "".join(get_emoji_dna_symbol(tweet["text"]) for tweet in tweets)
    return dna

# ---------------------------------------------------------------------------

# -------------------------
# CNN-based DNA Encoder (unchanged)
# -------------------------
def dna_to_tensor(dna, 
                  mapping={\
                      "N": 0, 
                      "S": 26,    # Smileys & Emotion
                      "P": 52,    # People & Body
                      "A": 78,    # Animals & Nature
                      "F": 104,    # Food & Drink
                      "T": 130,    # Travel & Places
                      "R": 156,    # Activities
                      "O": 182,   # Objects
                      "Y": 208,   # Symbols
                      "L": 234,   # Flags
                      "X": 255,   
                          },
                  desired_size=64):
    """
    Converts a DNA string into a grayscale image that is resized to desired_size
    and then converted to a normalized RGB tensor.
    """
    values = [mapping[symbol] for symbol in dna if symbol in mapping]
    length = len(values)
    n = int(np.ceil(np.sqrt(length)))
    total = n * n
    values += [mapping["N"]] * (total - length)
    arr = np.array(values, dtype=np.uint8).reshape((n, n))
    img = Image.fromarray(arr, mode="L")
    img = img.resize((desired_size, desired_size), Image.NEAREST)
    img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # scales pixels to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img)
    return tensor

class DNACNNEncoder(nn.Module):
    def __init__(self, output_dim=384):
        super(DNACNNEncoder, self).__init__()
        # Use pretrained MobileNetV2 from torchvision.
        self.cnn = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
        self.features = self.cnn.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, output_dim)
        
    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
            pooled = self.avgpool(features)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.fc(pooled)
        return out

# Instantiate the CNN-based encoder and set to evaluation mode.
dna_cnn_encoder = DNACNNEncoder()
dna_cnn_encoder.eval()

def encode_dna_batch_cnn(dna_sequences, desired_size=64):
    tensors = [dna_to_tensor(seq, desired_size=desired_size) for seq in dna_sequences]
    input_tensor = torch.stack(tensors)  # shape: [batch, 3, desired_size, desired_size]
    with torch.no_grad():
        embeddings = dna_cnn_encoder(input_tensor)
    return embeddings.numpy()
