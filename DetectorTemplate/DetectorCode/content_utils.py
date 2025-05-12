import re
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from datetime import datetime
from torchvision import transforms
# Removed unused torchvision models import since we're switching to MobileViT-Small

from transformers import MobileViTImageProcessor, MobileViTForImageClassification

# -------------------------
# Digital DNA Functions
# -------------------------
def get_content_dna_symbol(tweet_text):
    url_present = bool(re.search(r"https?://\S+", tweet_text))
    hashtag_present = bool(re.search(r"#\w+", tweet_text))
    mention_present = bool(re.search(r"@\w+", tweet_text))
    
    entity_types = sum([url_present, hashtag_present, mention_present])
    
    if entity_types == 0:
        return "N"
    elif entity_types == 1:
        if url_present:
            return "U"
        elif hashtag_present:
            return "H"
        elif mention_present:
            return "M"
    else:
        return "X"

def generate_content_dna(tweets):
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = "".join(get_content_dna_symbol(tweet["text"]) for tweet in tweets)
    return dna
# ---------------------------------------------------------------------------

DESIRED_SIZE = 128
# -------------------------
# CNN-based DNA Encoder
# -------------------------
def dna_to_tensor(dna, 
                  mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255},
                  desired_size=DESIRED_SIZE):
    """
    Converts a DNA string into a grayscale image that is resized to desired_size,
    then converts it to a normalized RGB tensor.
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

# Load MobileViT-Small model and feature extractor from Hugging Face
feature_extractor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
dna_cnn_encoder = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
# Replace the final classification layer with an identity mapping
dna_cnn_encoder.classifier = nn.Identity()
dna_cnn_encoder.eval()

def encode_content_dna_batch_cnn(dna_sequences, desired_size=DESIRED_SIZE):
    tensors = [dna_to_tensor(seq, desired_size=desired_size) for seq in dna_sequences]
    input_tensor = torch.stack(tensors)  # shape: [batch, 3, desired_size, desired_size]
    with torch.no_grad():
        # Note: MobileViT expects the input under the keyword 'pixel_values'
        outputs = dna_cnn_encoder(pixel_values=input_tensor)
        embeddings = outputs.logits
    return embeddings.numpy()


if __name__ == "__main__":
    # Sample test tweets
    tweets = [
        {"created_at": "2023-04-01T12:00:00Z", "text": "Hello world! Check out http://example.com"},
        {"created_at": "2023-04-01T13:00:00Z", "text": "Another tweet with #hashtag"},
        {"created_at": "2023-04-01T14:00:00Z", "text": "Tweet with @mention"},
        {"created_at": "2023-04-01T15:00:00Z", "text": "Just a simple tweet."}
    ]

    # Generate DNA string from tweets
    dna_str = generate_content_dna(tweets)
    print("Generated DNA:", dna_str)

    # Encode the generated DNA using MobileViT-Small CNN encoder
    embeddings = encode_content_dna_batch_cnn([dna_str])
    print("Embeddings shape:", embeddings.shape)