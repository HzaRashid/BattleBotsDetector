from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
import torch

from transformers import MobileViTImageProcessor, MobileViTForImageClassification

def get_time_dna_symbol(current_time, previous_time):
    """
    Returns a DNA symbol based on the time difference between two posts.
    current_time and previous_time should be datetime objects.
    """
    # For the first post, nothing to compare.
    if previous_time is None:
        return "n"
    
    delta = current_time - previous_time
    seconds = delta.total_seconds()
    
    if seconds < 1:
        return "o"  # less than 1 second
    elif seconds < 60:
        return "s"  # less than one minute
    elif seconds < 3600:
        return "m"  # less than one hour
    elif seconds < 21600:  # 6 hours = 6*3600
        return "h"  
    elif seconds < 43200:  # 12 hours = 12*3600
        return "u"  
    else:
        return "x"  # else

def generate_time_dna(tweets):
    """
    Generates a DNA string based on the time difference between posts.
    
    Each tweet in 'tweets' must have a "created_at" field in ISO format.
    The function sorts the tweets chronologically and assigns a symbol
    for each tweet based on the time since the previous post.
    """
    tweets.sort(key=lambda tweet: datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00")))
    time_dna = ""
    prev_time = None
    for tweet in tweets:
        current_time = datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00"))
        symbol = get_time_dna_symbol(current_time, prev_time)
        time_dna += symbol
        prev_time = current_time
    return time_dna

# Mapping: "n: first post, o: less than 1 second, s: less than one minute, 
# m: less than one hour, h: less than 6 hours, u: less than 12 hours, x: else"
time_dna_mapping = {
    "n": 0,  "o": 255, "s": 200, "m": 150, "h": 100,
    "u": 80, "x": 2
}

DESIRED_SIZE = 128

def time_dna_to_tensor(time_dna, mapping=time_dna_mapping, desired_size=DESIRED_SIZE):
    """
    Converts a time DNA string into a grayscale image that is resized to desired_size,
    then converts it to a normalized RGB tensor.
    """
    # Convert characters to mapped values.
    values = [mapping[symbol] for symbol in time_dna if symbol in mapping]
    length = len(values)
    # Determine the size of the square.
    n = int(np.ceil(np.sqrt(length)))
    total = n * n
    # Pad the sequence with a baseline value (for "n").
    values += [mapping["n"]] * (total - length)
    # Reshape to a 2D array.
    arr = np.array(values, dtype=np.uint8).reshape((n, n))
    # Create a grayscale image.
    img = Image.fromarray(arr, mode="L")
    # Resize to the desired dimensions.
    img = img.resize((desired_size, desired_size), Image.NEAREST)
    # Convert to RGB (to reuse a CNN expecting 3-channel input).
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
time_dna_encoder = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
# Replace the final classification layer with an identity mapping
time_dna_encoder.classifier = nn.Identity()
time_dna_encoder.eval()

def encode_time_dna_batch_cnn(time_dna_sequences=[], desired_size=DESIRED_SIZE):
    tensors = [time_dna_to_tensor(seq, desired_size=desired_size) for seq in time_dna_sequences]
    input_tensor = torch.stack(tensors)  # shape: [batch, 3, desired_size, desired_size]
    with torch.no_grad():
        # MobileViT expects inputs under the key 'pixel_values'
        outputs = time_dna_encoder(pixel_values=input_tensor)
        embeddings = outputs.logits
    return embeddings.numpy()

if __name__ == "__main__":
    # Dummy tweet data with ISO formatted dates.
    tweets = [
        {"created_at": "2025-03-15T12:00:00Z"},
        {"created_at": "2025-03-15T12:00:00.500000Z"},  # 0.5 sec later
        {"created_at": "2025-03-15T12:00:30Z"},           # 29.5 sec later
        {"created_at": "2025-03-15T12:30:00Z"},           # 29.5 min later
        {"created_at": "2025-03-15T18:00:00Z"},           # 5.5 hours later
        {"created_at": "2025-03-16T12:00:00Z"}            # 18 hours later
    ]
    
    # Generate time DNA from the tweets.
    time_dna = generate_time_dna(tweets)
    print("Time DNA:", time_dna)
    
    # Encode the generated time DNA using MobileViT-Small
    embeddings = encode_time_dna_batch_cnn([time_dna])
    print("Embeddings shape:", embeddings.shape)
