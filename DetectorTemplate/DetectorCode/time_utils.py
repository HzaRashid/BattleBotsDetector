from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from torchvision import transforms, models as tv_models  # for CNN encoder

from torch import nn
import torch


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
    # Sort tweets by creation time.
    tweets.sort(key=lambda tweet: datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00")))
    
    time_dna = ""
    prev_time = None
    for tweet in tweets:
        current_time = datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00"))
        symbol = get_time_dna_symbol(current_time, prev_time)
        time_dna += symbol
        prev_time = current_time
    return time_dna


"n: first post (nothing to compute), o: less than 1 second, s: less than one minute, m: less than one hour, h: less than 6 hours, u: less than 12 hours, t: less than 24 hours, d: less than 48 hours, w: less than a week, j: less than a month, q: less than three months, v: less than 6 months, y: less than a year, x: else"
time_dna_mapping = {
    "n": 0,  "o": 255, "s": 200, "m": 150, "h": 100,
    "u": 80, "x": 2
}


def time_dna_to_tensor(time_dna, mapping=time_dna_mapping, desired_size=64):
    # Convert characters to mapped values.
    values = [mapping[symbol] for symbol in time_dna if symbol in mapping]
    length = len(values)
    # Determine the size of the square.
    n = int(np.ceil(np.sqrt(length)))
    total = n * n
    # Pad the sequence with a baseline value (e.g., for "n").
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

class TimeDNAEncoder(nn.Module):
    def __init__(self, output_dim=384):
        super(TimeDNAEncoder, self).__init__()
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
    
# Instantiate and set to evaluation mode.
time_dna_encoder = TimeDNAEncoder()
time_dna_encoder.eval()

def encode_time_dna_batch_cnn(time_dna_sequences=[], desired_size=64):
    tensors = [time_dna_to_tensor(seq, desired_size=desired_size) for seq in time_dna_sequences]
    input_tensor = torch.stack(tensors)  # shape: [batch, 3, desired_size, desired_size]
    with torch.no_grad():
        embeddings = time_dna_encoder(input_tensor)
    return embeddings.numpy()

# Example usage:
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
    
    print("Time DNA:", generate_time_dna(tweets))