import os
import json
import re
import math
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer

# -------------------------
# Digital DNA Extraction (Content-based)
# -------------------------
def get_content_dna_symbol(tweet_text):
    """
    Returns a symbol based on the tweet content:
      - "R": if the tweet appears to be a reply (starts with '@')
      - "U": if the tweet contains URL(s) only.
      - "H": if the tweet contains hashtag(s) only.
      - "M": if the tweet contains mention(s) only (and not a reply).
      - "X": if the tweet contains more than one type of entity.
      - "N": if the tweet contains none of the above.
    """
    url_present = bool(re.search(r"https?://\S+", tweet_text))
    hashtag_present = bool(re.search(r"#\w+", tweet_text))
    mention_present = bool(re.search(r"@\w+", tweet_text))
    
    # If tweet starts with '@', treat it as a reply (separate symbol "R")
    if tweet_text.strip().startswith("@"):
        return "R"
    
    entity_count = sum([url_present, hashtag_present, mention_present])
    if entity_count == 0:
        return "N"
    elif entity_count == 1:
        if url_present:
            return "U"
        elif hashtag_present:
            return "H"
        elif mention_present:
            return "M"
    else:
        return "X"

def generate_content_dna(tweets):
    """
    Given a list of tweet dicts (with 'text' and 'created_at'),
    sorts them chronologically and returns a DNA sequence using content-based symbols.
    """
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = ""
    for tweet in tweets:
        symbol = get_content_dna_symbol(tweet["text"])
        dna += symbol
    return dna

def dna_to_tensor(dna, mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255, "R": 32},
                  desired_size=256):
    """
    Converts a DNA sequence into a grayscale image tensor.
    Each symbol is mapped to a pixel value according to 'mapping'.
    The sequence is padded to form a square, resized, converted to RGB,
    and then normalized for use with a CNN.
    """
    values = [mapping[symbol] for symbol in dna]
    length = len(values)
    n = math.ceil(math.sqrt(length))
    total = n * n
    values += [mapping["N"]] * (total - length)  # pad with neutral value for "N"
    arr = np.array(values, dtype=np.uint8).reshape((n, n))
    img = Image.fromarray(arr, mode="L")
    img = img.resize((desired_size, desired_size), Image.NEAREST)
    img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # scales to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img)
    return tensor

# -------------------------
# PyTorch Dataset for Multimodal Data
# -------------------------
class UserMultimodalDataset(Dataset):
    def __init__(self, json_path, st_model, mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255, "R": 32},
                 desired_size=256):
        """
        Loads a JSON file containing 'users' and 'posts'. For each user, we:
          - Group posts by user_id and generate a content-based digital DNA sequence.
          - Convert the DNA sequence into an image tensor.
          - Obtain the user description and compute its embedding.
        Ground-truth labels are assumed to be present in the user objects under "is_bot".
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)
        
        # Create a dictionary for user information (including description and label)
        self.users = {}
        for user in self.data.get("users", []):
            # Assume each user dict has keys: "user_id", "description", and "is_bot"
            uid = user["user_id"]
            self.users[uid] = {
                "description": user.get("description", ""),
                "is_bot": user.get("is_bot", 0)
            }
        
        # Group posts by author_id
        self.user_posts = {}
        for post in self.data.get("posts", []):
            uid = post["author_id"]
            if uid not in self.user_posts:
                self.user_posts[uid] = []
            self.user_posts[uid].append(post)
        
        # Use only users that have both user info and posts
        self.user_ids = [uid for uid in self.users if uid in self.user_posts]
        self.st_model = st_model
        self.mapping = mapping
        self.desired_size = desired_size

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        # Process posts to generate digital DNA from content
        posts = self.user_posts[uid]
        dna = generate_content_dna(posts)
        image_tensor = dna_to_tensor(dna, mapping=self.mapping, desired_size=self.desired_size)
        # Compute text embedding from user description
        description = self.users[uid]["description"]
        text_embedding = self.st_model.encode(description)
        text_embedding = torch.tensor(text_embedding, dtype=torch.float)
        label = torch.tensor(int(self.users[uid]["is_bot"]), dtype=torch.long)
        return text_embedding, image_tensor, label

# -------------------------
# Multimodal Fusion Model
# -------------------------
class MultimodalFusionClassifier(nn.Module):
    def __init__(self, text_dim=384, image_feature_dim=512, fusion_hidden=256, num_classes=2):
        """
        A multimodal classifier that fuses:
          - A text embedding (from all-MiniLM-L6-v2, dimension text_dim)
          - An image branch processing the digital DNA image via VGG16.
        The two feature vectors are concatenated and passed to a simple classifier.
        """
        super(MultimodalFusionClassifier, self).__init__()
        # Pretrained VGG16 for the image branch
        self.image_model = models.vgg16(pretrained=True)
        # Remove the final classifier layers by replacing them with Identity
        self.image_model.classifier = nn.Identity()  # output shape [batch, 25088]
        self.image_fc = nn.Linear(25088, image_feature_dim)
        self.relu = nn.ReLU()
        # Fusion layer: concatenate text and image features
        fusion_dim = text_dim + image_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, num_classes)
        )
    
    def forward(self, text_embedding, image_tensor):
        # Process image through CNN
        image_features = self.image_model(image_tensor)
        image_features = self.image_fc(image_features)
        image_features = self.relu(image_features)
        # Concatenate text and image features
        fused_features = torch.cat((text_embedding, image_features), dim=1)
        logits = self.classifier(fused_features)
        return logits

# -------------------------
# Training Loop
# -------------------------
def train_model(dataset, model, num_epochs=5, batch_size=4, learning_rate=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for text_emb, img_tensor, labels in dataloader:
            text_emb = text_emb.to(device)
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(text_emb, img_tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    # Load the all-MiniLM-L6-v2 model for text embeddings
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Path to your JSON file
    json_path = os.path.join(data_dir, f"session_13_results.json")
    # Create the dataset
    dataset = UserMultimodalDataset(json_path, st_model)
    # Instantiate the multimodal fusion classifier.
    # Note: all-MiniLM-L6-v2 outputs 384-d embeddings.
    model = MultimodalFusionClassifier(text_dim=384, image_feature_dim=512, fusion_hidden=256, num_classes=2)
    # Train the model
    train_model(dataset, model, num_epochs=5, batch_size=4)
