import os
import re
import math
import json
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer

# -------------------------
# Digital DNA Extraction (Content-based) with Emoji Label
# -------------------------
emoji_pattern = re.compile("[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U0001F700-\U0001F77F"  # alchemical symbols
    u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA00-\U0001FA6F"  # Chess Symbols, etc.
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+", flags=re.UNICODE)

def get_content_dna_symbol(tweet_text):
    url_present = bool(re.search(r"https?://\S+", tweet_text))
    hashtag_present = bool(re.search(r"#\w+", tweet_text))
    mention_present = bool(re.search(r"@\w+", tweet_text))
    
    if tweet_text.strip().startswith("@"):
        return "R"
    
    emoji_present = bool(emoji_pattern.search(tweet_text))
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

import numpy as np

def mutate_dna(dna, mutation_rate=0.05, deletion_rate=0.02, splicing_rate=0.1):
    """
    Randomly mutates the DNA string by performing:
      - Substitution: With probability `mutation_rate`, a character is replaced with another.
      - Deletion: With probability `deletion_rate`, a character is removed.
      - Splicing: With probability `splicing_rate`, the first 3 characters are moved to the end.
    
    Parameters:
        dna (str): Original digital DNA string.
        mutation_rate (float): Probability for each character to be substituted.
        deletion_rate (float): Probability for each character to be deleted.
        splicing_rate (float): Probability to perform a splicing operation on the entire string.
        
    Returns:
        str: Mutated DNA string.
    """
    if not dna: return dna
    symbols = ["N", "U", "H", "M", "X", "R", "E"]
    mutated_chars = []
    
    for char in dna:
        # First, decide if this character should be deleted.
        if np.random.rand() < deletion_rate:
            continue  # Skip adding this character (deletion).
        # Then, decide if we should substitute it.
        if np.random.rand() < mutation_rate:
            # Substitute with a different symbol.
            possible = [s for s in symbols if s != char]
            new_char = np.random.choice(possible)
            mutated_chars.append(new_char)
        else:
            mutated_chars.append(char)
    
    mutated_dna = "".join(mutated_chars)
    
    # Random splicing
    if len(mutated_dna) > 1 and np.random.rand() < splicing_rate:
        splice_index = np.random.choice(range(1, len(mutated_dna)))
        mutated_dna = mutated_dna[splice_index:] + mutated_dna[:splice_index]
    
    return mutated_dna


def generate_content_dna(tweets):
    tweets.sort(key=lambda x: datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")))
    dna = "".join(get_content_dna_symbol(tweet["text"]) for tweet in tweets)
    return dna

def dna_to_tensor(dna, mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255, "R": 32, "E": 32},
                  desired_size=256):
    values = [mapping[symbol] for symbol in dna]
    length = len(values)
    n = math.ceil(math.sqrt(length))
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

# -------------------------
# Dataset Definition
# -------------------------
class UserMultimodalDataset(Dataset):
    def __init__(self, data_dir, session_numbers, st_model,
                 mapping={"N": 0, "U": 64, "H": 128, "M": 192, "X": 255, "R": 32, "E": 32},
                 desired_size=256):
        self.users = {}
        self.user_posts = {}
        self.positive_ct = 0

        for num in session_numbers:
            json_file = os.path.join(data_dir, f"session_{num}_results.json")
            with open(json_file, "r") as f:
                data = json.load(f)

            for user in data.get("users", []):
                uid = user["user_id"]
                if uid not in self.users:
                    label = user.get("is_bot", 0)
                    self.users[uid] = {
                        "description": user.get("description", ""),
                        "is_bot": label
                    }
                    self.positive_ct += label

            for post in data.get("posts", []):
                uid = post["author_id"]
                if uid not in self.user_posts:
                    self.user_posts[uid] = []
                self.user_posts[uid].append(post)
        
        self.user_ids = [uid for uid in self.users if uid in self.user_posts]
        np.random.RandomState(seed=42).shuffle(self.user_ids)
        self.st_model = st_model
        self.mapping = mapping
        self.desired_size = desired_size

        print('HERE:', self.positive_ct)
        
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        posts = self.user_posts[uid]
        dna = generate_content_dna(posts)
        image_tensor = dna_to_tensor(dna, mapping=self.mapping, desired_size=self.desired_size)
        description = self.users[uid]["description"]
        text_embedding = self.st_model.encode(description)
        text_embedding = torch.tensor(text_embedding, dtype=torch.float)
        label = torch.tensor(int(self.users[uid]["is_bot"]), dtype=torch.long)
        return text_embedding, image_tensor, label

# -------------------------
# Cross-Attention Fusion Model with MobileNetV2
# -------------------------
class CrossAttentionFusionClassifier(nn.Module):
    def __init__(self, text_dim=384, num_heads=4, fusion_hidden=256, num_classes=2):
        super(CrossAttentionFusionClassifier, self).__init__()
        self.image_model = models.mobilenet_v2(pretrained=True)
        self.features_extractor = self.image_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.image_proj = nn.Conv2d(1280, text_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)
        fusion_input_dim = text_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, num_classes)
        )
        
    def forward(self, text_embedding, image_tensor):
        feat = self.features_extractor(image_tensor)
        feat = self.avgpool(feat)
        feat = self.image_proj(feat)
        batch_size, d, h, w = feat.shape
        tokens = feat.view(batch_size, d, h * w).permute(0, 2, 1)
        query = text_embedding.unsqueeze(1)
        attn_output, _ = self.attention(query, tokens, tokens)
        attn_output = attn_output.squeeze(1)
        fused = torch.cat([text_embedding, attn_output], dim=1)
        logits = self.classifier(fused)
        return logits

# -------------------------
# Focal Loss Implementation
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# -------------------------
# Helper: Image Augmentation for Minority Samples
# -------------------------
def augment_image(img_tensor):
    """
    Unnormalizes the image tensor, converts it to a PIL image,
    applies random augmentation, and re-normalizes.
    Assumes normalization parameters matching those in dna_to_tensor.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Inverse normalization transform
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    # Unnormalize and clip to valid range
    unnorm_tensor = unnormalize(img_tensor)
    unnorm_tensor = torch.clamp(unnorm_tensor, 0, 1)
    # Convert to PIL image
    pil_img = transforms.ToPILImage()(unnorm_tensor.cpu())
    # Define augmentation pipeline
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    augmented_tensor = augmentation(pil_img)
    return augmented_tensor

# -------------------------
# SMOTE Oversampling Function (on Text Embeddings) with Image Augmentation
# -------------------------
def create_oversampled_dataset(train_dataset):
    """
    Extracts text embeddings and DNA strings from train_dataset, applies SMOTE on the text embeddings,
    and for synthetic minority samples, applies digital DNA mutations before converting to image tensors.
    For majority samples, a random original image tensor is used.
    Returns a new TensorDataset with oversampled features.
    """
    from imblearn.over_sampling import SMOTE
    all_text_embs = []
    all_img_tensors = []
    all_labels = []
    all_dna = []  # Store raw DNA strings for potential mutation
    
    for i in range(len(train_dataset)):
        text_emb, img_tensor, label, dna = train_dataset[i]
        all_text_embs.append(text_emb.numpy())
        all_img_tensors.append(img_tensor.numpy())
        all_labels.append(label.item())
        all_dna.append(dna)
    
    X = np.vstack(all_text_embs)
    y = np.array(all_labels)
    
    # Apply SMOTE on text embeddings
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    # Prepare pools for image tensors and DNA strings by class
    minority_indices = [i for i, lab in enumerate(y) if lab == 1]
    majority_indices = [i for i, lab in enumerate(y) if lab == 0]
    minority_img_tensors = [all_img_tensors[i] for i in minority_indices]
    minority_dna = [all_dna[i] for i in minority_indices]
    majority_img_tensors = [all_img_tensors[i] for i in majority_indices]
    
    new_text_embs = []
    new_img_tensors = []
    new_labels = []
    
    for i in range(len(X_res)):
        label = y_res[i]
        new_text_embs.append(X_res[i])
        if label == 1:
            # For synthetic minority samples, pick a random minority DNA string and mutate it.
            rand_idx = np.random.choice(len(minority_dna))
            orig_dna = minority_dna[rand_idx]
            # Apply mutation only for oversampled (synthetic) samples.
            mutated_dna = mutate_dna(orig_dna, mutation_rate=0.05, deletion_rate=0.02, splicing_rate=0.1)
            # Convert the mutated DNA to an image tensor.
            new_img = dna_to_tensor(mutated_dna, mapping=train_dataset.mapping, desired_size=train_dataset.desired_size)
            new_img_tensors.append(new_img.numpy())
        else:
            # For majority samples, randomly pick an image tensor.
            rand_idx = np.random.choice(len(majority_img_tensors))
            new_img_tensors.append(majority_img_tensors[rand_idx])
        new_labels.append(label)
    
    new_text_embs = torch.tensor(new_text_embs, dtype=torch.float)
    new_img_tensors = torch.tensor(new_img_tensors, dtype=torch.float)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    
    return TensorDataset(new_text_embs, new_img_tensors, new_labels)



# -------------------------
# Training Loop Using Focal Loss and SMOTE Oversampling
# -------------------------
def train_model(dataset, model, num_epochs=10, batch_size=4, learning_rate=1e-4):
    oversampled_dataset = create_oversampled_dataset(dataset)
    dataloader = DataLoader(oversampled_dataset, batch_size=batch_size, shuffle=True)
    # Example: using Focal Loss with gamma=3
    criterion = FocalLoss(alpha=1, gamma=3, reduction="mean")
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
# Main Pipeline Example for Cross-Attention Fusion with Focal Loss + SMOTE + Augmentation
# -------------------------
if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")
    model_dir = os.path.join(cur_dir, "../models_test")
    session_numbers = [12, 13]  # Specify session numbers
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    train_dataset = UserMultimodalDataset(data_dir, session_numbers, st_model)

    model_attn = CrossAttentionFusionClassifier(text_dim=384, num_heads=4, fusion_hidden=256, num_classes=2)

    train_model(train_dataset, model_attn, num_epochs=5, batch_size=4, learning_rate=1e-4)

    # Save the trained cross-attention model
    torch.save(model_attn.state_dict(), os.path.join(model_dir, "model_attn_smote_focal.pth"))
    print("Model saved to 'model_attn_smote_focal.pth'")
