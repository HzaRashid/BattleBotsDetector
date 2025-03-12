import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import os
from dna_train import DataLoader
from dna_train import UserMultimodalDataset
from dna_train import CrossAttentionFusionClassifier
from sentence_transformers import SentenceTransformer

cur_dir = os.path.dirname(__file__)
data_dir = os.path.join(cur_dir, "../data")
model_foldername = "models_test"
model_filename = "model_attn_smote_focal.pth"
saved_model_dir = os.path.join(cur_dir, f'../{model_foldername}/{model_filename}')

# Assume these have been defined in your code:
# - UserMultimodalDataset: your custom dataset
# - GMUFusionClassifier: your fusion model architecture
# - st_model: your SentenceTransformer model for text embeddings
# Load the SentenceTransformer (it will download/cache "all-MiniLM-L6-v2")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the test dataset (adjust the JSON path as needed)

test_dataset = UserMultimodalDataset(data_dir=data_dir, session_numbers=[11], st_model=st_model)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Recreate the model architecture with the same parameters
model = CrossAttentionFusionClassifier(text_dim=384, num_heads=4, fusion_hidden=256, num_classes=2)

# Load the saved state dictionary (change filename as appropriate)
model.load_state_dict(torch.load(saved_model_dir, map_location=torch.device('cpu')))
model.eval()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []

with torch.no_grad():
    for text_emb, img_tensor, labels, _ in test_loader:
        text_emb = text_emb.to(device)
        img_tensor = img_tensor.to(device)
        labels = labels.to(device)
        outputs = model(text_emb, img_tensor)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate performance metrics using sklearn
acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

print("Accuracy: {:.4f}".format(acc))
print("Classification Report:\n", report)
