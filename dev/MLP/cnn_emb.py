import torch
import torch.nn as nn
import numpy as np
from transformers import AutoFeatureExtractor, MobileViTForImageClassification

# Load the feature extractor and model from Hugging Face
feature_extractor = AutoFeatureExtractor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

# Replace the final classification layer with an identity mapping
model.classifier = nn.Identity()
model.eval()

# Create a dummy image (using a random numpy array with values between 0 and 255)
# Note: MobileViT expects images of size 224x224 with 3 channels.
dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

# Preprocess the dummy image using the feature extractor
inputs = feature_extractor(images=dummy_image, return_tensors="pt")

model.config.output_hidden_states = True

# Run a forward pass through the model without computing gradients
with torch.no_grad():
    features = model(**inputs)
    hidden_states = features.hidden_states
    # print(hidden_states)
    print(len(hidden_states))


print("Output features shape:", features.logits)
