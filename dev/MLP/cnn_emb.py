# import torch
# import torch.nn as nn
# import torchvision.models as models

# # Load a pre-trained MobileNetV2 model
# model = models.mobilenet_v2(pretrained=True)

# # Replace the classifier with an identity layer to extract the embedding directly
# model.classifier = nn.Identity()

# # Create a dummy input tensor with shape [batch_size, channels, height, width]
# # MobileNetV2 expects images of size 224x224 with 3 color channels
# dummy_input = torch.randn(1, 3, 224, 224)

# # Get the embedding by forwarding the dummy input through the modified model
# embedding = model(dummy_input)

# print("Embedding shape:", embedding.shape)

# print(embedding)
# from transformers import AutoFeatureExtractor, MobileViTForImageClassification
# import torch.nn as nn

# # Load the feature extractor and model from Hugging Face
# feature_extractor = AutoFeatureExtractor.from_pretrained("apple/mobilevit-small")
# model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

# # Inspect the model to find the classification head
# print(model)

# # Depending on the model's architecture, you might remove or replace the classification head.
# # For example, if the model has an attribute named `classifier`, you could do:
# if hasattr(model, "classifier"):
#     model.classifier = nn.Identity()

# model.eval()
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
