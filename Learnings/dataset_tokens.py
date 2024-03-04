import torch

# Define custom tensors
encoder_input = torch.tensor([1, 2, 3, 0, 0])  # Example sequence with padding tokens (0)
pad_token = 0

# Calculate attention scores (replace with your actual calculation logic)
attention_scores = torch.randn(1, 1, 5, 5)  # Sample attention scores
print("Original attention scores:\n", attention_scores)

# Create the mask
mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
print("Mask: ", mask)

# Mask out attention scores for padding tokens
attention_scores.masked_fill_(mask == 0, -1e9)
print("Masked attention scores:\n", attention_scores)
