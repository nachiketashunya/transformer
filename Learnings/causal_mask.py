import torch

def attention(query, key, value, mask):
  """Simple attention layer with causal masking."""
  scores = torch.matmul(query, key.transpose(-2, -1))  # Calculate attention scores
  print("Scores: ", scores)
  scores.masked_fill_(mask == 0, -1e9)  # Apply causal mask
  print("Scores: ", scores)  
  weights = torch.softmax(scores, dim=-1)  # Normalize scores
  print("Weights: ", weights)
  output = torch.matmul(weights, value)  # Weighted sum of values

  return output

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    mask = (mask == 0).type(torch.int)
    return mask

# Example usage
size = 5
d_model = 12
mask = causal_mask(size)

print("Mask: ", mask)

# Sample tensors (replace with your actual data)
query = torch.randn(1, 1, size, d_model)
key = torch.randn(1, size, d_model)
value = torch.randn(1, size, d_model)

output = attention(query, key, value, mask)

print("Output shape:", output.shape)
