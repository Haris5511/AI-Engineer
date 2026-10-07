import torch
import torch.nn.functional as F

# Example input (Batch=1, Sequence Length=4, Embedding Dimension=8)
x = torch.rand(1, 4, 8)

# Query, Key, Value projections
Wq = torch.nn.Linear(8, 8)
Wk = torch.nn.Linear(8, 8)
Wv = torch.nn.Linear(8, 8)

Q = Wq(x)
K = Wk(x)
V = Wv(x)

# Scaled Dot-Product Attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / (8 ** 0.5)
attention_weights = F.softmax(scores, dim=-1)

# Final Attention Output
output = torch.matmul(attention_weights, V)

print("Attention Weights:")
print(attention_weights)

print("\nOutput Shape:", output.shape)