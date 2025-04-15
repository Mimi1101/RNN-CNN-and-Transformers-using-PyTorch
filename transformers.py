import torch
import torch.nn as nn
import math

class Transformers(nn.Module):
    def __init__(self, vocab_size=10000, num_heads=4, num_layers=4, hidden_dim=512, max_sequence_length=256):
        super(Transformers, self).__init__()
        