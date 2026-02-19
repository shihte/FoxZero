import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class FoxZeroNet(nn.Module):
    def __init__(self, num_res_blocks=5, num_channels=256):
        super().__init__()
        
        # Backbone
        # Input: [Batch, 11, 4, 13]
        self.conv_input = nn.Conv2d(11, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 4 * 13, 52) # 52 cards
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 4 * 13, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: [Batch, 11, 4, 13]
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)
        # We don't apply softmax here if using CrossEntropyLoss, or if we need log_probs.
        # But usually AlphaZero implementations output LogSoftmax or Softmax.
        # Let's output LogSoftmax for numerical stability with NLLLoss.
        # Or just raw logits. Let's return raw logits for policy.
        # Actually, MCTS needs probabilities.
        p = F.softmax(p, dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

    def predict(self, state_tensor):
        """
        Inference wrapper.
        state_tensor: np.array [11, 4, 13]
        Returns: policy (np.array [52]), value (float)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_tensor).unsqueeze(0) # Add batch dim
            p, v = self(x)
            return p.squeeze(0).numpy(), v.item()
