import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import json
import os


with open('sample_annotation.json') as f:
    annotations = json.load(f)

with open('instance.json') as f:
    instances = json.load(f)

with open('category.json') as f:
    categories = json.load(f)


cat_map = {c['token']: c['name'] for c in categories}


ped_instances = [i['token'] for i in instances if 'human.pedestrian' in cat_map[i['category_token']]]

print(f"{len(ped_instances)} pedestrians.")

trajectories = defaultdict(list)

for ann in annotations:
    if ann['instance_token'] in ped_instances:
        trajectories[ann['instance_token']].append({
            'x': ann['translation'][0],
            'y': ann['translation'][1],
            'sample_token': ann['sample_token']
        })

print(len(trajectories.keys()))

OBS_LEN = 4       # 2 seconds at 2Hz
PRED_LEN = 6      # 3 seconds at 2Hz
WINDOW_SIZE = OBS_LEN + PRED_LEN  # 10 frames total

obs_windows = []
target_windows = []

for ped_id, path in trajectories.items():

    coords = np.array([[frame['x'], frame['y']] for frame in path])  # [T, 2]

    if len(coords) < WINDOW_SIZE:
        continue

    # Slide
    for i in range(len(coords) - WINDOW_SIZE + 1):
        window = coords[i : i + WINDOW_SIZE]  # [10, 2]

        # Normalize — set start point as (0, 0)
        origin = window[0].copy()
        window = window - origin

        obs_windows.append(window[:OBS_LEN])      # [4, 2]
        target_windows.append(window[OBS_LEN:])   # [6, 2]

# Convert to tensors
obs_tensor    = torch.FloatTensor(np.array(obs_windows))     # [N, 4, 2]
target_tensor = torch.FloatTensor(np.array(target_windows))  # [N, 6, 2]

print(f"obs shape:    {obs_tensor.shape}")
print(f"target shape: {target_tensor.shape}")


dataset = TensorDataset(obs_tensor, target_tensor)

# 80% train, 20% validation split
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)


class VanillaLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, pred_len=6):
        super().__init__()
        self.pred_len = pred_len

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Output
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, obs):

        _, (h, c) = self.encoder(obs)


        # Start decoding
        dec_input = obs[:, -1:, :]  # [32, 1, 2]  last known position
        predictions = []

        # DECODE
        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.fc(out)          # [32, 1, 2]
            predictions.append(pred)
            dec_input = pred

        return torch.cat(predictions, dim=1)  # [32, 6, 2]


def calculate_ade(pred, target):
    displacement = torch.sqrt(((pred - target)**2).sum(dim=-1))
    return displacement.mean()

def calculate_fde(pred, target):
    final = torch.sqrt(((pred[:,-1] - target[:,-1])**2).sum(dim=-1))
    return final.mean()


# Setup
model     = VanillaLSTM(input_size=2, hidden_size=64, pred_len=6)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
EPOCHS    = 100

print("Starting Training...\n")

for epoch in range(EPOCHS):
    # -------- TRAIN --------
    model.train()
    train_loss, train_ade, train_fde = 0, 0, 0

    for obs, target in train_loader:
        optimizer.zero_grad()        # reset gradients
        pred = model(obs)            # forward pass
        loss = criterion(pred, target)
        loss.backward()              # backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()             # update parameters

        train_loss += loss.item()
        train_ade  += calculate_ade(pred, target)
        train_fde  += calculate_fde(pred, target)

    # -------- VALIDATE --------
    model.eval()
    val_loss, val_ade, val_fde = 0, 0, 0

    with torch.no_grad():
        for obs, target in val_loader:
            pred = model(obs)
            val_loss += criterion(pred, target).item()
            val_ade  += calculate_ade(pred, target)
            val_fde  += calculate_fde(pred, target)

    # -------- LOG --------
    n_train = len(train_loader)
    n_val   = len(val_loader)

    print(
        f"Epoch [{epoch+1:3d}/{EPOCHS}] "
        f"| Train Loss: {train_loss/n_train:.4f}  ADE: {train_ade/n_train:.4f}  FDE: {train_fde/n_train:.4f} "
        f"| Val Loss: {val_loss/n_val:.4f}  ADE: {val_ade/n_val:.4f}  FDE: {val_fde/n_val:.4f}"
    )

torch.save(model.state_dict(), 'vanilla_lstm.pth')
print("\nModel saved to vanilla_lstm.pth")
