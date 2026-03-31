import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

def add_noise(traj, sigma=0.1):
    noise = np.random.normal(0, sigma, traj.shape)
    return traj + noise


def build_grid(target, neighbors, grid_size=4, cell_size=1.0):
    grid = np.zeros((grid_size, grid_size))
    
    center = grid_size // 2
    
    for nx, ny in neighbors:
        dx = nx - target[0]
        dy = ny - target[1]
        
        gx = int(dx / cell_size) + center
        gy = int(dy / cell_size) + center
        
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            grid[gx, gy] = 1
    
    return grid.flatten()



class SocialLSTM(nn.Module):
    def __init__(self,
                 pos_dim    = 2,
                 social_dim = 16,   # 16
                 embed_dim  = 64,
                 hidden_dim = 64,
                 pred_len   = 6):
        super().__init__()
        self.pred_len = pred_len
 
        # FIX: declare the layers that forward() uses
        self.pos_fc    = nn.Linear(pos_dim,    embed_dim)
        self.social_fc = nn.Linear(social_dim, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, pos_dim)
 
    def forward(self, x, social):
        B, T, _ = x.shape          # FIX: indentation — body must be inside forward
 
        embeddings = []
        for t in range(T):
            pos = x[:, t, :]        # (B, 2)
            soc = social[:, t, :]   # (B, 16)
 
            pos_emb = torch.relu(self.pos_fc(pos))      # (B, 64)
            soc_emb = torch.relu(self.social_fc(soc))   # (B, 64)
 
            combined = pos_emb + soc_emb                # (B, 64)
            embeddings.append(combined)
 
        embeddings = torch.stack(embeddings, dim=1)     # (B, T, 64)
 
        _, (h, c) = self.lstm(embeddings)               # h: (1, B, hidden)
 

        dec_input = x[:, -1:, :]   # (B, 1, 2) — last observed step
        outputs   = []
        for _ in range(self.pred_len):
            pos_emb  = torch.relu(self.pos_fc(dec_input.squeeze(1)))
            soc_zero = torch.zeros(B, 16, device=x.device)  # no future social
            soc_emb  = torch.relu(self.social_fc(soc_zero))
            combined = (pos_emb + soc_emb).unsqueeze(1)             # (B,1,64)
            out, (h, c) = self.lstm(combined, (h, c))
            step = self.fc(out)          # (B, 1, 2)
            outputs.append(step)
            dec_input = step             # teacher-forcing off at inference
 
        return torch.cat(outputs, dim=1)     # (B, pred_len, 2)



import json
from collections import defaultdict

def load_data(data_dir="."):
    with open("sample_annotation.json") as f:
        annotations = json.load(f)
    with open("instance.json") as f:
        instances = json.load(f)
    with open("category.json") as f:
        categories = json.load(f)
    with open("sample.json") as f:                          
        samples = json.load(f)

    cat_map = {c["token"]: c["name"] for c in categories}
    sample_order = {s["token"]: s["timestamp"] for s in samples}  

    ped_tokens = {
        i["token"]
        for i in instances
        if "human.pedestrian" in cat_map.get(i["category_token"], "")
    }

    trajectories = defaultdict(list)
    for ann in annotations:
        if ann["instance_token"] in ped_tokens:
            trajectories[ann["instance_token"]].append((
                ann["sample_token"],
                ann["translation"][0],
                ann["translation"][1],
            ))

    frame_map = defaultdict(list)
    for ann in annotations:
        if ann["instance_token"] in ped_tokens:
            frame_map[ann["sample_token"]].append((
                ann["instance_token"],
                ann["translation"][0],
                ann["translation"][1],
            ))

    for pid in trajectories:                                
        trajectories[pid].sort(key=lambda x: sample_order.get(x[0], 0))

    return trajectories, frame_map


def build_windows(trajectories, frame_map):
    obs_windows    = []
    target_windows = []
    social_windows = []
    OBS_LEN     = 4
    PRED_LEN    = 6
    WINDOW_SIZE = OBS_LEN + PRED_LEN  
 
    for ped_id, path in trajectories.items():
        if len(path) < WINDOW_SIZE:
            continue
 
        sample_tokens = [p[0] for p in path]
        coords        = np.array([[p[1], p[2]] for p in path])  # (T, 2)
 
        for i in range(len(coords) - WINDOW_SIZE + 1):
            window        = coords[i : i + WINDOW_SIZE].copy()
            token_window  = sample_tokens[i : i + WINDOW_SIZE]
 
            # Normalise to first observed position
            origin = window[0].copy()
            window = window - origin
 
            # Displacement sequence
            disp       = np.zeros_like(window)
            disp[1:]   = window[1:] - window[:-1]
 
            # Social grids for observation frames only
            social_seq = []
            for t in range(OBS_LEN):
                time_token = token_window[t]
                target_pos = window[t]                    # normalised
 
                neighbors = []
                for pid, nx, ny in frame_map[time_token]:
                    if pid != ped_id:
                        neighbors.append((nx - origin[0], ny - origin[1]))
 
                grid = build_grid(target_pos, neighbors)
                social_seq.append(grid)
 
            social_seq = np.array(social_seq)   # (OBS_LEN, 16)
 
            obs_windows.append(disp[:OBS_LEN])          # (OBS_LEN, 2)
            target_windows.append(disp[OBS_LEN:])        # (PRED_LEN, 2)
            social_windows.append(social_seq)            # (OBS_LEN, 16)
 
    return (np.array(obs_windows),
            np.array(target_windows),
            np.array(social_windows))



def ade(pred, gt):
    pred_pos = np.cumsum(pred, axis=0)
    gt_pos   = np.cumsum(gt,   axis=0)
    return np.sqrt(((pred_pos - gt_pos) ** 2).sum(-1)).mean()
 
 
def fde(pred, gt):
    pred_pos = np.cumsum(pred, axis=0)
    gt_pos   = np.cumsum(gt,   axis=0)
    return np.sqrt(((pred_pos[-1] - gt_pos[-1]) ** 2).sum())

def train(obs_np, target_np, social_np, epochs=100, lr=5e-4, batch=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    obs_t    = torch.tensor(obs_np,    dtype=torch.float32).to(device)
    target_t = torch.tensor(target_np, dtype=torch.float32).to(device)
    social_t = torch.tensor(social_np, dtype=torch.float32).to(device)
 
    dataset = torch.utils.data.TensorDataset(obs_t, social_t, target_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
 
    model     = SocialLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
 
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for obs_b, soc_b, tgt_b in loader:
            pred = model(obs_b, soc_b)      # (B, pred_len, 2)
            loss = criterion(pred, tgt_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
 
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader)}")
 
        
 
    return model


from sklearn.model_selection import train_test_split
(obs_train, obs_test,
     tgt_train, tgt_test,
     soc_train, soc_test) = train_test_split(
        obs_np, target_np, social_np,
        test_size=0.2,      # 80% train, 20% test
        random_state=42,
    )


EPOCHS = 100

    # ── Train ───────────────────────────────────────────────────
model = train(obs_train, tgt_train, soc_train, epochs=100)

    # ── Evaluate on test set ────────────────────────────────────
model.eval()
device = next(model.parameters()).device

with torch.no_grad():
    pred = model(
        torch.tensor(obs_test,dtype=torch.float32 ,device=device),

        torch.tensor(soc_test,dtype=torch.float32 ,device=device),
        ).cpu().numpy()

    ades = [ade(pred[i], tgt_test[i]) for i in range(len(pred))]
    fdes = [fde(pred[i], tgt_test[i]) for i in range(len(pred))]

    print(sum(ades)/len(ades))
    print(sum(fdes)/len(fdes))
