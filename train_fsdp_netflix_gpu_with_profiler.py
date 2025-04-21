
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import profile, record_function, ProfilerActivity

import pandas as pd
import dask.dataframe as dd
import time

os.environ["MKL_THREADING_LAYER"] = "GNU"

# ------------------------
# Model Definition
# ------------------------
class NetflixNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# ------------------------
# Setup/Teardown
# ------------------------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ------------------------
# Train Function (FSDP)
# ------------------------
def train_fsdp(rank, world_size, X, y):
    setup(rank, world_size)
    torch.manual_seed(42)
    device = torch.device(f"cuda:{rank}")

    dataset = TensorDataset(torch.tensor(X).to(device), torch.tensor(y).unsqueeze(1).to(device))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = NetflixNet(X.shape[1]).to(device)
    fsdp_model = FSDP(model)

    optimizer = optim.Adam(fsdp_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    start_train = time.time()

    for epoch in range(5):
        epoch_start = time.time()
        fsdp_model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if epoch == 0 and batch_idx == 0:
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("model_training"):
                        output = fsdp_model(xb)
                        loss = criterion(output, yb)
                        loss.backward()
                        optimizer.step()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            else:
                output = fsdp_model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pred_labels = (output > 0.5).float()
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)

        epoch_end = time.time()
        accuracy = correct / total
        print(f"[Rank {rank}] Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_end - epoch_start:.2f}s")

    end_train = time.time()
    if rank == 0:
        print(f"🕒 Total training time on Rank 0: {end_train - start_train:.2f} seconds")

    cleanup()

# ------------------------
# Main Launcher
# ------------------------
def run_fsdp_training(X, y, world_size=1):
    mp.spawn(train_fsdp, args=(world_size, X, y), nprocs=world_size, join=True)

# ------------------------
# Main Block
# ------------------------
if __name__ == "__main__":
    df_dask = dd.read_csv("CLEANED_NETFLIX_REVIEWS.csv")
    df = df_dask.compute()
    print(f"📊 Loaded full dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    df = df.dropna(subset=['review_rating'])
    df['target'] = df['review_rating'].apply(lambda x: 1 if x == 5 else 0)

    X = df.drop(columns=[
        'review_id', 'review_text', 'author_name',
        'author_app_version', 'review_timestamp',
        'review_rating', 'target'
    ], errors='ignore').fillna(0).astype("float32").values

    y = df['target'].astype("float32").values

    run_fsdp_training(X, y, world_size=1)
