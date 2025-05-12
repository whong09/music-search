import torch
import torch.nn as nn

class MidiAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (1, 128, 512) → (16, 64, 256)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # → (32, 32, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # → (64, 16, 64)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 64)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # To keep output in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

from torch.utils.data import DataLoader
import torch.optim as optim

def train_autoencoder(model, dataset, epochs=10, batch_size=32, lr=1e-3, 
                      num_workers=0, checkpoint_path="ae_checkpoint.pt",
                      start_epoch=0, start_batch=0, optimizer=None):
    from tqdm import tqdm

    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    batch_idx = 0
    epoch = start_epoch

    dataloader = DataLoader(dataset, **kwargs)
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    def save_checkpoint(epoch, batch):
        state = {
            'epoch': epoch,
            'batch': batch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)
        print(f"\nCheckpoint saved to {checkpoint_path} (epoch {epoch})")

    try:
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            skip = start_batch if epoch == start_epoch else 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (batch_x, *_ ) in enumerate(progress):
                if batch_idx < skip:
                    continue
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                x_recon, _ = model(batch_x)
                loss = loss_fn(x_recon, batch_x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
            save_checkpoint(epoch + 1, 0)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        save_checkpoint(epoch, batch_idx+1)
        print("Checkpoint saved before exiting.")

def embed_dataset(model, dataset, save_path='vectors.npz'):
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    vectors = []
    metadata = []

    for i in tqdm(range(len(dataset)), desc="Embedding"):
        x, song_id, chunk_idx = dataset[i]
        with torch.no_grad():
            _, z = model(x.unsqueeze(0).to(device))
        vectors.append(z.cpu().numpy())
        metadata.append((song_id, chunk_idx))

    vectors = np.vstack(vectors)
    np.savez_compressed(save_path, vectors=vectors, metadata=np.array(metadata, dtype=object))
    print(f"Embedding complete. {len(vectors)} vectors saved to {save_path}")

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChunkedPianoRollDataset(Dataset):
    def __init__(self, root_dir):
        self.entries = []  # list of (path, chunk_idx)
        print("Indexing chunks...")
        for i, f in enumerate(sorted(os.listdir(root_dir))):
            if f.endswith('.npz'):
                path = os.path.join(root_dir, f)
                try:
                    with np.load(path) as data:
                        num_chunks = data['chunks'].shape[0]
                        for j in range(num_chunks):
                            self.entries.append((path, j))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            if i % 1000 == 0:
                print(f"  Indexed {i} files...")

        print(f"Total chunks: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, chunk_idx = self.entries[idx]
        song_id = os.path.basename(path).replace('.npz', '')
        with np.load(path) as data:
            chunk = data['chunks'][chunk_idx]
        chunk = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
        return chunk, song_id, chunk_idx

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--checkpoint-path", type=str, default="ae_checkpoint.pt")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ChunkedPianoRollDataset(args.data_dir)
    if args.limit is not None:
        dataset.entries = dataset.entries[:args.limit]
        print(f"Using limited dataset: {len(dataset)} chunks")

    model = MidiAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    start_batch = 0

    if os.path.exists(args.checkpoint_path) and args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint.get("batch", 0)
        print(f"Resumed at epoch {start_epoch}, batch {start_batch}")

    train_autoencoder(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint_path,
        start_epoch=start_epoch,
        start_batch=start_batch,
        optimizer=optimizer
    )
