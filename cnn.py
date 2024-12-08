import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import wfdb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Custom Dataset
class ECGDataset(Dataset):
    def __init__(self, data, seq_len=360, mask_ratio=0.5):
        self.data = data
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        mask = np.random.rand(len(seq)) < self.mask_ratio
        masked_seq = seq.copy()
        masked_seq[mask] = 0  # Masked values
        mask = torch.tensor(mask, dtype=torch.bool)
        return torch.tensor(masked_seq, dtype=torch.float32), torch.tensor(seq, dtype=torch.float32), mask


# Masked Autoencoder
class MAE1D(nn.Module):
    def __init__(self, seq_len, embed_dim=64, hidden_dim=128):
        super(MAE1D, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, mask):
        x = x.unsqueeze(1)  # Add channel dimension
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        x_reconstructed = decoded.squeeze(1)
        x_reconstructed[mask] = 0
        return x_reconstructed


# Data preparation
def prepare_data(data, seq_len=360, mask_ratio=0.4, batch_size=32):
    data = data.iloc[:, 0]
    data = data.to_numpy()
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    dataset = ECGDataset(data, seq_len, mask_ratio)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Training
def train_model_with_plot(model, train_loader, val_loader, epochs=50, lr=0.01, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for masked_seq, original_seq, mask in train_loader:
            masked_seq, original_seq, mask = masked_seq.to(device), original_seq.to(device), mask.to(device)

            optimizer.zero_grad()
            reconstructed = model(masked_seq, mask)
            loss = criterion(reconstructed, original_seq)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for masked_seq, original_seq, mask in val_loader:
                masked_seq, original_seq, mask = masked_seq.to(device), original_seq.to(device), mask.to(device)
                reconstructed = model(masked_seq, mask)
                loss = criterion(reconstructed, original_seq)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    # Visualize predictions vs. actual data
    plot_predictions_with_residuals(model, val_loader, device)
    plot_loss_curves(train_losses, val_losses)


def plot_predictions_with_residuals(model, data_loader, device, num_samples=5):
    model.eval()
    all_actual, all_predicted, all_masks = [], [], []
    
    with torch.no_grad():
        for masked_seq, original_seq, mask in data_loader:
            masked_seq, original_seq = masked_seq.to(device), original_seq.to(device)
            reconstructed = model(masked_seq, mask)
            all_actual.extend(original_seq.cpu().numpy())  # original (unmasked) ECG
            all_predicted.extend(reconstructed.cpu().numpy())  # predictions
            all_masks.extend(mask.cpu().numpy()) # Masked regions
            break  # Take only one batch for plotting
    
    # Flatten the lists for metric calculation
    all_actual_flat = np.concatenate(all_actual)
    all_predicted_flat = np.concatenate(all_predicted)
    
    # Calculate metrics
    mse = mean_squared_error(all_actual_flat, all_predicted_flat)
    mae = mean_absolute_error(all_actual_flat, all_predicted_flat)
    r2 = r2_score(all_actual_flat, all_predicted_flat)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Visualize predictions vs. actual data with residuals
    plt.figure(figsize=(15, num_samples * 5))
    for i in range(min(num_samples, len(all_actual))):
        original = all_actual[i]
        predicted = all_predicted[i]
        mask = all_masks[i]

        # Plot predictions vs actual data
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.plot(original, label="Original", color="blue", alpha=0.7)
        plt.plot(predicted, label="Predicted", color="orange", alpha=0.7)

        # Highlight masked regions
        masked_indices = np.where(mask)[0]
        plt.scatter(masked_indices, original[masked_indices], color="red", label="Masked Regions", alpha=0.7)

        plt.title(f"Sample {i + 1} - Predictions")
        plt.legend()

        # Compute and plot residuals
        residuals = np.array(original) - np.array(predicted)
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.plot(residuals, label="Residuals", color="green")
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"Sample {i + 1} - Residuals")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid()
    plt.show()

record_name = '100'
wfdb.dl_database('mitdb', './', records=[record_name])
record = wfdb.rdrecord(record_name)
signal_data = record.p_signal
data = pd.DataFrame(signal_data, columns=record.sig_name)
data.to_csv(f'{record_name}.csv', index=False)

# Instantiate and train the model
seq_len = 360
train_loader, val_loader = prepare_data(data, seq_len=seq_len)
mae_model = MAE1D(seq_len=seq_len)
train_model_with_plot(mae_model, train_loader, val_loader, epochs=20, device='mps' if torch.backends.mps.is_available() else 'cpu')