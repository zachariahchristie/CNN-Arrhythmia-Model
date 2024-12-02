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
    def __init__(self, data, seq_len=300, mask_len=100):
        self.data = (data - np.mean(data)) / np.std(data)
        self.seq_len = seq_len
        self.mask_len = mask_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        
        # Mask the middle portion
        mask_start = (self.seq_len - self.mask_len) // 2
        mask_end = mask_start + self.mask_len
        mask = np.zeros(self.seq_len)
        mask[mask_start:mask_end] = 1  # Mask the middle section

        masked_seq = seq.copy()
        masked_seq[mask_start:mask_end] = 0  # Set the middle section to 0

        return (
            torch.tensor(masked_seq, dtype=torch.float32),
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        ) 


# Masked Autoencoder
class MAE1D(nn.Module):
    def __init__(self, seq_len=300, embed_dim=64, hidden_dim=128):
        super(MAE1D, self).__init__()
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

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for 1D Conv
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)
    
def masked_loss(predicted, original, mask):
    return torch.sum((predicted-original)**2 * mask) / torch.sum(mask)

# Data preparation
def prepare_data(data, seq_len=300, batch_size=32):
    data = data.iloc[:, 0]
    data = data.to_numpy()
    mask_len = 100  # Fixed mask length
    dataset = ECGDataset(data, seq_len, mask_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Training
def train_model_with_plot(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for masked_seq, original_seq, mask in train_loader:
            masked_seq, original_seq, mask = masked_seq.to(device), original_seq.to(device), mask.to(device)
            optimizer.zero_grad()
            reconstructed = model(masked_seq)
            loss = masked_loss(reconstructed, original_seq, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for masked_seq, original_seq, mask in val_loader:
                masked_seq, original_seq, mask = masked_seq.to(device), original_seq.to(device), mask.to(device)
                reconstructed = model(masked_seq)
                loss = masked_loss(reconstructed, original_seq, mask)
                val_loss += loss.item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    # Visualize predictions vs. actual data
    plot_predictions_with_residuals(model, val_loader, device)


def plot_predictions_with_residuals(model, data_loader, device, num_samples=5):
    model.eval()
    all_actual, all_predicted = [], []
    
    with torch.no_grad():
        for masked_seq, original_seq, mask in data_loader:
            masked_seq, original_seq = masked_seq.to(device), original_seq.to(device)
            reconstructed = model(masked_seq)
            
            # Append unmasked original data and predicted data
            all_actual.extend(original_seq.cpu().numpy())  # Store the unmasked ECG (original)
            all_predicted.extend(reconstructed.cpu().numpy())  # Store the model's predictions
            
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
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.plot(all_actual[i], label="Actual", color="blue")
        plt.plot(all_predicted[i], label="Predicted", color="orange")
        plt.legend()
        plt.subplot(num_samples, 2, i * 2 + 2)
        residuals = np.array(all_actual[i]) - np.array(all_predicted[i])
        plt.plot(residuals, label="Residuals", color="green")
        plt.axhline(0, color="black", linestyle="--")
        plt.legend()

    plt.tight_layout()
    plt.show()

record_name = '100'
wfdb.dl_database('mitdb', './', records=[record_name])
record = wfdb.rdrecord(record_name)
signal_data = record.p_signal
data = pd.DataFrame(signal_data, columns=record.sig_name)

# Instantiate and train the model
seq_len = 360
train_loader, val_loader = prepare_data(data, seq_len=seq_len)
mae_model = MAE1D(seq_len=seq_len)
train_model_with_plot(mae_model, train_loader, val_loader, epochs=20, device='mps' if torch.backends.mps.is_available() else 'cpu')