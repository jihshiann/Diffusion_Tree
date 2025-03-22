import os
import re
import math
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional

logging.basicConfig(level=logging.INFO)
TIME_EMB_DIM = 128

def parse_lat_lon(column_name: str) -> tuple[float, float]:
    match = re.search(r'\(([\d.-]+),\s*([\d.-]+)\)', column_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"欄位名稱格式無效：{column_name}")

class PeopleFlowDatasetCondition(Dataset):
    def __init__(self, csv_path: str, H: int, W: int, condition_length: int, prediction_length: int, transform: Optional[callable] = None, normalize: bool = True, debug: bool = False):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 檔案未找到：{csv_path}")
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.condition_length = condition_length
        self.prediction_length = prediction_length
        self.total_length = condition_length + prediction_length
        self.normalize = normalize
        self.H, self.W = H, W
        flow_columns = [c for c in self.df.columns if '(' in c and ')' in c]
        column_info = [(col, *parse_lat_lon(col)) for col in flow_columns]
        coords = np.array([(lon, lat) for _, lon, lat in column_info])
        mean_lon, mean_lat = np.mean(coords, axis=0)
        distances_to_center = np.sqrt((coords[:, 0] - mean_lon)**2 + (coords[:, 1] - mean_lat)**2)
        central_idx = np.argmin(distances_to_center)
        central_coord = coords[central_idx]
        grid_size = 21
        grid = np.full((grid_size, grid_size), -1, dtype=int)
        central_row, central_col = 10, 10
        grid[central_row, central_col] = central_idx
        lon_diffs = np.diff(np.sort(coords[:, 0]))
        lat_diffs = np.diff(np.sort(coords[:, 1]))
        lon_step = np.median(lon_diffs[lon_diffs > 0]) if len(lon_diffs) > 0 else 0.005
        lat_step = np.median(lat_diffs[lat_diffs > 0]) if len(lat_diffs) > 0 else 0.005
        available_indices = list(range(len(coords)))
        available_indices.remove(central_idx)
        grid_positions = []
        for k in range(11):
            for r in range(max(0, 10 - k), min(21, 10 + k + 1)):
                for c in range(max(0, 10 - k), min(21, 10 + k + 1)):
                    if max(abs(r - 10), abs(c - 10)) == k:
                        grid_positions.append((r, c))
        for r, c in grid_positions:
            if grid[r, c] != -1:
                continue
            target_lon = central_coord[0] + (c - central_col) * lon_step
            target_lat = central_coord[1] - (r - central_row) * lat_step
            lon_constraint = None
            if c < central_col:
                lon_constraint = lambda x: x < central_coord[0]
            elif c > central_col:
                lon_constraint = lambda x: x > central_coord[0]
            lat_constraint = None
            if r < central_row:
                lat_constraint = lambda x: x > central_coord[1]
            elif r > central_row:
                lat_constraint = lambda x: x < central_coord[1]
            filtered_indices = [idx for idx in available_indices if (lon_constraint is None or lon_constraint(coords[idx][0])) and (lat_constraint is None or lat_constraint(coords[idx][1]))]
            if filtered_indices:
                distances = np.sqrt((coords[filtered_indices, 0] - target_lon)**2 + (coords[filtered_indices, 1] - target_lat)**2)
                closest_idx = filtered_indices[np.argmin(distances)]
            else:
                distances = np.sqrt((coords[available_indices, 0] - target_lon)**2 + (coords[available_indices, 1] - target_lat)**2)
                closest_idx = available_indices[np.argmin(distances)]
            grid[r, c] = closest_idx
            available_indices.remove(closest_idx)
        if len(grid[grid != -1]) != grid_size * grid_size:
            raise ValueError(f"網格未填滿：選取 {len(grid[grid != -1])} 個，需 {grid_size * grid_size} 個")
        sorted_indices = grid.flatten()
        self.sorted_flow_columns = [column_info[idx][0] for idx in sorted_indices]
        if debug:
            self._plot_grid(save_path=r"C:\thesis\code\result_ddpm\plot_grid.png")
        flow_values = self.df[self.sorted_flow_columns].values.reshape(-1, H, W).astype(np.float32)
        self.data = torch.from_numpy(flow_values)
        if normalize:
            self.mean_val = self.data.mean()
            self.std_val = self.data.std() + 1e-5
            self.data = (self.data - self.mean_val) / self.std_val
        self.max_index = self.data.shape[0] - self.total_length + 1

    def _plot_grid(self, save_path: str):
        locations = [parse_lat_lon(col) for col in self.sorted_flow_columns]
        longitudes, latitudes = zip(*locations)
        plt.figure(figsize=(12, 12))
        plt.scatter(longitudes, latitudes, c='blue', marker='o', label='Grid Points')
        for i in range(self.H):
            for j in range(self.W):
                idx = i * self.W + j
                plt.text(longitudes[idx], latitudes[idx], f'[{i},{j}]', fontsize=6, ha='right')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Grid Arrangement")
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def __len__(self) -> int:
        return self.max_index

    def __getitem__(self, idx):
        cond_seq = self.data[idx:idx + self.condition_length]
        target_seq = self.data[idx + self.condition_length:idx + self.total_length]
        model_input = torch.cat([cond_seq, target_seq], dim=0).unsqueeze(0)
        return model_input, target_seq.unsqueeze(0)

def collate_fn(batch):
    conds, targets = zip(*batch)
    return torch.stack(conds), torch.stack(targets)

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, dropout_rate=0.0):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=(2, 2, 2), stride=(1, 2, 2), output_padding=(0, 1, 1))
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2), output_padding=(1, 0, 0))
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)
        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.time_proj = nn.Sequential(nn.Linear(time_emb_dim, base_channels * 8), nn.SiLU())
        self.x_full_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x_t, x_full, t_emb):
        x_full_conv = self.x_full_conv(x_full)
        x_input = x_t + x_full_conv
        e1 = self.enc1(x_input)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        p4 = self.pool4(e4)
        t_emb = self.time_proj(t_emb)[:, :, None, None, None]
        b = self.bottleneck(p4 + t_emb)
        b = self.dropout(b)
        d4 = self.up4(b)
        if d4.shape[-3:] != e4.shape[-3:]:
            d4 = F.interpolate(d4, size=e4.shape[-3:], mode='trilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        if d3.shape[-3:] != e3.shape[-3:]:
            d3 = F.interpolate(d3, size=e3.shape[-3:], mode='trilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = F.interpolate(d2, size=e2.shape[-3:], mode='trilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = F.interpolate(d1, size=e1.shape[-3:], mode='trilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out_conv(d1)
        return out[:, :, :1, :, :]

class DDPM3D(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = 'cuda'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.half_dim = TIME_EMB_DIM // 2
        self.freq_factor = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -(math.log(10000.0) / (self.half_dim - 1))).to(device)

    def get_time_embedding(self, t):
        t = t.float()
        emb = t[:, None] * self.freq_factor.to(t.device)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def get_condition_embedding(self, cond):
        return torch.zeros(cond.shape[0], TIME_EMB_DIM, device=cond.device)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def p_losses(self, cond, target, t):
        x_full = torch.cat([target, cond[:, :, 1:]], dim=2)
        noise = torch.randn_like(target)
        x_noisy_target = self.q_sample(target, t, noise=noise)
        x_t = torch.cat([x_noisy_target, cond[:, :, 1:]], dim=2)
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(cond)
        combined_emb = time_emb + cond_emb
        pred_noise = self.model(x_t, x_full, combined_emb)
        pred_noise_target = pred_noise[:, :, :1, :, :]
        return F.mse_loss(pred_noise_target, noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        if x_t.dim() == 4:
            x_t = x_t.unsqueeze(1)
        if cond.dim() == 4:
            cond = cond.unsqueeze(1)
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        time_emb = self.get_time_embedding(t).to(self.device)
        cond_emb = self.get_condition_embedding(cond)
        combined_emb = time_emb + cond_emb
        x_t_full = torch.cat([x_t, cond[:, :, 1:]], dim=2)
        x_full = cond
        eps_theta = self.model(x_t_full, x_full, combined_emb)
        eps_theta_target = eps_theta[:, :, :1, :, :]
        x_t_minus_1 = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * eps_theta_target)
        mask = (t > 0).float().view(-1, 1, 1, 1, 1)
        sigma_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        return x_t_minus_1 + mask * sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        if cond.dim() == 4:
            cond = cond.unsqueeze(1)
        if len(shape) == 4:
            batch_size = shape[0]
            shape = (batch_size, 1, shape[1], shape[2], shape[3])
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        return x

def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})', cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def visualize_predictions(cond, generated, target, sample_idx: int = 0, save_dir: str = r"C:\thesis\code\result_ddpm"):
    os.makedirs(save_dir, exist_ok=True)
    pred_length = generated.shape[2]
    for t in range(pred_length):
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(generated[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Generated (t={t})')
        plt.subplot(1, 4, 2)
        plt.imshow(target[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'True (t={t})')
        error_sq = (generated[sample_idx, 0, t].cpu().numpy() - target[sample_idx, 0, t].cpu().numpy()) ** 2
        plt.subplot(1, 4, 3)
        plt.imshow(error_sq, cmap='hot')
        plt.colorbar()
        plt.title(f'MSE (t={t})')
        error_abs = np.abs(generated[sample_idx, 0, t].cpu().numpy() - target[sample_idx, 0, t].cpu().numpy())
        plt.subplot(1, 4, 4)
        plt.imshow(error_abs, cmap='hot')
        plt.colorbar()
        plt.title(f'MAE (t={t})')
        plt.suptitle(f'Sample {sample_idx} - Time Step {t}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'prediction_sample{sample_idx}_t{t}.png'), dpi=300)
        plt.close()

def plot_grid_with_error(sorted_flow_columns: list, H: int, W: int, error_matrix: np.ndarray, save_dir: str = r"C:\\thesis\\code\\result_ddpm"):
    os.makedirs(save_dir, exist_ok=True)
    locations = [parse_lat_lon(col) for col in sorted_flow_columns]
    longitudes, latitudes = zip(*locations)
    mse_matrix = error_matrix
    mae_matrix = np.abs(np.sqrt(error_matrix))
    if mae_matrix.ndim > 2:
        mae_matrix = np.mean(mae_matrix, axis=(0, 1, 2))
    else:
        mae_matrix = mae_matrix
    plt.figure(figsize=(12, 12))
    orig_cmap = plt.get_cmap('OrRd')
    trunc_cmap = truncate_colormap(orig_cmap, 0.3, 1.0)
    scatter = plt.scatter(longitudes, latitudes, c=mse_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MSE')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MSE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mse.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(longitudes, latitudes, c=mae_matrix.flatten(), cmap=trunc_cmap, marker='o')
    plt.colorbar(scatter, label='MAE')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Grid with MAE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_grid_with_error_mae.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    table_data = {
        'Grid Index': [f'[{i},{j}]' for i in range(H) for j in range(W)],
        'Longitude': longitudes,
        'Latitude': latitudes,
        'MSE': mse_matrix.flatten(),
        'MAE': mae_matrix.flatten()
    }
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'mse_mae_per_coordinate.csv'), index=False)
    df.to_excel(os.path.join(save_dir, 'mse_mae_per_coordinate.xlsx'), index=False)

def train_ddpm(diffusion: DDPM3D, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 20, lr: float = 1e-4, device: str = 'cuda', patience: int = 3, weight_decay: float = 1e-6, save_dir: str = r"C:\thesis\code\result_ddpm") -> DDPM3D:
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    diffusion.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
        diffusion.train()
        total_train_loss = 0
        for cond, target in train_loader:
            cond, target = cond.to(device), target.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
            loss = diffusion.p_losses(cond, target, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for cond, target in val_loader:
                cond, target = cond.to(device), target.to(device)
                t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
                loss = diffusion.p_losses(cond, target, t)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(diffusion.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return diffusion

@torch.no_grad()
def evaluate_model(diffusion: DDPM3D, dataset: Dataset, device: str = 'cuda', max_samples: int = 100, save_dir: str = r"C:\\thesis\\code\\result_ddpm", sample_idx: int = 0) -> dict:
    diffusion.eval()
    metrics = {'mse': 0.0, 'mae': 0.0}
    N = min(len(dataset), max_samples)
    sample_indices = random.sample(range(len(dataset)), N)
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset
    H, W = base_dataset.H, base_dataset.W
    cond_channels = dataset[0][0].shape[1]
    pred_length = base_dataset.prediction_length
    mean_val = base_dataset.mean_val.to(device)
    std_val = base_dataset.std_val.to(device)
    generated_batch = torch.zeros(N, 1, pred_length, H, W, device=device)
    cond_batch = torch.zeros(N, 1, cond_channels, H, W, device=device)
    target_batch = torch.zeros(N, 1, pred_length, H, W, device=device)
    for i, idx in enumerate(sample_indices):
        cond, target = dataset[idx]
        cond, target = cond.to(device), target.to(device)
        target = target.unsqueeze(2)
        x_recon = diffusion.p_sample_loop(target.shape, cond)
        x_recon_original = x_recon * std_val + mean_val
        target_original = target * std_val + mean_val
        cond_original = cond * std_val + mean_val
        assert cond_original.shape == (1, cond_channels, H, W), f"cond_original shape mismatch: {cond_original.shape}"
        assert x_recon_original.shape == (1, 1, pred_length, H, W), f"x_recon_original shape mismatch: {x_recon_original.shape}"
        assert target_original.shape == (1, 1, pred_length, H, W), f"target_original shape mismatch: {target_original.shape}"
        generated_batch[i] = x_recon_original
        cond_batch[i] = cond_original
        target_batch[i] = target_original
        metrics['mse'] += F.mse_loss(x_recon_original, target_original).item()
        metrics['mae'] += F.l1_loss(x_recon_original, target_original).item()
    metrics['mse'] /= N
    metrics['mae'] /= N
    os.makedirs(save_dir, exist_ok=True)
    for t in range(pred_length):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(generated_batch[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Generated (t={t})')
        plt.subplot(1, 3, 2)
        plt.imshow(target_batch[sample_idx, 0, t].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'True (t={t})')
        plt.subplot(1, 3, 3)
        error = np.abs(generated_batch[sample_idx, 0, t].cpu().numpy() - target_batch[sample_idx, 0, t].cpu().numpy())
        plt.imshow(error, cmap='hot')
        plt.colorbar()
        plt.title(f'Error (t={t})')
        plt.suptitle(f'Sample {sample_idx} - Time Step {t}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'prediction_sample{sample_idx}_t{t}.png'), dpi=300)
        plt.close()
    visualize_predictions(cond_batch, generated_batch, target_batch, sample_idx=sample_idx, save_dir=save_dir)
    error_matrix = (generated_batch - target_batch) ** 2
    mse_matrix = torch.mean(error_matrix, dim=(0, 2)).cpu().numpy()[0]
    plot_grid_with_error(base_dataset.sorted_flow_columns, H, W, mse_matrix, save_dir)
    with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Evaluation Metrics (computed on {N} samples):\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Reconstruction MSE: {metrics['mse']:.6f}\n")
        f.write(f"Reconstruction MAE: {metrics['mae']:.6f}\n")
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump({
            "mse": metrics['mse'], "mae": metrics['mae'],
            "sample_size": N, "timestamp": pd.Timestamp.now().isoformat()
        }, f, indent=4)
    return metrics

if __name__ == "__main__":
    H, W = 21, 21
    condition_length, prediction_length = 8, 1
    batch_size, epochs, lr, timesteps, patience = 100, 1000, 0.0001, 1000, 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = r"C:\thesis\code\result_ddpm"
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = PeopleFlowDatasetCondition(csv_path=r"C:\thesis\code\Taipei_CF\all_merged.csv", H=H, W=W, condition_length=condition_length, prediction_length=prediction_length, normalize=True, debug=True)
    train_end = int(0.7 * len(dataset))
    val_end = int(0.85 * len(dataset))
    train_dataset = Subset(dataset, range(0, train_end))
    val_dataset = Subset(dataset, range(train_end, val_end))
    test_dataset = Subset(dataset, range(val_end, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    unet = UNet3D(in_channels=1, base_channels=64, time_emb_dim=TIME_EMB_DIM, dropout_rate=0.0)
    diffusion = DDPM3D(model=unet, timesteps=timesteps, beta_start=1e-4, beta_end=0.02, device=device)
    trained_diffusion = train_ddpm(diffusion, train_loader, val_loader, epochs=epochs, lr=lr, device=device, patience=patience, save_dir=save_dir)
    metrics = evaluate_model(trained_diffusion, val_dataset, device=device, max_samples=100, save_dir=save_dir)
    logging.info(f"Reconstruction MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Evaluation Metrics (computed on 2 samples):\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Reconstruction MSE: {metrics['mse']:.6f}\n")
        f.write(f"Reconstruction MAE: {metrics['mae']:.6f}\n")
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump({
            "mse": metrics['mse'], "mae": metrics['mae'],
            "sample_size": 2, "timestamp": pd.Timestamp.now().isoformat()
        }, f, indent=4)
