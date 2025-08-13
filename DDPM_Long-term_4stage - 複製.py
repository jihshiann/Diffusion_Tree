
    BASEMODEL = 1
    STAGE2 = 2
    STAGE3 = 3
    STAGE4 = 4
    BASELINE_EVAL = 5

#%%
# ==============================================================================
# UNet3D, DDPM3D
# ==============================================================================

# UNet3D 建構模組及 UNet3D 類別的預留位置
class SinusoidalTimeEmbedding(nn.Module):
    """正弦時間嵌入"""
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device; half_dim = self.dim // 2; emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]; emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DoubleConv3D(nn.Module):
    """(卷積3D -> BN -> SiLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, kernel_size: int = 3, padding: int = 1):
        super().__init__(); mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm3d(mid_channels), nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False), nn.BatchNorm3d(out_channels), nn.SiLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.double_conv(x)

class Down3D(nn.Module):
    """下採樣模組 (MaxPool3D -> DoubleConv3D)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)), # 深度維度不壓縮
            DoubleConv3D(in_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.maxpool_conv(x)

class Up3D(nn.Module):
    """上採樣模組"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__(); self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True) # 深度維度不放大
            self.conv = DoubleConv3D(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2)) # 深度維度不放大
            self.conv = DoubleConv3D(in_channels, out_channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor: # x1 是上採樣的張量, x2 是殘差連接的張量
        x1 = self.up(x1)
        # 輸入大小: C D H W
        diffY = x2.size()[3] - x1.size()[3] # H
        diffX = x2.size()[4] - x1.size()[4] # W
        # 深度維度 (dim 2) 不需要填充
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, # W
                        diffY // 2, diffY - diffY // 2, # H
                        0, 0])                          # D
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    """輸出卷積層 (1x1x1 Conv3D)"""
    def __init__(self, in_channels: int, out_channels: int): super().__init__(); self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net 模型，帶有正確的時間嵌入投影"""
    def __init__(self, input_image_channels: int, base_channels: int = 64, time_emb_dim: int = 256,
                 condition_encode_dim: Optional[int] = None, bilinear_upsample: bool = True, dropout_rate: float = 0.05):
        super().__init__()
        self.input_image_channels = input_image_channels
        self.condition_encode_dim = condition_encode_dim or 0

        # 共享的時間嵌入 MLP (輸出維度是 time_emb_dim)
        self.shared_time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        actual_in_channels = self.input_image_channels + self.condition_encode_dim
        
        # --- U-Net 結構 ---
        self.inc = DoubleConv3D(actual_in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear_upsample else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor) # Bottleneck 層的前一層
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, bilinear_upsample)
        self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, bilinear_upsample)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, bilinear_upsample)
        self.up4 = Up3D(base_channels * 2, base_channels, bilinear_upsample)
        self.outc = OutConv3D(base_channels, self.input_image_channels)

        # --- 為每個需要添加時間嵌入的層級定義線性投影層 ---
        self.time_proj_inc = nn.Linear(time_emb_dim, base_channels)
        self.time_proj_down1 = nn.Linear(time_emb_dim, base_channels * 2)
        self.time_proj_down2 = nn.Linear(time_emb_dim, base_channels * 4)
        self.time_proj_down3 = nn.Linear(time_emb_dim, base_channels * 8)
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, base_channels * 16 // factor) # 對應 down4 的輸出 (bottleneck)

        self.time_proj_up1 = nn.Linear(time_emb_dim, base_channels * 8 // factor)
        self.time_proj_up2 = nn.Linear(time_emb_dim, base_channels * 4 // factor)
        self.time_proj_up3 = nn.Linear(time_emb_dim, base_channels * 2 // factor)
        self.time_proj_up4 = nn.Linear(time_emb_dim, base_channels)

    def _add_time_embedding(self, x: torch.Tensor, t_emb_projected: torch.Tensor) -> torch.Tensor:
        # t_emb_projected 應該已經是 (N, C_feature_map) 的形狀
        t_emb_expanded = t_emb_projected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x + t_emb_expanded

    def forward(self, x_t: torch.Tensor, time_steps: torch.Tensor, processed_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 首先計算共享的時間嵌入 (N, time_emb_dim)
        shared_t_emb = self.shared_time_mlp(time_steps)

        if processed_condition is not None:
            if x_t.shape[2:] != processed_condition.shape[2:]: # 檢查 D, H, W 是否一致
                raise ValueError(f"x_t DHW {x_t.shape[2:]} != processed_condition DHW {processed_condition.shape[2:]}")
            x_input = torch.cat((x_t, processed_condition), dim=1) # 沿通道維度合併
        else:
            x_input = x_t

        x1 = self.inc(x_input)
        x1 = self._add_time_embedding(x1, self.time_proj_inc(shared_t_emb))

        x2 = self.down1(x1)
        x2 = self._add_time_embedding(x2, self.time_proj_down1(shared_t_emb))

        x3 = self.down2(x2)
        x3 = self._add_time_embedding(x3, self.time_proj_down2(shared_t_emb))

        x4 = self.down3(x3)
        x4 = self._add_time_embedding(x4, self.time_proj_down3(shared_t_emb))

        x5 = self.down4(x4) # Bottleneck 特徵
        x5 = self._add_time_embedding(x5, self.time_proj_bottleneck(shared_t_emb)) # 使用對應的投影
        x5 = self.dropout(x5)

        x = self.up1(x5, x4) # x4 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up1(shared_t_emb))

        x = self.up2(x, x3) # x3 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up2(shared_t_emb))

        x = self.up3(x, x2) # x2 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up3(shared_t_emb))

        x = self.up4(x, x1) # x1 是來自 encoder 的 skip connection
        x = self._add_time_embedding(x, self.time_proj_up4(shared_t_emb))
        
        return self.outc(x)
def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """線性 beta 排程"""
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM3D(nn.Module):
    def __init__(self,
                 unet_model: UNet3D,
                 timesteps: int,
                 image_size: Tuple[int, int, int], # (D, H, W)
                 image_channels: int,
                 condition_input_channels: int,
                 condition_encode_dim: int,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 device: str = "cuda"):
        super().__init__()
        self.model = unet_model
        self.timesteps = timesteps
        self.image_size_D, self.image_size_H, self.image_size_W = image_size
        self.image_channels = image_channels
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.condition_processor = nn.Sequential(
            nn.Conv3d(condition_input_channels, condition_encode_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim // 2), nn.SiLU(),
            nn.Conv3d(condition_encode_dim // 2, condition_encode_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(condition_encode_dim), nn.SiLU()
        ).to(device)
        self.logger.info(f"DDPM3D instance created. Condition processor expects {condition_input_channels} input channels.")

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        sact = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        soma_ct = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sact * x_start + soma_ct * noise

    def _prepare_original_conditional_input_grids(self,
                                            hour_scalars_batch: torch.Tensor,
                                            is_holiday_scalars_batch: torch.Tensor,
                                            ) -> torch.Tensor: # 輸出 (N, 2, D, H, W)
        batch_size = hour_scalars_batch.shape[0]
        if hour_scalars_batch.shape[0] != is_holiday_scalars_batch.shape[0]:
            self.logger.error(f"Batch size mismatch in _prepare_original_conditional_input_grids: hour_batch={hour_scalars_batch.shape[0]}, holiday_batch={is_holiday_scalars_batch.shape[0]}")
            raise ValueError("Batch sizes for hour and holiday scalars must match.")

        norm_hours = hour_scalars_batch.float().to(self.device) / 23.0
        holiday_values = is_holiday_scalars_batch.float().to(self.device)

        hour_grid_vals = norm_hours.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W)
        holiday_grid_vals = holiday_values.view(batch_size, 1, 1).expand(batch_size, self.image_size_H, self.image_size_W)

        hour_grids_t = hour_grid_vals.unsqueeze(1).unsqueeze(2)
        holiday_grids_t = holiday_grid_vals.unsqueeze(1).unsqueeze(2)

        if self.image_size_D != 1:
            hour_grids_t = hour_grids_t.repeat(1,1,self.image_size_D,1,1)
            holiday_grids_t = holiday_grids_t.repeat(1,1,self.image_size_D,1,1)

        if self.condition_processor[0].in_channels != 2:
             self.logger.warning(f"_prepare_original_conditional_input_grids: Condition processor input channels ({self.condition_processor[0].in_channels}) is not 2, but this method produces 2 channels.")

        final_stacked_grids = torch.cat((hour_grids_t, holiday_grids_t), dim=1)
        return final_stacked_grids.to(self.device)

    def _prepare_stage_condition_grids(self,
                                     condition_grid_1_batch: torch.Tensor,
                                     condition_grid_2_batch: torch.Tensor
                                     ) -> torch.Tensor:
        expected_single_grid_shape = (1, self.image_size_D, self.image_size_H, self.image_size_W)
        # 檢查第一個條件網格的通道數是否為1 (因為通常是單一來源的網格，如BM輸出)
        if condition_grid_1_batch.shape[1] != 1:
            self.logger.warning(f"Stage condition_grid_1_batch has {condition_grid_1_batch.shape[1]} channels, expected 1. Using as is.")

        # 檢查第二個條件網格的通道數是否為1 (因為通常是單一來源的網格，如新特徵網格)
        if condition_grid_2_batch.shape[1] != 1:
            self.logger.warning(f"Stage condition_grid_2_batch has {condition_grid_2_batch.shape[1]} channels, expected 1. Using as is.")

        # 確保空間維度 (D, H, W) 匹配
        if condition_grid_1_batch.shape[2:] != expected_single_grid_shape[1:] or \
           condition_grid_2_batch.shape[2:] != expected_single_grid_shape[1:]:
            self.logger.error(f"Stage condition input grid spatial dimensions (D,H,W) are incorrect or mismatched. "
                              f"Grid1 spatial: {condition_grid_1_batch.shape[2:]}, Grid2 spatial: {condition_grid_2_batch.shape[2:]}. "
                              f"Expected spatial: {expected_single_grid_shape[1:]}")

        
        if self.condition_processor[0].in_channels != 2:
             self.logger.warning(f"_prepare_stage_condition_grids: Condition processor input channels ({self.condition_processor[0].in_channels}) is not 2, but this method produces 2 channels by concatenating two 1-channel grids.")
        return torch.cat((condition_grid_1_batch, condition_grid_2_batch), dim=1)

    def p_losses(self, x_start_target_flow: torch.Tensor, t: torch.Tensor,
                 mode: ConditionMode, # 明確的模式參數
                 condition_args: Dict[str, Optional[torch.Tensor]], # 一個包含所有可能條件的字典
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:

        if noise is None: noise = torch.randn_like(x_start_target_flow)
        x_t_noisy_target = self.q_sample(x_start=x_start_target_flow, t=t, noise=noise)
        stacked_cond_grids: Optional[torch.Tensor] = None
        
        self.logger.debug(f"p_losses called with mode: {mode}, condition_args keys: {list(condition_args.keys())}")

        if mode == ConditionMode.BASEMODEL:
            hour_s = condition_args.get("hour_scalars_batch")
            is_hol_s = condition_args.get("is_holiday_scalars_batch")
            if hour_s is None or is_hol_s is None:
                raise ValueError("p_losses (Basemodel mode): Requires 'hour_scalars_batch' and 'is_holiday_scalars_batch' in condition_args.")
            # 檢查是否提供了其他不應存在的鍵 (可選，但更穩健)
            unexpected_keys = [k for k in condition_args if k not in ["hour_scalars_batch", "is_holiday_scalars_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Basemodel mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(hour_s, is_hol_s)
        
        elif mode == ConditionMode.STAGE2:
            bm_out = condition_args.get("basemodel_output_grid_batch")
            s2_new_feat = condition_args.get("stage2_new_condition_feature_grid_batch")
            if bm_out is None or s2_new_feat is None:
                raise ValueError("p_losses (Stage2 mode): Requires 'basemodel_output_grid_batch' and 'stage2_new_condition_feature_grid_batch' in condition_args.")
            unexpected_keys = [k for k in condition_args if k not in ["basemodel_output_grid_batch", "stage2_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Stage2 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(bm_out, s2_new_feat)

        elif mode == ConditionMode.STAGE3:
            s2_out = condition_args.get("stage2_output_grid_batch_for_s3")
            s3_new_feat = condition_args.get("stage3_new_condition_feature_grid_batch")
            if s2_out is None or s3_new_feat is None:
                raise ValueError("p_losses (Stage3 mode): Requires 'stage2_output_grid_batch_for_s3' and 'stage3_new_condition_feature_grid_batch' in condition_args.")
            unexpected_keys = [k for k in condition_args if k not in ["stage2_output_grid_batch_for_s3", "stage3_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Stage3 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s2_out, s3_new_feat)

        elif mode == ConditionMode.STAGE4: # 新增 Stage4 處理
            s3_out = condition_args.get("stage3_output_grid_batch_for_s4")
            s4_new_feat = condition_args.get("stage4_new_condition_feature_grid_batch")
            if s3_out is None or s4_new_feat is None:
                raise ValueError("p_losses (Stage4 mode): Requires 'stage3_output_grid_batch_for_s4' and 'stage4_new_condition_feature_grid_batch' in condition_args.")
            unexpected_keys = [k for k in condition_args if k not in ["stage3_output_grid_batch_for_s4", "stage4_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"p_losses (Stage4 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s3_out, s4_new_feat)
        else:
            raise ValueError(f"p_losses: Unsupported condition mode: {mode}")

        expected_cond_proc_input_channels = self.condition_processor[0].in_channels
        if stacked_cond_grids.shape[1] != expected_cond_proc_input_channels:
             raise ValueError(f"Prepared condition grids channel mismatch for p_losses. "
                              f"ConditionProcessor expected {expected_cond_proc_input_channels} channels, "
                              f"but got {stacked_cond_grids.shape[1]}.")
        stacked_cond_grids = stacked_cond_grids.to(self.device)
        processed_condition = self.condition_processor(stacked_cond_grids)
        predicted_noise = self.model(x_t_noisy_target, t, processed_condition)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, batch_size: int,
               mode: ConditionMode, # 明確的模式參數
               condition_args: Dict[str, Optional[torch.Tensor]] # 一個包含所有可能條件的字典
               ) -> torch.Tensor:

        img_shape = (batch_size, self.image_channels, self.image_size_D, self.image_size_H, self.image_size_W)
        img = torch.randn(img_shape, device=self.device)
        stacked_cond_grids: Optional[torch.Tensor] = None
        
        self.logger.debug(f"sample called with mode: {mode}, condition_args keys: {list(condition_args.keys())}")
        if mode == ConditionMode.BASELINE_EVAL:
                stacked_cond_grids = condition_args.get("direct_condition")
                if stacked_cond_grids is None:
                    raise ValueError("sample (BASELINE_EVAL mode): 需要在 condition_args 中提供 'direct_condition'。")
        elif mode == ConditionMode.BASEMODEL:
            hour_s = condition_args.get("hour_scalars_batch")
            is_hol_s = condition_args.get("is_holiday_scalars_batch")
            if hour_s is None or is_hol_s is None or hour_s.shape[0] != batch_size or is_hol_s.shape[0] != batch_size:
                raise ValueError("sample (Basemodel mode): Requires 'hour_scalars_batch' and 'is_holiday_scalars_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["hour_scalars_batch", "is_holiday_scalars_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Basemodel mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_original_conditional_input_grids(hour_s, is_hol_s).to(self.device)
        
        elif mode == ConditionMode.STAGE2:
            bm_out = condition_args.get("basemodel_output_grid_batch")
            s2_new_feat = condition_args.get("stage2_new_condition_feature_grid_batch")
            if bm_out is None or s2_new_feat is None or bm_out.shape[0] != batch_size or s2_new_feat.shape[0] != batch_size:
                raise ValueError("sample (Stage2 mode): Requires 'basemodel_output_grid_batch' and 'stage2_new_condition_feature_grid_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["basemodel_output_grid_batch", "stage2_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Stage2 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(bm_out, s2_new_feat)

        elif mode == ConditionMode.STAGE3:
            s2_out = condition_args.get("stage2_output_grid_batch_for_s3")
            s3_new_feat = condition_args.get("stage3_new_condition_feature_grid_batch")
            if s2_out is None or s3_new_feat is None or s2_out.shape[0] != batch_size or s3_new_feat.shape[0] != batch_size:
                raise ValueError("sample (Stage3 mode): Requires 'stage2_output_grid_batch_for_s3' and 'stage3_new_condition_feature_grid_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["stage2_output_grid_batch_for_s3", "stage3_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Stage3 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s2_out, s3_new_feat)

        elif mode == ConditionMode.STAGE4: # 新增 Stage4 處理
            s3_out = condition_args.get("stage3_output_grid_batch_for_s4")
            s4_new_feat = condition_args.get("stage4_new_condition_feature_grid_batch")
            if s3_out is None or s4_new_feat is None or s3_out.shape[0] != batch_size or s4_new_feat.shape[0] != batch_size:
                raise ValueError("sample (Stage4 mode): Requires 'stage3_output_grid_batch_for_s4' and 'stage4_new_condition_feature_grid_batch' matching batch_size.")
            unexpected_keys = [k for k in condition_args if k not in ["stage3_output_grid_batch_for_s4", "stage4_new_condition_feature_grid_batch"]]
            if unexpected_keys:
                self.logger.warning(f"sample (Stage4 mode): Unexpected keys in condition_args: {unexpected_keys}")
            stacked_cond_grids = self._prepare_stage_condition_grids(s3_out, s4_new_feat)
        else:
            raise ValueError(f"sample: Unsupported condition mode: {mode}")

        # 驗证準備好的條件網格形狀
        if stacked_cond_grids.shape[1] != self.condition_processor[0].in_channels:
             raise ValueError(f"Prepared condition grids channel mismatch for sampling. "
                              f"ConditionProcessor expected {self.condition_processor[0].in_channels} channels, "
                              f"but got {stacked_cond_grids.shape[1]}.")
        processed_conditions = self.condition_processor(stacked_cond_grids)

        for i in reversed(range(0, self.timesteps)):
            t_tensor_batch = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            betas_t = self._extract(self.betas, t_tensor_batch, img.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor_batch, img.shape)
            sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t_tensor_batch, img.shape)
            
            predicted_noise_from_model = self.model(img, t_tensor_batch, processed_conditions)
            
            model_mean = sqrt_recip_alphas_t * (img - betas_t * predicted_noise_from_model / sqrt_one_minus_alphas_cumprod_t)
            if i == 0:
                img = model_mean
            else:
                posterior_variance_t = self._extract(self.posterior_variance, t_tensor_batch, img.shape)
                noise_sample = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise_sample
        return img

