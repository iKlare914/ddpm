import torch as th

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def to_tensor(x, d_type=th.float32, device=device):
    """Convert array-like input to a tensor with the requested dtype and device."""
    return th.as_tensor(x, dtype=d_type, device=device)

def make_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2):
    """Create a linear beta schedule."""
    if num_timesteps <= 50:
        raise ValueError("num_timesteps must be greater than 50 for a meaningful beta schedule.")
    scale = 1000 / num_timesteps
    beta_start = beta_start * scale
    beta_end = beta_end * scale
    betas = th.linspace(beta_start, beta_end, num_timesteps)
    if th.any(betas <= 0) or th.any(betas >= 1):
        raise ValueError("Beta values must be in the range (0, 1).")
    return betas


class DDPMSampler:
    def __init__(self, model, betas=None, d_type=th.float32, device=device, cond_fn=None, num_classes=None):
        """
        Args:
            model: The diffusion model to sample from.
            betas: Tensor or array of shape (T,). The beta schedule.
            d_type: The data type for the tensors.
            device: The device to run the sampling on.
        """
        self.model = model
        self.betas = to_tensor(betas, d_type=d_type, device=device) if betas is not None else None
        self.d_type = d_type
        self.cond_fn = cond_fn
        self.num_classes = num_classes
        self.device = th.device(device)
        if self.betas is not None:
            self._set_betas(self.betas)

    def load_config_from_model(self, model):
        """Load the configuration from a pre-trained model. If model has betas, use them; otherwise, use the default betas."""
        self.model = model
        betas = getattr(model, "betas", None)
        self._set_betas(self.betas if betas is None else betas)

    def _set_betas(self, betas):
        """Compute schedule tensors using the configured dtype and device."""
        if betas is None:
            raise ValueError("Betas must be provided either in the model or as an argument.")
        self.betas = to_tensor(betas, d_type=self.d_type, device=self.device) if not isinstance(betas, th.Tensor) else betas.to(dtype=self.d_type, device=self.device)
        self.num_timesteps = len(self.betas)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = th.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = th.cat([th.tensor([1.0], dtype=self.d_type, device=self.device), self.alpha_bars[:-1]], dim=0)
        self.m1alpha_bars = 1 - self.alpha_bars
        self.m1alpha_bars_prev = 1 - self.alpha_bars_prev
        self.sqrt_m1alpha_bars = th.sqrt(self.m1alpha_bars)
        self.sqrt_alphas_bars = th.sqrt(self.alpha_bars)
        self.reciprocal_sqrt_alphas = th.rsqrt(self.alphas)
        self.reciprocal_sqrt_m1alpha_bars = th.rsqrt(self.m1alpha_bars)
        self.reciprocal_sqrt_alphas_bars = th.rsqrt(self.alpha_bars)
        self.sqrt_m1alpha_bars_prev = th.sqrt(self.m1alpha_bars_prev)
        
    def q_mean_std(self, x_0, t):
       """
       Compute the mean and standard deviation of q(x_t | x_0) for a given x_0 and timestep t.
       Args:
           x_0: The original data point [B, C, H, W].
           t: The timestep [T].

        Returns:
            mean: The mean of q(x_t | x_0).
            std: The standard deviation of q(x_t | x_0).
            x_t: A sample from q(x_t | x_0).
       """
       if len(t.shape) != 1:
           raise ValueError(f"Expected t to be a 1D tensor, but got shape {t.shape}.")
       mean = self.sqrt_alphas_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_0
       std = self.sqrt_m1alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
       noise = th.randn_like(x_0) * std
       x_t = mean + noise
       return mean, std, x_t

    def p_mean_std(self, x_t, t, guided=False, y=None, guidance_scale=1.0):
        """
        Compute the mean and standard deviation of p(x_{t-1} | x_t) for a given x_t and timestep t.
        Args:
            x_t: The noisy data point at timestep t, [B, C, H, W].
            t: The timestep, [B].
            guided: Whether to use guided sampling, defalult is False.
            y: The labels for guided sampling, [B], required if guided is True.
            guidance_scale: The scale for guided sampling, default is 1.0.
        Returns:
            mean: The mean of p(x_{t-1} | x_t).
            std: The standard deviation of p(x_{t-1} | x_t).
            x_t_prev: A sample from p(x_{t-1} | x_t).
        """
        if len(t.shape) != 1:
            raise ValueError(f"Expected t to be a 1D tensor, but got shape {t.shape}.")
        pred_noise = self.model(x_t, t)
        reciprocal_sqrt_m1alpha_bars_t = self.reciprocal_sqrt_m1alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        betas_t = self.betas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_m1alpha_bars_prev_t = self.sqrt_m1alpha_bars_prev[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_m1alpha_bars_t = self.sqrt_m1alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        reciprocal_sqrt_alphas_t = self.reciprocal_sqrt_alphas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if guided:
            if self.cond_fn is None or y is None:
                raise ValueError("Guided sampling requires a conditional function and labels y.")
            if self.cond_fn is not None and y is not None:
                pred_noise = pred_noise - self.cond_fn(x_t, t, y) * guidance_scale * sqrt_m1alpha_bars_t
        mean = reciprocal_sqrt_alphas_t * (x_t - betas_t * pred_noise * reciprocal_sqrt_m1alpha_bars_t)
        std = th.sqrt(betas_t) * sqrt_m1alpha_bars_prev_t * reciprocal_sqrt_m1alpha_bars_t
        noise = th.randn_like(x_t) * std
        x_t_prev = mean + noise
        return mean, std, x_t_prev


    def q_sample(self, x_0, t):
        """
        Sample from q(x_t | x_0) for a given x_0 and timestep t.
        Args:
            x_0: The original data point [B, C, H, W].
            t: The timestep [B].
        Returns:
            x_t: A sample from q(x_t | x_0).
        """
        _, _ , x_t = self.q_mean_std(x_0, t)
        return x_t

    def p_sample(self, x_t, t, guided=False, y=None, guidance_scale=1.0):
        """
        Sample from p(x_{t-1} | x_t) for a given x_t and timestep t.
        Args:
            x_t: The noisy data point at timestep t, [B, C, H, W].
            t: The timestep, [T].
            guided: Whether to use guided sampling, default is False.
            y: The labels for guided sampling, [B], required if guided is True.
            guidance_scale: The scale for guided sampling, default is 1.0.
        Returns:
            x_t_prev: A sample from p(x_{t-1} | x_t).
        """
        _, _, x_t_prev = self.p_mean_std(x_t, t, guided, y, guidance_scale)
        return x_t_prev