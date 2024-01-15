import torch


class DDPM:
    def __init__(self, beta1: float, beta2: float, time_steps: int, device: str) -> None:
        # construct DDPM noise schedule
        self.time_steps = time_steps
        self.b_t = torch.linspace(0, 1, time_steps + 1, device=device) * (beta2 - beta1) + beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()  # type: ignore
        self.ab_t[0] = 1

    # helper function: perturbs an image to a specified noise level
    def perturb_input(self, x: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise  # type: ignore

    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(
        self, x: torch.Tensor, t: int, pred_noise: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z  # type: ignore
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()  # type: ignore
        return mean + noise  # type: ignore
