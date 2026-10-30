"""Writing the pretrained weights to the core array (hidden device 1 of a TransferCompound tile)."""
import torch


@torch.no_grad()
def program_core_noise(tile, sigma_abs: float = 0.0, sigma_rel: float = 0.0, idx: int = 1):
    """Exact write followed by Gaussian noise, w + N(0, sigma_abs^2) + N(0, (sigma_rel |w|)^2),
    clipped to the cell bounds. Returns (written weights, target weights)."""
    hp = tile.get_hidden_parameters()
    tgt = hp[f"hidden_weights_{idx}"].clone()
    dev = tgt.device
    w = tgt + sigma_abs * torch.randn_like(tgt) + sigma_rel * tgt.abs() * torch.randn_like(tgt)
    bmax, bmin = hp[f"max_bound_{idx}"].to(dev), hp[f"min_bound_{idx}"].to(dev)
    w = torch.maximum(torch.minimum(w, bmax), bmin)
    hp[f"hidden_weights_{idx}"] = w
    tile.set_hidden_parameters(hp)
    return w, tgt
