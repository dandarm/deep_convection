"""Historical tube, random and cuboid-block masks with equal batch counts."""
import torch


def make_ssl_mask(config, batch_size, ratio=.9, strategy="tube", *, generator=None, device="cpu"):
    if not 0 < ratio < 1 or batch_size < 1:
        raise ValueError("Mask ratio must be in (0,1) and batch size positive")
    time = config.num_frames // config.tubelet_size
    grid = config.image_size // config.patch_size
    if grid < 2:
        raise ValueError("Masking needs at least a 2x2 spatial patch grid")
    total = time * grid * grid
    rows = []
    for _ in range(batch_size):
        if strategy == "tube":
            count = min(max(round(grid * grid * ratio), 1), grid * grid - 1)
            row = torch.zeros(grid * grid, dtype=torch.bool)
            row[torch.randperm(len(row), generator=generator)[:count]] = True
            rows.append(row.repeat(time))
            continue
        count = min(max(round(total * ratio), 1), total - 1)
        if strategy == "random":
            order = torch.randperm(total, generator=generator)
        elif strategy == "block":
            # Randomly ordered disjoint 2x2x2 cuboids (tubelet/patch units).
            # The final cuboid may be partial to obtain exactly `count` tokens.
            t, y, x = torch.meshgrid(torch.arange(time), torch.arange(grid), torch.arange(grid), indexing="ij")
            blocks = (t // 2) * ((grid + 1) // 2) ** 2 + (y // 2) * ((grid + 1) // 2) + x // 2
            priorities = torch.randperm(int(blocks.max()) + 1, generator=generator)
            order = torch.argsort(priorities[blocks.flatten()], stable=True)
        else:
            raise ValueError("Mask strategy must be tube, random or block")
        row = torch.zeros(total, dtype=torch.bool)
        row[order[:count]] = True
        rows.append(row)
    return torch.stack(rows).to(device)


def fixed_ssl_mask(config, indices, ratio, strategy, seed, device):
    return torch.cat([make_ssl_mask(config, 1, ratio, strategy,
        generator=torch.Generator().manual_seed(seed + int(index))) for index in indices]).to(device)
