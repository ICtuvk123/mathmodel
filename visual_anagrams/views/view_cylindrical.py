import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .view_base import BaseView


class CylindricalMirrorView(BaseView):
    """
    Approximate cylindrical-mirror anamorphosis as a view transform.

    The canonical image is the distorted paper image. Calling `view()` simulates
    the image seen in the cylindrical mirror by sampling an undistorted square
    patch from the annular distortion around the cylinder. `inverse_view()`
    performs the corresponding splat back into the paper coordinate system.
    """

    def __init__(self, radius_ratio=0.2, theta_deg=180.0, flip_horizontal=True):
        if not 0.0 < radius_ratio < 0.5:
            raise ValueError("`radius_ratio` must be in (0, 0.5).")
        if not 0.0 < theta_deg <= 360.0:
            raise ValueError("`theta_deg` must be in (0, 360].")

        self.radius_ratio = radius_ratio
        self.theta_deg = theta_deg
        self.flip_horizontal = flip_horizontal
        self._view_grid_cache = {}
        self._inverse_cache = {}

    def _cache_key(self, size, device, dtype):
        return (size, str(device), str(dtype))

    def _geometry(self, size):
        radius = self.radius_ratio * size
        content = int(round(size / 2.0 - radius))
        content = max(2, min(size, content))
        offset = (size - content) // 2
        center = (size - 1) / 2.0
        return radius, content, offset, center

    def _build_sampling_coords(self, size, device, dtype):
        radius, content, offset, center = self._geometry(size)

        rows = torch.arange(content, device=device, dtype=torch.float32)
        cols = torch.arange(content, device=device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')

        if self.flip_horizontal:
            col_grid = (content - 1) - col_grid

        if content == 1:
            gamma = torch.zeros_like(col_grid)
        else:
            gamma = (0.5 - col_grid / (content - 1)) * math.radians(self.theta_deg)

        radial = radius + (content - 1 - row_grid)
        sample_y = center - radial * torch.cos(gamma)
        sample_x = center - radial * torch.sin(gamma)

        sample_x = sample_x.to(dtype)
        sample_y = sample_y.to(dtype)
        return sample_x, sample_y, content, offset

    def _get_view_grid(self, size, device, dtype):
        key = self._cache_key(size, device, dtype)
        if key in self._view_grid_cache:
            return self._view_grid_cache[key]

        sample_x, sample_y, content, offset = self._build_sampling_coords(size, device, dtype)

        grid = torch.full((size, size, 2), 2.0, device=device, dtype=dtype)
        denom = max(size - 1, 1)
        grid[offset:offset + content, offset:offset + content, 0] = 2.0 * sample_x / denom - 1.0
        grid[offset:offset + content, offset:offset + content, 1] = 2.0 * sample_y / denom - 1.0

        self._view_grid_cache[key] = grid
        return grid

    def _get_inverse_cache(self, size, device, dtype):
        key = self._cache_key(size, device, dtype)
        if key in self._inverse_cache:
            return self._inverse_cache[key]

        sample_x, sample_y, content, offset = self._build_sampling_coords(size, device, dtype)
        sample_x = sample_x.reshape(-1)
        sample_y = sample_y.reshape(-1)

        base_rows = torch.arange(offset, offset + content, device=device, dtype=torch.long)
        base_cols = torch.arange(offset, offset + content, device=device, dtype=torch.long)
        src_row_grid, src_col_grid = torch.meshgrid(base_rows, base_cols, indexing='ij')
        src_flat_idx = (src_row_grid * size + src_col_grid).reshape(-1)

        x0 = torch.floor(sample_x).to(torch.long)
        y0 = torch.floor(sample_y).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = sample_x - x0.to(dtype)
        wy1 = sample_y - y0.to(dtype)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        src_neighbor_indices = []
        neighbor_indices = []
        neighbor_weights = []
        for x_idx, y_idx, weight in [
            (x0, y0, wx0 * wy0),
            (x1, y0, wx1 * wy0),
            (x0, y1, wx0 * wy1),
            (x1, y1, wx1 * wy1),
        ]:
            valid = (x_idx >= 0) & (x_idx < size) & (y_idx >= 0) & (y_idx < size)
            src_neighbor_indices.append(src_flat_idx[valid])
            neighbor_indices.append((y_idx[valid] * size + x_idx[valid]).to(torch.long))
            neighbor_weights.append(weight[valid].to(dtype))

        cache = {
            'src_neighbor_indices': src_neighbor_indices,
            'dst_flat_indices': neighbor_indices,
            'dst_weights': neighbor_weights,
        }
        self._inverse_cache[key] = cache
        return cache

    def view(self, im):
        _, height, width = im.shape
        if height != width:
            raise ValueError("CylindricalMirrorView expects square inputs.")

        grid = self._get_view_grid(height, im.device, im.dtype)
        viewed = F.grid_sample(
            im.unsqueeze(0),
            grid.unsqueeze(0),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return viewed[0]

    def inverse_view(self, noise):
        channels, height, width = noise.shape
        if height != width:
            raise ValueError("CylindricalMirrorView expects square inputs.")

        cache = self._get_inverse_cache(height, noise.device, noise.dtype)

        noise_flat = noise.reshape(channels, -1)
        out_flat = torch.zeros_like(noise_flat)
        weight_flat = torch.zeros(height * width, device=noise.device, dtype=noise.dtype)

        for src_idx, dst_idx, weight in zip(
            cache['src_neighbor_indices'],
            cache['dst_flat_indices'],
            cache['dst_weights'],
        ):
            if dst_idx.numel() == 0:
                continue
            src_vals = noise_flat[:, src_idx]
            weighted_vals = src_vals * weight.unsqueeze(0)
            out_flat.scatter_add_(1, dst_idx.unsqueeze(0).expand(channels, -1), weighted_vals)
            weight_flat.scatter_add_(0, dst_idx, weight)

        valid = weight_flat > 0
        if valid.any():
            out_flat[:, valid] = out_flat[:, valid] / weight_flat[valid].unsqueeze(0)

        return out_flat.reshape(channels, height, width)

    def save_view(self, im):
        viewed = self.view(im).clone()
        _, size, _ = viewed.shape
        _, content, offset, _ = self._geometry(size)

        viewed[:, :offset, :] = 1.0
        viewed[:, offset + content:, :] = 1.0
        viewed[:, offset:offset + content, :offset] = 1.0
        viewed[:, offset:offset + content, offset + content:] = 1.0
        return viewed

    def make_frame(self, im, t):
        im_tensor = TF.to_tensor(im) * 2 - 1
        viewed = self.save_view(im_tensor)
        viewed = ((viewed + 1) / 2).clamp(0, 1)
        viewed_pil = TF.to_pil_image(viewed)

        if viewed_pil.size != im.size:
            viewed_pil = viewed_pil.resize(im.size, resample=Image.Resampling.BILINEAR)

        return Image.blend(im, viewed_pil, t)
