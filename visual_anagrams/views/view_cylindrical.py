import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .view_base import BaseView
from .lpw import blend_lod_pyramid


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
        self._lod_cache = {}

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

        coverage_flat = torch.zeros(size * size, device=device, dtype=dtype)
        for dst_idx, weight in zip(neighbor_indices, neighbor_weights):
            if dst_idx.numel() == 0:
                continue
            coverage_flat.scatter_add_(0, dst_idx, weight)

        cache = {
            'src_neighbor_indices': src_neighbor_indices,
            'dst_flat_indices': neighbor_indices,
            'dst_weights': neighbor_weights,
            'coverage': coverage_flat.reshape(1, size, size),
        }
        self._inverse_cache[key] = cache
        return cache

    def _get_lod_map(self, size, device, dtype):
        key = self._cache_key(size, device, dtype)
        if key in self._lod_cache:
            return self._lod_cache[key]

        sample_x, sample_y, content, offset = self._build_sampling_coords(size, device, dtype)

        dx_x = torch.zeros_like(sample_x)
        dx_y = torch.zeros_like(sample_y)
        dy_x = torch.zeros_like(sample_x)
        dy_y = torch.zeros_like(sample_y)

        dx_x[:, 1:-1] = (sample_x[:, 2:] - sample_x[:, :-2]) * 0.5
        dx_y[:, 1:-1] = (sample_y[:, 2:] - sample_y[:, :-2]) * 0.5
        dy_x[1:-1, :] = (sample_x[2:, :] - sample_x[:-2, :]) * 0.5
        dy_y[1:-1, :] = (sample_y[2:, :] - sample_y[:-2, :]) * 0.5

        dx_x[:, 0] = sample_x[:, 1] - sample_x[:, 0]
        dx_y[:, 0] = sample_y[:, 1] - sample_y[:, 0]
        dx_x[:, -1] = sample_x[:, -1] - sample_x[:, -2]
        dx_y[:, -1] = sample_y[:, -1] - sample_y[:, -2]
        dy_x[0, :] = sample_x[1, :] - sample_x[0, :]
        dy_y[0, :] = sample_y[1, :] - sample_y[0, :]
        dy_x[-1, :] = sample_x[-1, :] - sample_x[-2, :]
        dy_y[-1, :] = sample_y[-1, :] - sample_y[-2, :]

        stretch_x = torch.sqrt(dx_x.square() + dx_y.square())
        stretch_y = torch.sqrt(dy_x.square() + dy_y.square())
        stretch = torch.maximum(stretch_x, stretch_y).clamp_min(1.0)
        lod_patch = torch.log2(stretch)

        lod_map = torch.zeros((1, size, size), device=device, dtype=dtype)
        lod_map[:, offset:offset + content, offset:offset + content] = lod_patch.unsqueeze(0)
        self._lod_cache[key] = lod_map
        return lod_map

    def _prefilter_for_inverse(self, image, max_levels=4):
        _, height, width = image.shape
        if height != width:
            raise ValueError("CylindricalMirrorView expects square inputs.")

        lod_map = self._get_lod_map(height, image.device, image.dtype)
        return blend_lod_pyramid(image, lod_map, max_levels=max_levels)

    def _inverse_scatter(self, tensor):
        channels, height, width = tensor.shape
        cache = self._get_inverse_cache(height, tensor.device, tensor.dtype)

        tensor_flat = tensor.reshape(channels, -1)
        out_flat = torch.zeros_like(tensor_flat)

        for src_idx, dst_idx, weight in zip(
            cache['src_neighbor_indices'],
            cache['dst_flat_indices'],
            cache['dst_weights'],
        ):
            if dst_idx.numel() == 0:
                continue
            src_vals = tensor_flat[:, src_idx]
            weighted_vals = src_vals * weight.unsqueeze(0)
            out_flat.scatter_add_(1, dst_idx.unsqueeze(0).expand(channels, -1), weighted_vals)

        coverage = cache['coverage']
        valid = coverage.reshape(-1) > 0
        if valid.any():
            out_flat[:, valid] = out_flat[:, valid] / coverage.reshape(-1)[valid].unsqueeze(0)

        return out_flat.reshape(channels, height, width), coverage

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

        inverted, _ = self._inverse_scatter(noise)
        return inverted

    def inverse_clean(self, image, return_mask=False):
        filtered = self._prefilter_for_inverse(image)
        inverted, coverage = self._inverse_scatter(filtered)
        if return_mask:
            return inverted, coverage
        return inverted

    def uses_clean_sync(self):
        return True

    def get_valid_mask(self, size, device, dtype):
        coverage = self._get_inverse_cache(size, device, dtype)['coverage']
        return (coverage > 0).to(dtype)

    def get_coverage_map(self, size, device, dtype):
        return self._get_inverse_cache(size, device, dtype)['coverage']

    def get_lod_map(self, size, device, dtype):
        return self._get_lod_map(size, device, dtype)

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
