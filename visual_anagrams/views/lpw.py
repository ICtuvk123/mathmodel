import torch
import torchvision.transforms.functional as TF


def gaussian_blur_level(image, level):
    if level <= 0:
        return image

    kernel_size = min(image.shape[-1] | 1, 2 * (2 ** level) + 1)
    sigma = float(2 ** (level - 1))
    return TF.gaussian_blur(image, [kernel_size, kernel_size], [sigma, sigma])


def build_gaussian_pyramid(image, max_levels=4):
    return [gaussian_blur_level(image, level) for level in range(max_levels)]


def blend_lod_pyramid(image, lod_map, max_levels=4):
    lod_map = lod_map.clamp(0, max_levels - 1)
    base_level = torch.floor(lod_map)
    next_level = (base_level + 1).clamp(max=max_levels - 1)
    mix = lod_map - base_level

    blended = image.new_zeros(image.shape)
    for level, level_image in enumerate(build_gaussian_pyramid(image, max_levels=max_levels)):
        weight = ((base_level == level).to(image.dtype) * (1.0 - mix)) + \
                 ((next_level == level).to(image.dtype) * mix)
        blended = blended + level_image * weight

    return blended
