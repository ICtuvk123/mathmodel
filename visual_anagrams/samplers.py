from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffusers.utils.torch_utils import randn_tensor

TIME_TRAVEL_START_FRAC = 0.2
TIME_TRAVEL_END_FRAC = 0.8
TIME_TRAVEL_REPEATS = 2


def _get_execution_device(model, fallback):
    device = getattr(model, "_execution_device", None)
    if device is not None:
        return torch.device(device)
    if fallback is not None and getattr(fallback, "device", None) is not None:
        return fallback.device
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _prediction_type(scheduler):
    config = getattr(scheduler, "config", None)
    prediction_type = getattr(config, "prediction_type", None)
    return 'epsilon' if prediction_type is None else prediction_type


def _alpha_terms(scheduler, timestep, sample):
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
    beta_prod_t = (1 - alpha_prod_t).clamp_min(1e-8)
    return torch.sqrt(alpha_prod_t), torch.sqrt(beta_prod_t)


def _predict_original_sample(scheduler, model_output, sample, timestep):
    prediction_type = _prediction_type(scheduler)
    sqrt_alpha_prod_t, sqrt_beta_prod_t = _alpha_terms(scheduler, timestep, sample)

    if prediction_type == 'epsilon':
        return (sample - sqrt_beta_prod_t * model_output) / sqrt_alpha_prod_t
    if prediction_type == 'sample':
        return model_output
    if prediction_type == 'v_prediction':
        return sqrt_alpha_prod_t * sample - sqrt_beta_prod_t * model_output
    raise ValueError(f'Unsupported prediction type: {prediction_type}')


def _predict_noise_from_original_sample(scheduler, original_sample, sample, timestep):
    prediction_type = _prediction_type(scheduler)
    sqrt_alpha_prod_t, sqrt_beta_prod_t = _alpha_terms(scheduler, timestep, sample)

    if prediction_type == 'epsilon':
        return (sample - sqrt_alpha_prod_t * original_sample) / sqrt_beta_prod_t
    if prediction_type == 'sample':
        return original_sample
    if prediction_type == 'v_prediction':
        eps = (sample - sqrt_alpha_prod_t * original_sample) / sqrt_beta_prod_t
        return sqrt_alpha_prod_t * eps - sqrt_beta_prod_t * original_sample
    raise ValueError(f'Unsupported prediction type: {prediction_type}')


def _stabilize_clean_sample(scheduler, clean_sample):
    config = getattr(scheduler, "config", None)
    if config is None:
        return clean_sample.clamp(-1.0, 1.0)

    if getattr(config, "thresholding", False) and hasattr(scheduler, "_threshold_sample"):
        return scheduler._threshold_sample(clean_sample)

    if getattr(config, "clip_sample", True):
        clip_sample_range = float(getattr(config, "clip_sample_range", 1.0))
        return clean_sample.clamp(-clip_sample_range, clip_sample_range)

    return clean_sample


def _inverse_with_mask(view, tensor, clean_sync):
    if clean_sync:
        inverted, mask = view.inverse_clean(tensor, return_mask=True)
        return inverted, mask

    inverted = view.inverse_view(tensor)
    mask = tensor.new_ones((1, inverted.shape[-2], inverted.shape[-1]))
    return inverted, mask


def _progress(step_idx, total_steps):
    if total_steps <= 1:
        return 1.0
    return step_idx / (total_steps - 1)


def _view_weight(view, progress, clean_sync):
    if not clean_sync:
        return 1.0
    return float(view.clean_sync_weight(progress))


def _aggregate_inverse_predictions(predictions,
                                   views,
                                   reduction,
                                   step_idx,
                                   clean_sync,
                                   progress):
    if reduction == 'alternate':
        idx = step_idx % len(views)
        inverted, _ = _inverse_with_mask(views[idx], predictions[idx], clean_sync)
        return inverted

    inverted_predictions = []
    masks = []
    weights = []
    for pred, view in zip(predictions, views):
        inverted, mask = _inverse_with_mask(view, pred, clean_sync)
        inverted_predictions.append(inverted)
        masks.append(mask)
        weights.append(pred.new_full((1, 1, 1), _view_weight(view, progress, clean_sync)))

    inverted_predictions = torch.stack(inverted_predictions)
    masks = torch.stack(masks)
    weights = torch.stack(weights)
    weighted_masks = masks * weights

    if reduction == 'sum':
        return (inverted_predictions * weighted_masks).sum(0)
    if reduction != 'mean':
        raise ValueError('Reduction must be either `mean`, `sum`, or `alternate`')

    weighted = (inverted_predictions * weighted_masks).sum(0)
    normalizer = weighted_masks.sum(0).clamp_min(1e-6)
    return weighted / normalizer


def _sync_clean_prediction(noisy_images,
                           viewed_noisy_images,
                           noise_pred_uncond,
                           noise_pred_text,
                           views,
                           scheduler,
                           timestep,
                           guidance_scale,
                           reduction,
                           step_idx,
                           total_steps):

    progress = _progress(step_idx, total_steps)

    sample_channels = noisy_images.shape[1]
    eps_uncond, _ = noise_pred_uncond.split(sample_channels, dim=1)
    eps_text, predicted_variance = noise_pred_text.split(sample_channels, dim=1)

    clean_uncond = torch.stack([
        _predict_original_sample(scheduler, pred, sample, timestep)
        for pred, sample in zip(eps_uncond, viewed_noisy_images)
    ])
    clean_text = torch.stack([
        _predict_original_sample(scheduler, pred, sample, timestep)
        for pred, sample in zip(eps_text, viewed_noisy_images)
    ])

    synced_clean_uncond = _aggregate_inverse_predictions(
        clean_uncond, views, reduction, step_idx, clean_sync=True, progress=progress
    )
    synced_clean_text = _aggregate_inverse_predictions(
        clean_text, views, reduction, step_idx, clean_sync=True, progress=progress
    )
    synced_clean = synced_clean_uncond + guidance_scale * (synced_clean_text - synced_clean_uncond)
    synced_clean = _stabilize_clean_sample(scheduler, synced_clean)

    synced_noise = _predict_noise_from_original_sample(
        scheduler, synced_clean, noisy_images[0], timestep
    )[None]
    synced_variance = _aggregate_inverse_predictions(
        predicted_variance, views, reduction, step_idx, clean_sync=False, progress=progress
    )[None]

    return torch.cat([synced_noise, synced_variance], dim=1), synced_clean[None]


def _time_travel_passes(step_idx,
                        total_steps,
                        enabled,
                        start_frac,
                        end_frac,
                        repeats):
    if not enabled or repeats <= 1:
        return 1

    progress = _progress(step_idx, total_steps)
    if start_frac <= progress <= end_frac:
        return repeats
    return 1


def _renoise_from_clean(scheduler, clean_image, timestep, generator):
    clean_image = _stabilize_clean_sample(scheduler, clean_image)
    sqrt_alpha_prod_t, sqrt_beta_prod_t = _alpha_terms(scheduler, timestep, clean_image[0])
    noise = randn_tensor(
        clean_image.shape,
        generator=generator,
        device=clean_image.device,
        dtype=clean_image.dtype,
    )
    return sqrt_alpha_prod_t * clean_image + sqrt_beta_prod_t * noise

@torch.no_grad()
def sample_stage_1(model,
                   prompt_embeds,
                   negative_prompt_embeds, 
                   views,
                   ref_im=None,
                   num_inference_steps=100,
                   guidance_scale=7.0,
                   reduction='mean',
                   generator=None,
                   enable_time_travel=True,
                   time_travel_start=TIME_TRAVEL_START_FRAC,
                   time_travel_end=TIME_TRAVEL_END_FRAC,
                   time_travel_repeats=TIME_TRAVEL_REPEATS):

    # Params
    num_images_per_prompt = 1
    device = _get_execution_device(model, prompt_embeds)
    height = model.unet.config.sample_size
    width = model.unet.config.sample_size
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    assert num_prompts == len(views), \
        "Number of prompts must match number of views!"

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Setup timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # Make intermediate_images
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        model.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # Resize ref image to correct size
    if ref_im is not None:
        ref_im = TF.resize(ref_im, height)
        ref_im = ref_im.to(noisy_images.device).to(noisy_images.dtype)

    total_steps = len(timesteps)
    for i, t in enumerate(tqdm(timesteps)):
        passes = _time_travel_passes(
            i, total_steps, enable_time_travel, time_travel_start, time_travel_end, time_travel_repeats
        )
        for pass_idx in range(passes):
            # If solving an inverse problem, then project x_t so
            # that first component matches reference image's first component
            if ref_im is not None:
                alpha_cumprod = model.scheduler.alphas_cumprod[t]
                ref_noisy = torch.sqrt(alpha_cumprod) * ref_im + \
                            torch.sqrt(1 - alpha_cumprod) * torch.randn_like(ref_im)

                ref_noisy_component = views[0].inverse_view(ref_noisy)
                noisy_images_component = views[1].inverse_view(noisy_images[0])
                noisy_images = (ref_noisy_component + noisy_images_component)[None]

            viewed_noisy_images = []
            for view_fn in views:
                viewed_noisy_images.append(view_fn.view(noisy_images[0]))
            viewed_noisy_images = torch.stack(viewed_noisy_images)

            model_input = torch.cat([viewed_noisy_images] * 2)
            model_input = model.scheduler.scale_model_input(model_input, t)

            noise_pred = model.unet(
                model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred, synced_clean = _sync_clean_prediction(
                noisy_images,
                viewed_noisy_images,
                noise_pred_uncond,
                noise_pred_text,
                views,
                model.scheduler,
                t,
                guidance_scale,
                reduction,
                i,
                total_steps,
            )

            next_noisy_images = model.scheduler.step(
                noise_pred, t, noisy_images, generator=generator, return_dict=False
            )[0]
            if pass_idx + 1 < passes:
                noisy_images = _renoise_from_clean(model.scheduler, synced_clean, t, generator)
            else:
                noisy_images = next_noisy_images

    # Return denoised images
    return noisy_images







@torch.no_grad()
def sample_stage_2(model,
                   image,
                   prompt_embeds,
                   negative_prompt_embeds, 
                   views,
                   ref_im=None,
                   num_inference_steps=100,
                   guidance_scale=7.0,
                   reduction='mean',
                   noise_level=50,
                   generator=None,
                   enable_time_travel=True,
                   time_travel_start=TIME_TRAVEL_START_FRAC,
                   time_travel_end=TIME_TRAVEL_END_FRAC,
                   time_travel_repeats=TIME_TRAVEL_REPEATS):

    # Params
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    height = model.unet.config.sample_size
    width = model.unet.config.sample_size
    device = _get_execution_device(model, prompt_embeds)
    num_images_per_prompt = 1

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Get timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    num_channels = model.unet.config.in_channels // 2
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        num_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # Resize ref image to correct size
    if ref_im is not None:
        ref_im = TF.resize(ref_im, height)
        ref_im = ref_im.to(noisy_images.device).to(noisy_images.dtype)

    # Prepare upscaled image and noise level
    image = model.preprocess_image(image, num_images_per_prompt, device)
    upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

    noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
    noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
    upscaled = model.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

    # Condition on noise level, for each model input
    noise_level = torch.cat([noise_level] * num_prompts * 2)

    # Denoising Loop
    total_steps = len(timesteps)
    for i, t in enumerate(tqdm(timesteps)):
        passes = _time_travel_passes(
            i, total_steps, enable_time_travel, time_travel_start, time_travel_end, time_travel_repeats
        )
        for pass_idx in range(passes):
            if ref_im is not None:
                alpha_cumprod = model.scheduler.alphas_cumprod[t]
                ref_noisy = torch.sqrt(alpha_cumprod) * ref_im + \
                            torch.sqrt(1 - alpha_cumprod) * torch.randn_like(ref_im)

                ref_noisy_component = views[0].inverse_view(ref_noisy)
                noisy_images_component = views[1].inverse_view(noisy_images[0])
                noisy_images = (ref_noisy_component + noisy_images_component)[None]

            model_input = torch.cat([noisy_images, upscaled], dim=1)

            viewed_inputs = []
            for view_fn in views:
                viewed_inputs.append(view_fn.view(model_input[0]))
            viewed_inputs = torch.stack(viewed_inputs)
            viewed_noisy_images = viewed_inputs[:, :num_channels]

            model_input = torch.cat([viewed_inputs] * 2)
            model_input = model.scheduler.scale_model_input(model_input, t)

            noise_pred = model.unet(
                model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=noise_level,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred, synced_clean = _sync_clean_prediction(
                noisy_images,
                viewed_noisy_images,
                noise_pred_uncond,
                noise_pred_text,
                views,
                model.scheduler,
                t,
                guidance_scale,
                reduction,
                i,
                total_steps,
            )

            next_noisy_images = model.scheduler.step(
                noise_pred, t, noisy_images, generator=generator, return_dict=False
            )[0]
            if pass_idx + 1 < passes:
                noisy_images = _renoise_from_clean(model.scheduler, synced_clean, t, generator)
            else:
                noisy_images = next_noisy_images

    # Return denoised images
    return noisy_images
