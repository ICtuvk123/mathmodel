import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if 'einops' not in sys.modules:
    einops_stub = types.ModuleType('einops')

    def _unused_rearrange(*args, **kwargs):
        raise RuntimeError('einops.rearrange should not be used in this test')

    einops_stub.rearrange = _unused_rearrange
    einops_stub.repeat = _unused_rearrange
    einops_stub.einsum = _unused_rearrange
    sys.modules['einops'] = einops_stub


if 'diffusers.utils.torch_utils' not in sys.modules:
    diffusers_stub = types.ModuleType('diffusers')
    diffusers_utils_stub = types.ModuleType('diffusers.utils')
    diffusers_torch_utils_stub = types.ModuleType('diffusers.utils.torch_utils')

    def _randn_tensor(shape, generator=None, device=None, dtype=None):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    diffusers_torch_utils_stub.randn_tensor = _randn_tensor
    diffusers_utils_stub.torch_utils = diffusers_torch_utils_stub
    diffusers_stub.utils = diffusers_utils_stub

    sys.modules['diffusers'] = diffusers_stub
    sys.modules['diffusers.utils'] = diffusers_utils_stub
    sys.modules['diffusers.utils.torch_utils'] = diffusers_torch_utils_stub


from visual_anagrams.samplers import _aggregate_inverse_predictions, _time_travel_passes
from visual_anagrams.views.view_cylindrical import CylindricalMirrorView
from visual_anagrams.views.view_identity import IdentityView


class LookingGlassCylindricalTests(unittest.TestCase):
    def test_cylindrical_maps_have_expected_shapes(self):
        view = CylindricalMirrorView(radius_ratio=0.2, theta_deg=180.0)

        lod = view.get_lod_map(64, torch.device('cpu'), torch.float32)
        coverage = view.get_coverage_map(64, torch.device('cpu'), torch.float32)
        mask = view.get_valid_mask(64, torch.device('cpu'), torch.float32)

        self.assertEqual(lod.shape, (1, 64, 64))
        self.assertEqual(coverage.shape, (1, 64, 64))
        self.assertEqual(mask.shape, (1, 64, 64))
        self.assertGreater(float(mask.sum()), 0.0)
        self.assertGreater(float(coverage.max()), 0.0)
        self.assertGreater(float(lod.max()), 0.0)

    def test_inverse_clean_is_reasonable_on_smooth_signal(self):
        view = CylindricalMirrorView(radius_ratio=0.2, theta_deg=180.0)
        coords = torch.linspace(-1.0, 1.0, 64)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        image = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0)

        mirrored = view.view(image)
        recovered, mask = view.inverse_clean(mirrored, return_mask=True)
        valid = mask.expand_as(recovered) > 0

        mse = ((recovered - image)[valid] ** 2).mean()
        self.assertLess(float(mse), 0.08)

    def test_identity_weight_ramps_late(self):
        view = IdentityView()
        self.assertEqual(view.clean_sync_weight(0.4), 1.0)
        self.assertGreater(view.clean_sync_weight(0.95), 1.0)

    def test_time_travel_only_runs_mid_schedule(self):
        self.assertEqual(_time_travel_passes(0, 30, True, 0.2, 0.8, 2), 1)
        self.assertEqual(_time_travel_passes(10, 30, True, 0.2, 0.8, 2), 2)
        self.assertEqual(_time_travel_passes(29, 30, True, 0.2, 0.8, 2), 1)

    def test_clean_aggregation_respects_identity_priority(self):
        identity = IdentityView()
        cylindrical = CylindricalMirrorView(radius_ratio=0.2, theta_deg=180.0)
        predictions = torch.stack([
            torch.ones(3, 8, 8),
            torch.zeros(3, 8, 8),
        ])

        early = _aggregate_inverse_predictions(
            predictions, [identity, cylindrical], 'mean', 0, True, 0.5
        )
        late = _aggregate_inverse_predictions(
            predictions, [identity, cylindrical], 'mean', 29, True, 0.95
        )

        self.assertGreater(float(late.mean()), float(early.mean()))


if __name__ == '__main__':
    unittest.main()
