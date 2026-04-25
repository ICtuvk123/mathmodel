from pathlib import Path
from PIL import Image
import numpy as np

from .view_identity import IdentityView
from .view_flip import FlipView
from .view_rotate import Rotate180View, Rotate90CCWView, Rotate90CWView
from .view_negate import NegateView
from .view_skew import SkewView
from .view_patch_permute import PatchPermuteView
from .view_jigsaw import JigsawView
from .view_inner_circle import InnerCircleView, InnerCircleViewFailure
from .view_square_hinge import SquareHingeView
from .view_blur import BlurViewFailure
from .view_white_balance import WhiteBalanceViewFailure
from .view_hybrid import HybridLowPassView, HybridHighPassView, \
    TripleHybridHighPassView, TripleHybridLowPassView, \
    TripleHybridMediumPassView
from .view_color import ColorView, GrayscaleView
from .view_motion import MotionBlurResView, MotionBlurView
from .view_scale import ScaleView
from .view_cylindrical import CylindricalMirrorView

VIEW_MAP = {
    'identity': IdentityView,
    'flip': FlipView,
    'rotate_cw': Rotate90CWView,
    'rotate_ccw': Rotate90CCWView,
    'rotate_180': Rotate180View,
    'negate': NegateView,
    'skew': SkewView,
    'patch_permute': PatchPermuteView,
    'pixel_permute': PatchPermuteView,
    'jigsaw': JigsawView,
    'inner_circle': InnerCircleView,
    'square_hinge': SquareHingeView,
    'inner_circle_failure': InnerCircleViewFailure,
    'blur_failure': BlurViewFailure,
    'white_balance_failure': WhiteBalanceViewFailure,
    'low_pass': HybridLowPassView,
    'high_pass': HybridHighPassView,
    'triple_low_pass': TripleHybridLowPassView,
    'triple_medium_pass': TripleHybridMediumPassView,
    'triple_high_pass': TripleHybridHighPassView,
    'grayscale': GrayscaleView,
    'color': ColorView,
    'motion': MotionBlurView,
    'motion_res': MotionBlurResView,
    'scale': ScaleView,
    'cylindrical': CylindricalMirrorView,
}

VIEWS_WITH_ARGS = {
    'patch_permute',
    'pixel_permute',
    'skew',
    'low_pass',
    'high_pass',
    'scale',
    'cylindrical',
}


def _normalize_view_arg(view_arg):
    if view_arg is None:
        return None
    if isinstance(view_arg, str) and view_arg.strip().lower() in {'none', 'null', ''}:
        return None
    return view_arg


def _align_view_args(view_names, view_args):
    if view_args is None:
        return [None for _ in view_names]

    normalized_args = [_normalize_view_arg(arg) for arg in view_args]

    # Positional mode: one arg slot per view.
    if len(normalized_args) == len(view_names):
        return normalized_args

    # Compact mode: only views that actually accept args consume them.
    aligned_args = []
    arg_idx = 0
    for view_name in view_names:
        if view_name in VIEWS_WITH_ARGS and arg_idx < len(normalized_args):
            aligned_args.append(normalized_args[arg_idx])
            arg_idx += 1
        else:
            aligned_args.append(None)

    if arg_idx != len(normalized_args):
        raise ValueError(
            "Too many --view_args values were provided for the selected --views."
        )

    return aligned_args

def get_views(view_names, view_args=None):
    '''
    Bespoke function to get views (just to make command line usage easier)
    '''

    views = []
    view_args = _align_view_args(view_names, view_args)

    for view_name, view_arg in zip(view_names, view_args):
        if view_name == 'patch_permute':
            args = [8 if view_arg is None else int(view_arg)]
        elif view_name == 'pixel_permute':
            args = [64 if view_arg is None else int(view_arg)]
        elif view_name == 'skew':
            args = [1.5 if view_arg is None else float(view_arg)]
        elif view_name in ['low_pass', 'high_pass']:
            args = [2.0 if view_arg is None else float(view_arg)]
        elif view_name in ['scale']:
            args = [0.5 if view_arg is None else float(view_arg)]
        elif view_name == 'cylindrical':
            if view_arg is None:
                args = [0.2, 180.0]
            else:
                parts = [p.strip() for p in str(view_arg).split(',') if p.strip()]
                if len(parts) == 1:
                    args = [float(parts[0]), 180.0]
                elif len(parts) == 2:
                    args = [float(parts[0]), float(parts[1])]
                else:
                    raise ValueError(
                        "Cylindrical view expects `radius_ratio[,theta_deg]` as view_arg."
                    )
        else:
            args = []

        view = VIEW_MAP[view_name](*args)
        views.append(view)

    return views
