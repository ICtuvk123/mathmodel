from .view_base import BaseView

class IdentityView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return im

    def inverse_view(self, noise):
        return noise

    def clean_sync_weight(self, progress):
        if progress <= 0.8:
            return 3.0
        ramp = min(max((progress - 0.8) / 0.2, 0.0), 1.0)
        return 3.0 + 2.0 * ramp
