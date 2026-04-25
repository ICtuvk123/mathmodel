class BaseView:
    '''
    BaseView class, from which all views inherit. Implements the 
        following functions:
    '''

    def __init__(self):
        pass

    def view(self, im):
        '''
        Apply transform to an image.

        im (`torch.tensor`):
            For stage 1: Tensor of shape (3, H, W) representing a noisy image
            OR
            For stage 2: Tensor of shape (6, H, W) representing a noisy image
            concatenated with an upsampled conditioning image from stage 1
        '''
        raise NotImplementedError()

    def inverse_view(self, noise):
        '''
        Apply inverse transform to noise estimates.
            Because DeepFloyd estimates the variance in addition to
            the noise, this function must apply the inverse to the
            variance as well.

        noise (`torch.tensor`):
            Tensor of shape (6, H, W) representing the noise estimate
            (first three channel dims) and variance estimates (last
            three channel dims)
        '''
        raise NotImplementedError()

    def inverse_clean(self, image, return_mask=False):
        '''
        Map an image-like tensor from view space back into canonical space.

        By default this reuses `inverse_view` and assumes full coverage.
        Views with partial coverage or frequency-aware synchronization can
        override this method.
        '''
        inverted = self.inverse_view(image)
        if return_mask:
            mask = image.new_ones((1, inverted.shape[-2], inverted.shape[-1]))
            return inverted, mask
        return inverted

    def clean_sync_weight(self, progress):
        '''
        Relative weight when aggregating clean predictions across views.

        `progress` is in [0,1], where 0 corresponds to the earliest/noisiest
        denoising step and 1 corresponds to the final/cleanest step.
        '''
        return 1.0

    def make_frame(self, im, t):
        '''
        Make a frame, transitioning linearly from the identity view (t=0) 
            to this view (t=1)

        im (`PIL.Image`):
            A PIL Image of the illusion

        t (float):
            A float in [0,1] indicating time in the animation. Should start
            at the identity view at t=0, and continuously transition to the
            view at t=1.
        '''
        raise NotImplementedError()
