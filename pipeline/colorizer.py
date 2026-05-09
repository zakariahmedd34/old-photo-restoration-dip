import cv2
import numpy as np


class Colorizer:
    """
    Corrects the color cast found in old photographs.

    Old photos fall into three categories that need different treatment:

    Grayscale
        Black-and-white photo.
        Apply unsharp mask + histogram stretch.

    Sepia
        Paper has yellowed, giving a warm brown-orange tone.
        Apply white-balance correction to neutralize the cast.

    Color
        Original colors still present — no correction needed.

    Classification uses mean HSV saturation and hue.
    """

    def __init__(
        self,
        sepia_sat_min=15,
        sepia_sat_max=60,
        sepia_hue_min=10,
        sepia_hue_max=30,
    ):
        """
        Parameters
        ----------
        sepia_sat_min : mean saturation below this  -> classified as grayscale
        sepia_sat_max : mean saturation above this  -> classified as real color
        sepia_hue_min : hue range start for the yellow-orange sepia band
        sepia_hue_max : hue range end  for the yellow-orange sepia band
        """
        self.sepia_sat_min = sepia_sat_min
        self.sepia_sat_max = sepia_sat_max
        self.sepia_hue_min = sepia_hue_min
        self.sepia_hue_max = sepia_hue_max

    def classify(self, image):
        """Return 'grayscale', 'sepia', or 'color' based on HSV statistics."""
        hsv      = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_sat = hsv[:, :, 1].mean()
        mean_hue = hsv[:, :, 0].mean()

        if mean_sat < self.sepia_sat_min:
            return 'grayscale'

        if (self.sepia_hue_min <= mean_hue <= self.sepia_hue_max
                and mean_sat < self.sepia_sat_max):
            return 'sepia'

        return 'color'

    def remove_sepia(self, image):
        """
        White-balance correction by gray-world:
        scale each channel so all three have the same mean.
        This neutralizes the yellow cast without changing scene content.
        """
        result      = image.astype(np.float32)
        global_mean = result.mean()

        for ch in range(3):           # B, G, R
            ch_mean = result[:, :, ch].mean()
            if ch_mean > 0:
                result[:, :, ch] *= (global_mean / ch_mean)

        return np.clip(result, 0, 255).astype(np.uint8)

    def clean_grayscale(self, image):
        """
        Sharpen and stretch contrast for black-and-white photos.

        Steps
        -----
        1. Unsharp mask (alpha=2.5, beta=-1.5, sigma=0.5) -- stronger
           to recover texture lost during denoising.
        2. Histogram stretch -- remap values to use the full 0-255 range.
        """
        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(gray, (1, 1), 0.5)
        sharpened = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
        stretched = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

    def apply(self, image):
        """
        Classify the image and apply the appropriate correction.

        Returns
        -------
        corrected : np.ndarray  the corrected image (BGR)
        kind      : str         'grayscale', 'sepia', or 'color'
        """
        kind = self.classify(image)

        if kind == 'sepia':
            return self.remove_sepia(image), kind

        if kind == 'grayscale':
            return self.clean_grayscale(image), kind

        return image.copy(), kind
