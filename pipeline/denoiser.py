import cv2
import numpy as np
import pywt


class Denoiser:
    """
    Advanced denoiser for old photo restoration.

    Supports:
    - spatial methods (median, bilateral, non-local means)
    - frequency method (wavelet denoising)
    """

    def __init__(
        self,
        method="nlm",
        median_ksize=3,
        nlm_h=10,
        nlm_template=7,
        nlm_search=21,
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        wavelet="db1",
        wavelet_level=2,
        wavelet_thresh=20,
        verbose=True,
    ):
        self.method = method
        self.median_ksize = median_ksize
        self.nlm_h = nlm_h
        self.nlm_template = nlm_template
        self.nlm_search = nlm_search
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_thresh = wavelet_thresh
        self.verbose = verbose

    # -------------------------
    # INTERNAL HELPERS
    # -------------------------

    def _validate_image(self, image):
        """
        Ensure image is valid.
        """
        if image is None:
            raise ValueError("Input image is None.")

    # -------------------------
    # CORE METHODS
    # -------------------------

    def _median(self, image):
        if self.verbose:
            print("Applying Median Filter")
        return cv2.medianBlur(image, self.median_ksize)

    def _bilateral(self, image):
        if self.verbose:
            print("Applying Bilateral Filter")
        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

    def _nlm(self, image):
        if self.verbose:
            print("Applying Non-Local Means")

        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                self.nlm_h,
                self.nlm_template,
                self.nlm_search,
            )
        else:
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                self.nlm_h,
                self.nlm_h,
                self.nlm_template,
                self.nlm_search,
            )

    def _wavelet_denoise(self, image):
        if self.verbose:
            print("Applying Wavelet Denoising")

        img = np.float32(image) / 255.0

        if len(img.shape) == 3:
            channels = cv2.split(img)
            denoised_channels = [
                self._wavelet_channel(c) for c in channels
            ]
            result = cv2.merge(denoised_channels)
        else:
            result = self._wavelet_channel(img)

        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def _wavelet_channel(self, channel):
        coeffs = pywt.wavedec2(channel, self.wavelet, level=self.wavelet_level)

        new_coeffs = [coeffs[0]]

        for detail_level in coeffs[1:]:
            new_level = []
            for arr in detail_level:
                arr = pywt.threshold(
                    arr,
                    self.wavelet_thresh,
                    mode="soft"
                )
                new_level.append(arr)
            new_coeffs.append(tuple(new_level))

        return pywt.waverec2(new_coeffs, self.wavelet)

    # -------------------------
    # PUBLIC API
    # -------------------------

    def apply(self, image):
        """
        Apply selected denoising method.
        """
        self._validate_image(image)

        if self.method == "median":
            return self._median(image)

        elif self.method == "bilateral":
            return self._bilateral(image)

        elif self.method == "nlm":
            return self._nlm(image)

        elif self.method == "wavelet":
            return self._wavelet_denoise(image)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def apply_batch(self, images):
        """
        Apply denoising to multiple images.
        """
        results = []

        for i, image in enumerate(images):
            if self.verbose:
                print(f"\nImage {i+1} — Denoising")

            result = self.apply(image)
            results.append(result)

        return results

    # -------------------------
    # COMPARISON (NOTEBOOK)
    # -------------------------

    def compare_methods(self, image):
        """
        Return all methods for visual comparison.
        """
        self._validate_image(image)

        return {
            "original": image,
            "median": self._median(image),
            "bilateral": self._bilateral(image),
            "nlm": self._nlm(image),
            "wavelet": self._wavelet_denoise(image),
        }
