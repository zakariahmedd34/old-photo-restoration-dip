import cv2
import numpy as np


class Denoiser:
    """
    Denoising module for old photo restoration.

    Methods:
    - Median filter
    - Non-local means
    - Bilateral filter
    - Frequency-domain low-pass filter
    """

    def __init__(
        self,
        median_kernel=5,
        nlm_h=10,
        nlm_template=7,
        nlm_search=21,
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        freq_radius=30,
        verbose=True,
    ):

        self.median_kernel = median_kernel

        self.nlm_h = nlm_h
        self.nlm_template = nlm_template
        self.nlm_search = nlm_search

        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

        self.freq_radius = freq_radius

        self.verbose = verbose

    def median_filter(self, image):

        return cv2.medianBlur(
            image,
            self.median_kernel
        )

    def nlm_filter(self, image):

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.nlm_h,
            self.nlm_h,
            self.nlm_template,
            self.nlm_search,
        )

    def bilateral_filter(self, image):

        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

    def frequency_filter(self, image):
        """
        Frequency-domain low-pass filtering using FFT.
        """

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Circular low-pass mask
        mask = np.zeros((rows, cols), np.uint8)

        cv2.circle(
            mask,
            (ccol, crow),
            self.freq_radius,
            1,
            -1
        )

        # Apply mask
        fshift_filtered = fshift * mask

        # Inverse FFT
        ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(ishift)
        img_back = np.abs(img_back)

        img_back = np.clip(img_back, 0, 255).astype(np.uint8)

        # Convert back to BGR for visualization consistency
        freq_bgr = cv2.cvtColor(
            img_back,
            cv2.COLOR_GRAY2BGR
        )

        return freq_bgr

    def process_image(self, image):
        """
        Apply all denoising methods to one image.
        """

        median = self.median_filter(image)

        nlm = self.nlm_filter(image)

        bilateral = self.bilateral_filter(image)

        frequency = self.frequency_filter(image)

        return {
            "median": median,
            "nlm": nlm,
            "bilateral": bilateral,
            "frequency": frequency,
        }

    def process_batch(self, images):
        """
        Apply denoising methods to all images.
        """

        results = []

        for i, image in enumerate(images):

            if self.verbose:
                print(f"Processing image {i+1}")

            outputs = self.process_image(image)

            results.append(outputs)

        return results
