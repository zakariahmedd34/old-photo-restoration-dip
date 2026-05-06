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

    # ======================================================
    # Median Filter
    # ======================================================

    def median_filter(self, image):

        return cv2.medianBlur(
            image,
            self.median_kernel
        )

    # ======================================================
    # Non-Local Means
    # ======================================================

    def nlm_filter(self, image):

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.nlm_h,
            self.nlm_h,
            self.nlm_template,
            self.nlm_search,
        )

    # ======================================================
    # Bilateral Filter
    # ======================================================

    def bilateral_filter(self, image):

        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

    # ======================================================
    # Frequency-Domain Low-Pass Filter
    # ======================================================

    def low_pass_channel(self, channel):
        """
        Apply low-pass filtering to one channel.
        """

        f = np.fft.fft2(channel)

        fshift = np.fft.fftshift(f)

        rows, cols = channel.shape

        crow, ccol = rows // 2, cols // 2

        # Circular mask
        mask = np.zeros((rows, cols), np.uint8)

        cv2.circle(
            mask,
            (ccol, crow),
            self.freq_radius,
            1,
            -1
        )

        # Apply mask
        filtered = fshift * mask

        # Inverse FFT
        ishift = np.fft.ifftshift(filtered)

        img_back = np.fft.ifft2(ishift)

        img_back = np.abs(img_back)

        img_back = np.clip(
            img_back,
            0,
            255
        ).astype(np.uint8)

        return img_back

    def frequency_filter(self, image):
        """
        Apply frequency-domain filtering
        separately to B, G, R channels.
        """

        if len(image.shape) == 2:
            return self.low_pass_channel(image)

        # Split BGR channels
        b, g, r = cv2.split(image)

        # Filter each channel
        b_filtered = self.low_pass_channel(b)

        g_filtered = self.low_pass_channel(g)

        r_filtered = self.low_pass_channel(r)

        # Merge back
        merged = cv2.merge(
            [b_filtered, g_filtered, r_filtered]
        )

        return merged

    # ======================================================
    # Process Single Image
    # ======================================================

    def process_image(self, image):

        median = self.median_filter(image)

        bilateral = self.bilateral_filter(image)

        frequency = self.frequency_filter(image)

        nlm = self.nlm_filter(image)

        return {
            "median": median,
            "bilateral": bilateral,
            "frequency": frequency,
            "nlm": nlm,
        }

    # ======================================================
    # Batch Processing
    # ======================================================

    def process_batch(self, images):

        results = []

        for i, image in enumerate(images):

            if self.verbose:
                print(f"Processing image {i+1}")

            outputs = self.process_image(image)

            results.append(outputs)

        return results
