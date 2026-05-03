import cv2
import numpy as np


class ScratchDetector:
    """
    Scratch/crack detector for old photo restoration.

    This class detects thin bright scratches using:
    - morphological top-hat transform
    - horizontal, vertical, and diagonal line kernels
    - adaptive thresholding
    - connected component filtering
    - safety check to avoid huge masks
    """

    def __init__(
        self,
        kernel_length=23,
        bright_cutoff=25,
        adaptive_factor=0.50,
        min_scratch_len=10,
        min_aspect_ratio=2.0,
        dilation_iter=1,
        max_mask_ratio=0.08,
        max_component_area_ratio=0.015,
        min_threshold=25,
        density_threshold=0.20,
        verbose=True,
    ):
        self.kernel_length = kernel_length
        self.bright_cutoff = bright_cutoff
        self.adaptive_factor = adaptive_factor
        self.min_scratch_len = min_scratch_len
        self.min_aspect_ratio = min_aspect_ratio
        self.dilation_iter = dilation_iter
        self.max_mask_ratio = max_mask_ratio
        self.max_component_area_ratio = max_component_area_ratio
        self.min_threshold = min_threshold
        self.density_threshold = density_threshold
        self.verbose = verbose

    def _to_gray(self, image):
        """
        Convert image to grayscale if it is BGR/color.
        """
        if image is None:
            raise ValueError("Input image is None.")

        if len(image.shape) == 2:
            return image.copy()

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def create_diagonal_kernel(self, length=None, direction="main"):
        """
        Create diagonal structuring element.

        direction='main' detects diagonal lines like \
        direction='anti' detects diagonal lines like /
        """
        if length is None:
            length = self.kernel_length

        kernel = np.zeros((length, length), dtype=np.uint8)

        if direction == "main":
            np.fill_diagonal(kernel, 1)
        elif direction == "anti":
            np.fill_diagonal(np.fliplr(kernel), 1)
        else:
            raise ValueError("direction must be either 'main' or 'anti'.")

        return kernel

    def compute_tophat_response(self, gray):
        """
        Apply top-hat transform using horizontal, vertical,
        and diagonal kernels, then combine their responses.
        """
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_length, 1)
        )

        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, self.kernel_length)
        )

        d1_kernel = self.create_diagonal_kernel(
            self.kernel_length,
            direction="main"
        )

        d2_kernel = self.create_diagonal_kernel(
            self.kernel_length,
            direction="anti"
        )

        tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, h_kernel)
        tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, v_kernel)
        tophat_d1 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, d1_kernel)
        tophat_d2 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, d2_kernel)

        combined = cv2.bitwise_or(tophat_h, tophat_v)
        combined = cv2.bitwise_or(combined, tophat_d1)
        combined = cv2.bitwise_or(combined, tophat_d2)

        return combined

    def threshold_response(self, combined, gray):
        """
        Convert top-hat response into a binary mask.
        """
        _, bright_mask = cv2.threshold(
            gray,
            self.bright_cutoff,
            255,
            cv2.THRESH_BINARY
        )

        p97 = np.percentile(combined, 97)
        adaptive_thresh = max(
            self.min_threshold,
            int(p97 * self.adaptive_factor)
        )

        _, mask = cv2.threshold(
            combined,
            adaptive_thresh,
            255,
            cv2.THRESH_BINARY
        )

        # Remove detections in very dark regions
        mask = cv2.bitwise_and(mask, bright_mask)

        return mask, adaptive_thresh

    def filter_components(self, mask):
        """
        Keep only scratch-like connected components.

        A valid scratch component should be:
        - long enough
        - line-like or sparse
        - not too huge
        - not too tiny
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        filtered = np.zeros_like(mask)

        total_pixels = mask.shape[0] * mask.shape[1]
        max_component_area = int(self.max_component_area_ratio * total_pixels)

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]

            max_dim = max(w, h)
            min_dim = max(min(w, h), 1)
            aspect_ratio = max_dim / min_dim

            bbox_area = max(w * h, 1)
            density = area / bbox_area

            long_enough = max_dim >= self.min_scratch_len
            line_like = (
                aspect_ratio >= self.min_aspect_ratio
                or density <= self.density_threshold
            )
            not_huge = area <= max_component_area
            not_tiny = area >= 4

            if long_enough and line_like and not_huge and not_tiny:
                filtered[labels == lbl] = 255

        return filtered

    def dilate_mask(self, mask):
        """
        Slightly expand the mask so inpainting covers the full scratch width.
        """
        if self.dilation_iter <= 0:
            return mask

        cross_kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3)
        )

        return cv2.dilate(
            mask,
            cross_kernel,
            iterations=self.dilation_iter
        )

    def safety_check(self, mask):
        """
        If the mask is too large, reject it to avoid destroying the image
        during inpainting.
        """
        total_pixels = mask.shape[0] * mask.shape[1]
        scratch_pixels = cv2.countNonZero(mask)
        scratch_ratio = scratch_pixels / total_pixels

        if scratch_ratio > self.max_mask_ratio:
            if self.verbose:
                print(
                    f"Mask too large ({scratch_ratio * 100:.2f}%). "
                    "Using empty mask to avoid image damage."
                )

            mask = np.zeros_like(mask)
            scratch_pixels = 0
            scratch_ratio = 0.0

        return mask, scratch_pixels, scratch_ratio

    def detect(self, image, return_debug=False):
        """
        Detect scratches in one image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR or grayscale format.
        return_debug : bool
            If True, returns mask and debug dictionary.

        Returns
        -------
        mask : np.ndarray
            Binary scratch mask.
        debug : dict, optional
            Includes top-hat response, threshold, and scratch ratio.
        """
        gray = self._to_gray(image)

        combined = self.compute_tophat_response(gray)
        raw_mask, adaptive_thresh = self.threshold_response(combined, gray)
        filtered_mask = self.filter_components(raw_mask)
        dilated_mask = self.dilate_mask(filtered_mask)

        final_mask, scratch_pixels, scratch_ratio = self.safety_check(
            dilated_mask
        )

        debug = {
            "gray": gray,
            "tophat_response": combined,
            "raw_mask": raw_mask,
            "filtered_mask": filtered_mask,
            "adaptive_threshold": adaptive_thresh,
            "scratch_pixels": scratch_pixels,
            "scratch_ratio": scratch_ratio,
        }

        if self.verbose:
            print(
                f"Scratch pixels: {scratch_pixels} "
                f"({scratch_ratio * 100:.2f}%) | "
                f"threshold used: {adaptive_thresh}"
            )

        if return_debug:
            return final_mask, debug

        return final_mask

    def detect_batch(self, images, return_debug=False):
        """
        Detect scratches for multiple images.
        """
        masks = []
        debug_info = []

        for i, image in enumerate(images):
            if self.verbose:
                print(f"\nImage {i + 1} — Scratch Detection")

            if return_debug:
                mask, debug = self.detect(image, return_debug=True)
                masks.append(mask)
                debug_info.append(debug)
            else:
                mask = self.detect(image, return_debug=False)
                masks.append(mask)

        if return_debug:
            return masks, debug_info

        return masks

    def overlay_mask(self, image, mask, color=(220, 50, 50)):
        """
        Create an RGB visualization with scratches highlighted in red.
        """
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

        overlay[mask == 255] = color

        return overlay