import cv2


class Inpainter:
    """
    Fills in damaged pixels identified by a binary mask.

    Usage
    -----
        result = Inpainter(image, mask).telea(radius=3)
        result = Inpainter(image, mask).navier_stokes(radius=7)
        result = Inpainter(image, mask).navier_stokes_blend(radius=7, blend_h=5)
    """

    def __init__(self, image, mask):
        self.image = image
        self.mask  = mask

    def telea(self, radius=3):
        """
        Fast Marching Method (TELEA).
        Good for thin, sharp scratches.
        """
        return cv2.inpaint(self.image, self.mask, radius, cv2.INPAINT_TELEA)

    def navier_stokes(self, radius=7):
        """
        Navier-Stokes fluid-dynamics inpainting.
        Better than TELEA for wider damage and complex edges.
        """
        return cv2.inpaint(self.image, self.mask, radius, cv2.INPAINT_NS)

    def navier_stokes_blend(self, radius=7, blend_h=5):
        """
        Navier-Stokes inpainting followed by a light NLM pass
        to blend the repaired seams with the surrounding texture.

        Parameters
        ----------
        radius  : inpainting neighborhood radius (default 7)
        blend_h : NLM filter strength for the post-blend (default 5)
                  Keep low (3-5) so the blend smooths seams without
                  softening the whole image.
        """
        restored = cv2.inpaint(self.image, self.mask, radius, cv2.INPAINT_NS)
        blended  = cv2.fastNlMeansDenoisingColored(
            restored, None,
            blend_h, blend_h,
            7, 21,
        )
        return blended
