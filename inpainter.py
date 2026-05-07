import cv2

class Inpainter:
    
    
    def __init__(self, image, mask):
        
        self.image = image
        self.mask = mask

    def telea(self, radius=3):

        return cv2.inpaint(self.image, self.mask, radius, cv2.INPAINT_TELEA)

    def navier_stokes(self, radius=3):
        
        return cv2.inpaint(self.image, self.mask, radius, cv2.INPAINT_NS)
