import cv2

class Enhancer:
  
    def __init__(self, image):
        self.image = image

    def apply_clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        #convert the image from BGR to LAB color space
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        #split the channels so we can work on Lightness (L) alone
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        l_enhanced = clahe.apply(l)
        
        lab_final = cv2.merge((l_enhanced, a, b))
        
        #convert it back to standard BGR color to save it
        result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        
        return result
