class Colorizer:
    pass

import cv2
import numpy as np

def detect_sepia(image):
    """
    Detects if an image is sepia by checking if the R and G channels 
    are significantly higher than the B channel in a specific ratio.
    """
    # Convert to YCrCb to analyze chromaticity
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    avg_cr = np.mean(ycrcb[:, :, 1])
    avg_cb = np.mean(ycrcb[:, :, 2])
    
    # Sepia tones usually have high Cr (red-shift) and low Cb (blue-deficiency)
    if avg_cr > 145 and avg_cb < 110:
        return True
    return False

def grayscale_cleanup(image):
    """
    Enhances grayscale images using Histogram Equalization 
    to improve contrast and remove 'muddiness'.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) for better cleanup
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cleaned = clahe.apply(gray)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def color_correction(image):
    """
    Applies Simple White Balance (Gray World Hypothesis) 
    to correct color casts.
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    
    # Shift channels to center them around 128
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# Example Usage
img = cv2.imread('assets\gallery_originals\img1.jpg')
if detect_sepia(img):
    print("Sepia detected.")
    
cleaned = grayscale_cleanup(img)
corrected = color_correction(img)

cv2.imwrite('corrected_output.jpg', corrected)



#testing the functions
def test_pipeline():
    print("Starting automated tests...")
    
    # Test Sepia Detection
    sepia_sample = np.full((100, 100, 3), (110, 150, 220), dtype=np.uint8)
    assert detect_sepia(sepia_sample) == True, "Sepia detection failed!"

    # Test Grayscale Cleanup
    low_contrast = np.full((100, 100, 3), 120, dtype=np.uint8)
    low_contrast[0,0] = 125 
    cleaned = grayscale_cleanup(low_contrast)
    assert np.std(cleaned) >= np.std(low_contrast), "Cleanup did not improve contrast"

    # Test Color Correction
    tinted = np.zeros((100, 100, 3), dtype=np.uint8)
    tinted[:,:,0] = 200 
    corrected = color_correction(tinted)
    assert np.mean(corrected[:,:,0]) < 200, "Color correction failed to reduce tint"

    print("✅ All internal tests passed!")

# --- 3. Execution Logic ---

if __name__ == "__main__":
    # Running the automated tests first
    test_pipeline()
    
    # After tests pass, you can process your actual 10 images
    # img = cv2.imread('your_image.jpg')
    # ... process and save results ...