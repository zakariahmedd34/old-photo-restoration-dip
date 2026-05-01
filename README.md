# Old Photo Restoration — Digital Image Processing

A modular Python pipeline for restoring old and damaged photographs using classical Digital Image Processing (DIP) techniques.

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `ScratchDetector` | Detects scratches and damaged regions → produces a binary mask |
| 2 | `Denoiser` | Removes film grain and noise |
| 3 | `Inpainter` | Fills in damaged regions using the mask |
| 4 | `Enhancer` | Improves contrast and sharpness |
| 5 | `Colorizer` | Adds colour to grayscale photographs |

---

## Project Structure

```
old-photo-restoration-dip/
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── pipeline/
│   ├── __init__.py
│   ├── scratch_detector.py
│   ├── denoiser.py
│   ├── inpainter.py
│   ├── enhancer.py
│   └── colorizer.py
├── notebook/
│   └── old_photo_restoration_pipeline.ipynb
├── assets/
│   ├── test_images/         # Input images for testing
│   └── gallery_originals/   # Original reference photos
└── outputs/                 # Restored images saved here
```

## Course

CSCI 451 — Digital Image Processing
