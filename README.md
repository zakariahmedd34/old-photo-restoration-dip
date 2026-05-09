# Old Photo Restoration — Digital Image Processing

A modular Python pipeline for restoring old and damaged photographs using
classical Digital Image Processing (DIP) techniques. No machine learning —
every stage is a well-understood mathematical operation.

---

## Pipeline Stages

```
Original Image
     |
     v
Step 1 -- Denoising          (NLM removes grain and sensor noise)
     |
     v
Step 2 -- Scratch Detection  (4-direction top-hat -> binary mask)
     |
     v
Step 3 -- Inpainting         (Navier-Stokes fills damage + NLM blend)
     |
     v
Step 4 -- Enhancement        (CLAHE improves local contrast)
     |
     v
Step 5 -- Color Correction   (fix sepia cast or sharpen grayscale)
     |
     v
Final Restored Image
```

| Stage | Module | Method |
|-------|--------|--------|
| 1 | `Denoiser` | Non-Local Means (NLM), h=10 |
| 2 | `ScratchDetector` | Morphological top-hat at H, V, D1, D2 orientations |
| 3 | `Inpainter` | Navier-Stokes (radius=7) + post-NLM blend (h=5) |
| 4 | `Enhancer` | CLAHE on L channel (LAB space), clip=2.0, tile=8x8 |
| 5 | `Colorizer` | Auto-detect grayscale / sepia / color, then correct |

---

## Project Structure

```
old-photo-restoration-dip/
|
├── main.py                              # Run the pipeline on all images
├── app.py                               # Flask web demo
├── requirements.txt
|
├── pipeline/
│   ├── __init__.py
│   ├── denoiser.py          Stage 1
│   ├── scratch_detector.py  Stage 2
│   ├── inpainter.py         Stage 3
│   ├── enhancer.py          Stage 4
│   └── colorizer.py         Stage 5
|
├── notebook/
│   └── old_photo_restoration_pipeline.ipynb   <- Final notebook
|
├── assets/
│   └── gallery_originals/   Input images (img1.jpg ... img13.jpg)
|
├── output/                  Restored images saved here
|
├── templates/               Flask HTML templates
└── static/                  Flask CSS and uploaded files
```

---

## Quick Start

### Run on all images

```bash
python main.py
```

Results are saved to `output/restored_<name>.jpg`.

### Run the web demo

```bash
pip install flask
python app.py
```

Open http://localhost:5000 in your browser.

### Run the notebook

Open `notebook/old_photo_restoration_pipeline.ipynb` in Jupyter.
The notebook runs the same pipeline step-by-step with visualizations
and tuning cells for every stage.

---

## Agreed Parameters

These values were decided after testing on the full image set.
See `notebook/` Step 8 for the full reasoning.

| Step | Parameter | Value |
|------|-----------|-------|
| Denoising | h (NLM strength) | 10 |
| Scratch Detection | kernel_length | 23 |
| Scratch Detection | adaptive_factor | 0.50 |
| Scratch Detection | max_mask_ratio | 0.08 (safety cap) |
| Inpainting | radius (NS) | 7 |
| Inpainting | blend_h (post-NLM) | 5 |
| CLAHE | clip_limit | 2.0 |
| CLAHE | tile_grid_size | (8, 8) |
| Color | sepia sat range | 15 - 60 |
| Color | sepia hue range | 10 - 30 |
| Color (grayscale) | unsharp alpha/beta | 2.5 / -1.5 |

---

## Dependencies

```
opencv-python
numpy
matplotlib
flask
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Course

Digital Image Processing — University Project
