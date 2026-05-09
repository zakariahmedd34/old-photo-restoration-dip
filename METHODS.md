# Methods Reference — Old Photo Restoration

**Course:** Digital Image Processing  
**Purpose:** This document explains every decision made in the pipeline —
which method was chosen at each step, which alternatives were tested,
and the final agreed parameters. Single source of truth for the whole team.

---

## Pipeline Overview

```
Original Image
     |
     v
Step 1 -- Denoising          ->  Non-Local Means (NLM)
     |
     v
Step 2 -- Scratch Detection  ->  4-direction Top-Hat + shape filter -> binary mask
     |
     v
Step 3 -- Inpainting         ->  Navier-Stokes + post-NLM blend
     |
     v
Step 4 -- Contrast           ->  CLAHE on L channel (LAB space)
     |
     v
Step 5 -- Color Correction   ->  Auto-classify -> White Balance or Unsharp Mask
     |
     v
Final Restored Image
```

Each step is a separate class in `pipeline/`.
The notebook (`notebook/old_photo_restoration_pipeline.ipynb`) shows all
alternatives side-by-side and documents how parameters were chosen.

---

## Step 1 — Denoising

**File:** `pipeline/denoiser.py`  
**Method chosen:** Non-Local Means (NLM)  
**Call:** `Denoiser(...).nlm_filter(image)`

### Why NLM?

Old photos have random grain from the film or scanner. We must remove it
**before** scratch detection — otherwise noise pixels look like scratches.

NLM searches the whole image for similar patches and averages them.
This preserves edges and skin texture better than a simple blur.

### Methods compared

| Method | How it works | Result |
|--------|-------------|--------|
| **Median Filter** | Replaces pixel with median of neighbors | Good for salt-and-pepper noise |
| **Non-Local Means (NLM)** ✓ | Averages similar patches across image | Best — keeps edges and texture |
| Bilateral Filter | Edge-preserving Gaussian | Similar to NLM but less detail |
| FFT Low-Pass | Frequency-domain smoothing | Too smooth, loses texture |

Median is kept as `denoiser.median_filter(image)` for comparison.

### Final parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `nlm_h` | 10 | Filter strength. h=5 is mild, h=15 is aggressive. |
| `nlm_template` | 7 | Patch size for comparison (always odd) |
| `nlm_search` | 21 | Search radius for similar patches |

---

## Step 2 — Scratch Detection

**File:** `pipeline/scratch_detector.py`  
**Method chosen:** Morphological Top-Hat at 4 orientations + shape filter  
**Call:** `ScratchDetector(...).detect(image)` → binary mask (white=damaged)

### How it works

Scratches are thin, bright lines. The top-hat transform highlights features
that are brighter than their surroundings and narrower than the kernel.

1. Run top-hat with kernels at **4 directions** (H, V, diagonal \, diagonal /)
2. Combine all 4 results with OR — catches scratches at any angle
3. Threshold: `max(25, p97 * 0.50)` — adapts to image brightness
4. **Shape filter**: keep only connected components that are long (≥10 px)
   AND elongated (max/min ≥ 2) OR sparse (area/bbox < 0.20)
5. **Safety cap**: if mask covers > 8% of the image, discard it entirely
   to prevent the pipeline from destroying the photo
6. Dilate by 1 pixel so inpainting covers the full scratch edge

### Why 4 directions?

Version 1 of the notebook used only H and V kernels. Diagonal kernels
were added in v3 after finding that many cracks run diagonally.
Testing on the full dataset confirmed they improved detection.

### Why adaptive threshold?

The 97th-percentile approach (instead of a fixed value) means the same
parameters work on bright photos and dark photos without manual tuning.

### Final parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `kernel_length` | 23 | Line detector length |
| `bright_cutoff` | 25 | Ignore pixels darker than this |
| `adaptive_factor` | 0.50 | Threshold = max(25, p97 * 0.50) |
| `min_scratch_len` | 10 | Minimum component area to keep |
| `min_aspect_ratio` | 2.0 | Max / min bounding-box ratio |
| `density_threshold` | 0.20 | area/bbox — lines are sparse |
| `max_mask_ratio` | 0.08 | Safety: reject mask if > 8% of pixels |
| `dilation_iter` | 1 | Expand mask 1 pixel outward |

---

## Step 3 — Inpainting

**File:** `pipeline/inpainter.py`  
**Method chosen:** Navier-Stokes + post-NLM blend  
**Call:** `Inpainter(image, mask).navier_stokes_blend(radius=7, blend_h=5)`

### Why Navier-Stokes instead of TELEA?

Both methods fill the masked pixels from the surrounding context.

| Method | How it works | Best for |
|--------|-------------|----------|
| **TELEA** | Fast Marching — fills inward from mask edge | Thin sharp scratches |
| **Navier-Stokes (NS)** ✓ | Continues intensity lines smoothly | Wider damage, complex edges |

After testing on the full dataset, NS gives smoother gradients across
wider cracks. Version 4 of the notebook showed this clearly.

### Why a post-NLM blend?

After NS inpainting, the repaired area can look slightly different from
the surrounding texture. A light NLM pass (h=5) over the result blends
the seam without softening the whole image.
This idea was introduced in version 4 of the notebook.

### Final parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `radius` | 7 | NS neighborhood radius. Larger = smoother, slower. |
| `blend_h` | 5 | NLM strength for post-blend. Keep 3-5 to avoid over-smoothing. |

Both `telea()` and `navier_stokes()` methods are still available in the
module for comparison in the notebook.

---

## Step 4 — Contrast Enhancement

**File:** `pipeline/enhancer.py`  
**Method chosen:** CLAHE on L channel (LAB color space)  
**Call:** `Enhancer(image).apply_clahe(clip_limit=2.0, tile_grid_size=(8, 8))`

### How it works

Old photos look flat because the tonal range has compressed. CLAHE
(Contrast Limited Adaptive Histogram Equalization) improves local contrast:

1. Split image into tiles (8x8)
2. Equalize histogram independently per tile
3. Clip to prevent noise amplification
4. Blend tile borders

### Why LAB color space?

Applying CLAHE directly to BGR channels would shift colors. Instead:
1. Convert BGR to LAB (L=lightness, A and B=color)
2. Apply CLAHE only to L — brightness changes, colors unchanged
3. Convert back to BGR

### clip_limit comparison

| clip | Result |
|------|--------|
| 1.0 | Too subtle — barely visible |
| **2.0** ✓ | Balanced — clearly better contrast without artifacts |
| 4.0 | Too aggressive — looks unnatural on faces |

### Final parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `clip_limit` | 2.0 | Contrast strength per tile |
| `tile_grid_size` | (8, 8) | Size of each local region |

---

## Step 5 — Color Correction

**File:** `pipeline/colorizer.py`  
**Method chosen:** Auto-classify + correction per type  
**Call:** `Colorizer(...).apply(image)` → returns `(corrected, kind)`

### Classification

Uses mean HSV saturation and hue to classify automatically:

| Type | Condition | Correction |
|------|-----------|-----------|
| grayscale | mean saturation < 15 | Unsharp mask + histogram stretch |
| sepia | sat 15-60 AND hue 10-30 | White balance correction |
| color | anything else | No change |

### Sepia correction — Gray World White Balance

Scale each channel so all three (B, G, R) have the same mean:
```
channel = channel * (global_mean / channel_mean)
```
This neutralizes the yellow-orange cast without changing the scene.

### Grayscale correction — Unsharp Mask + Histogram Stretch

```
blurred   = GaussianBlur(gray, sigma=0.5)
sharpened = gray * 2.5 - blurred * 1.5     <- aggressive unsharp
stretched = normalize(sharpened, 0, 255)
```

The 2.5/-1.5 coefficients (stronger than the traditional 1.5/-0.5)
were chosen from version 4 of the notebook. They recover texture
lost during NLM denoising without making the image look plastic.

### Final parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sepia_sat_min` | 15 | Below this -> grayscale |
| `sepia_sat_max` | 60 | Above this -> real color |
| `sepia_hue_min` | 10 | Yellow-orange band start |
| `sepia_hue_max` | 30 | Yellow-orange band end |

---

## How the Versions Were Compared

We had 4 notebook versions (v1, v2, v3, v4). Here is what each contributed:

| Feature | Source version |
|---------|---------------|
| NLM denoising | v1 (all agreed) |
| H+V top-hat kernels | v1 |
| Diagonal kernels (D1, D2) | v3 |
| Adaptive threshold (p97) | v1, v3 |
| Density filter (area/bbox) | v3 |
| Safety mask cap (8%) | v3 |
| Navier-Stokes inpainting | v4 |
| Post-inpainting NLM blend | v4 |
| CLAHE clip=2.0, tile=8x8 | v1 (v4 used clip=0.5 — too conservative) |
| Color classification thresholds | v1 |
| Grayscale unsharp 2.5/-1.5 | v4 (v1 used 1.5/-0.5 — too weak) |

---

## Module Usage Reference

```python
from pipeline import ScratchDetector, Denoiser, Inpainter, Enhancer, Colorizer

# Stage 1 — Denoising
denoiser  = Denoiser(nlm_h=10, nlm_template=7, nlm_search=21)
denoised  = denoiser.nlm_filter(image)

# Stage 2 — Scratch Detection
detector  = ScratchDetector(kernel_length=23, adaptive_factor=0.50,
                            max_mask_ratio=0.08)
mask      = detector.detect(denoised)

# Stage 3 — Inpainting
inpainted = Inpainter(denoised, mask).navier_stokes_blend(radius=7, blend_h=5)

# Stage 4 — Enhancement
enhanced  = Enhancer(inpainted).apply_clahe(clip_limit=2.0, tile_grid_size=(8, 8))

# Stage 5 — Color Correction
colorizer            = Colorizer(sepia_sat_min=15, sepia_sat_max=60,
                                 sepia_hue_min=10, sepia_hue_max=30)
corrected, kind      = colorizer.apply(enhanced)
# kind is 'grayscale', 'sepia', or 'color'
```

---

> If you change any parameter, update the table in **Step 8 of the notebook**
> and tell the whole team before changing `main.py`.
