import os
import glob
import cv2

from pipeline import ScratchDetector, Denoiser, Inpainter, Enhancer, Colorizer


# ---------------------------------------------------------------------------
# Input / output configuration
# ---------------------------------------------------------------------------
INPUT_DIR = os.path.join("assets", "gallery_originals")
OUTPUT_DIR = "output"
PATTERNS   = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")


# ---------------------------------------------------------------------------
# Pipeline objects  (parameters agreed in Step 8 of the final notebook)
# ---------------------------------------------------------------------------
denoiser  = Denoiser(
    median_kernel=5,
    nlm_h=10,
    nlm_template=7,
    nlm_search=21,
)

detector  = ScratchDetector(
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
)

enhancer  = Enhancer(None)          # image is passed per-call in the loop

colorizer = Colorizer(
    sepia_sat_min=15,
    sepia_sat_max=60,
    sepia_hue_min=10,
    sepia_hue_max=30,
)


def restore(image):
    """
    Run the full five-stage restoration pipeline on one image.

    Stage 1 -- Denoising        : NLM removes grain and sensor noise
    Stage 2 -- Scratch Detection: 4-direction top-hat locates damage
    Stage 3 -- Inpainting       : Navier-Stokes + post-blend fills damage
    Stage 4 -- Enhancement      : CLAHE improves local contrast
    Stage 5 -- Color Correction : fix sepia cast or sharpen grayscale
    """
    # Stage 1
    denoised = denoiser.nlm_filter(image)

    # Stage 2
    mask = detector.detect(denoised)

    # Stage 3 — Navier-Stokes + post-NLM seam blend
    inpainted = Inpainter(denoised, mask).navier_stokes_blend(radius=7, blend_h=5)

    # Stage 4
    enhanced = Enhancer(inpainted).apply_clahe(clip_limit=2.0, tile_grid_size=(8, 8))

    # Stage 5
    corrected, kind = colorizer.apply(enhanced)
    print(f"  color type detected: {kind}")

    return corrected


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = []
    for p in PATTERNS:
        paths.extend(glob.glob(os.path.join(INPUT_DIR, p)))
    paths = sorted(paths)

    if not paths:
        print(f"No images found in {INPUT_DIR}")
        return

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"WARNING: could not load '{path}' -- skipping.")
            continue

        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\nRestoring {name}")
        result = restore(img)

        out_path = os.path.join(OUTPUT_DIR, f"restored_{name}.jpg")
        cv2.imwrite(out_path, result)
        print(f"  saved -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
