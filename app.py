import os
import uuid
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash

from pipeline import ScratchDetector, Denoiser, Inpainter, Enhancer, Colorizer

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "old-photo-restoration-dip"

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
ALLOWED_EXT   = {"jpg", "jpeg", "png", "tif", "tiff"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared pipeline objects (stateless, safe to reuse across requests)
# ---------------------------------------------------------------------------
denoiser = Denoiser(
    median_kernel=5,
    nlm_h=10,
    nlm_template=7,
    nlm_search=21,
)

detector = ScratchDetector(
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
    verbose=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


colorizer = Colorizer(sepia_sat_min=15, sepia_sat_max=60,
                      sepia_hue_min=10, sepia_hue_max=30)


def restore_image(image):
    """
    Run the full five-stage restoration pipeline.

    Stage 1 -- Denoising        : NLM removes film grain
    Stage 2 -- Scratch Detection: 4-direction top-hat locates damage
    Stage 3 -- Inpainting       : Navier-Stokes + post-blend fills damage
    Stage 4 -- Enhancement      : CLAHE improves local contrast
    Stage 5 -- Color Correction : fix sepia cast or sharpen grayscale
    """
    denoised          = denoiser.nlm_filter(image)
    mask              = detector.detect(denoised)
    inpainted         = Inpainter(denoised, mask).navier_stokes_blend(radius=7, blend_h=5)
    enhanced          = Enhancer(inpainted).apply_clahe(clip_limit=2.0, tile_grid_size=(8, 8))
    corrected, kind   = colorizer.apply(enhanced)
    return corrected, mask, kind


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/restore", methods=["POST"])
def restore():
    if "image" not in request.files:
        flash("No file selected.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG, PNG, or TIFF image.")
        return redirect(url_for("index"))

    # Save upload with a unique prefix to avoid collisions
    uid         = uuid.uuid4().hex[:8]
    ext         = file.filename.rsplit(".", 1)[1].lower()
    filename    = f"{uid}_original.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Read with OpenCV
    image = cv2.imread(upload_path)
    if image is None:
        flash("Could not read the image. Please try a different file.")
        return redirect(url_for("index"))

    # Run the pipeline
    restored, mask, color_kind = restore_image(image)

    # Save outputs
    mask_filename     = f"{uid}_mask.jpg"
    restored_filename = f"{uid}_restored.jpg"
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, mask_filename),     mask)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, restored_filename), restored)

    context = {
        "original_url" : url_for("static", filename=f"uploads/{filename}"),
        "mask_url"     : url_for("static", filename=f"outputs/{mask_filename}"),
        "restored_url" : url_for("static", filename=f"outputs/{restored_filename}"),
        "color_kind"   : color_kind,
        "filename"     : file.filename,
    }
    return render_template("result.html", **context)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
