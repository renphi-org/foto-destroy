import io
import json
import math
import os
import random
import shutil
import threading

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from glitch_this import ImageGlitcher

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "images", "source.png")
ORIGINAL_PATH = os.path.join(BASE_DIR, "images", "original.png")
DEFAULT_PATH = os.path.join(BASE_DIR, "images", "default.png")  # committed seed image
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "images", "snapshots")
STATE_PATH = os.path.join(BASE_DIR, "state.json")
MAX_VIEWS = 1024
SAVE_INTERVAL = 10

# --- Global State ---
state_lock = threading.Lock()
current_image_f: np.ndarray = None  # float32 array, shape (H, W, 3), range [0, 255]
image_size: tuple = None
view_count: int = 0
glitcher = ImageGlitcher()

# --- Pillow Filters ---
GENTLE_FILTERS = [
    ImageFilter.GaussianBlur(radius=1),
    ImageFilter.SMOOTH_MORE,
    ImageFilter.MedianFilter(size=3),
]
HARSH_FILTERS = [
    ImageFilter.GaussianBlur(radius=3),
    ImageFilter.EMBOSS,
    ImageFilter.EDGE_ENHANCE_MORE,
    ImageFilter.MedianFilter(size=5),
]


def generate_placeholder(width=800, height=600):
    """Generate a colorful gradient image as the starting placeholder."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        for x in range(width):
            r = int(255 * (x / width))
            g = int(255 * (y / height))
            b = int(255 * abs(math.sin(x * 0.02) * math.cos(y * 0.02)))
            draw.point((x, y), fill=(r, g, b))
    return img


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
            return data.get("view_count", 0)
    return 0


def save_state(count):
    with open(STATE_PATH, "w") as f:
        json.dump({"view_count": count}, f)


def load_image():
    """Load source.png from disk, falling back to default.png or placeholder. Returns float32 array."""
    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH).convert("RGB")
    elif os.path.exists(DEFAULT_PATH):
        img = Image.open(DEFAULT_PATH).convert("RGB")
        os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
        img.save(IMAGE_PATH, format="PNG")
    else:
        img = generate_placeholder()
        os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
        img.save(IMAGE_PATH, format="PNG")
    return np.array(img, dtype=np.float32)


def save_image(arr):
    """Overwrite source.png with the current degraded image."""
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img.save(IMAGE_PATH, format="PNG")


def save_snapshot(arr, view_num):
    """Save a snapshot image for the given view number."""
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOTS_DIR, f"snap_{view_num:04d}.png")
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img.save(path, format="PNG")


def ensure_original():
    """Ensure original.png exists (the pristine, never-overwritten copy)."""
    if not os.path.exists(ORIGINAL_PATH):
        if os.path.exists(IMAGE_PATH):
            shutil.copy2(IMAGE_PATH, ORIGINAL_PATH)
        elif os.path.exists(DEFAULT_PATH):
            shutil.copy2(DEFAULT_PATH, ORIGINAL_PATH)
        else:
            img = generate_placeholder()
            os.makedirs(os.path.dirname(ORIGINAL_PATH), exist_ok=True)
            img.save(ORIGINAL_PATH, format="PNG")
            img.save(IMAGE_PATH, format="PNG")


def get_snapshot_list():
    """Return sorted list of available snapshot view numbers."""
    if not os.path.exists(SNAPSHOTS_DIR):
        return []
    nums = []
    for f in os.listdir(SNAPSHOTS_DIR):
        if f.startswith("snap_") and f.endswith(".png"):
            try:
                num = int(f[5:9])
                nums.append(num)
            except ValueError:
                pass
    return sorted(nums)


def arr_to_pil(arr):
    """Convert float32 array to PIL Image."""
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def pil_to_arr(img):
    """Convert PIL Image to float32 array."""
    return np.array(img.convert("RGB"), dtype=np.float32)


def jpeg_corrupt(arr, quality):
    """Run image through JPEG compression to create blocky artifacts."""
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img, dtype=np.float32)


def rgb_channel_shift(arr, max_shift):
    """Shift RGB channels independently for chromatic aberration."""
    h, w, _ = arr.shape
    result = arr.copy()
    for ch in range(3):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        shifted = np.roll(np.roll(arr[:, :, ch], dx, axis=1), dy, axis=0)
        result[:, :, ch] = shifted
    return result


def block_corrupt(arr, num_blocks):
    """Randomly copy rectangular blocks to wrong positions (datamosh effect)."""
    h, w, _ = arr.shape
    result = arr.copy()
    for _ in range(num_blocks):
        bw = random.randint(20, max(21, w // 4))
        bh = random.randint(5, max(6, h // 8))
        sx, sy = random.randint(0, w - bw), random.randint(0, h - bh)
        dx, dy = random.randint(0, w - bw), random.randint(0, h - bh)
        result[dy:dy + bh, dx:dx + bw] = arr[sy:sy + bh, sx:sx + bw]
    return result


def degrade(arr, count):
    """Apply one step of degradation to a float32 numpy array. Returns new float32 array."""
    if count >= MAX_VIEWS:
        return np.zeros_like(arr)

    progress = count / MAX_VIEWS  # 0.0 to 1.0

    # 1. Darken: multiply by a factor that smoothly approaches 0
    alpha = 0.001 + 0.007 * (progress ** 2)
    arr = arr * (1.0 - alpha)

    # 2. RGB channel shift every 6 views (chromatic aberration)
    if count % 6 == 0 and progress > 0.02:
        max_shift = max(1, int(progress * 12))
        pre_mean = arr.mean()
        arr = rgb_channel_shift(arr, max_shift)
        post_mean = arr.mean()
        if post_mean > pre_mean and pre_mean > 0:
            arr = arr * (pre_mean / post_mean)

    # 3. Dead pixels + color distortion every 8 views
    if count % 8 == 0 and progress > 0.05:
        kill_prob = progress * 0.004
        mask = np.random.random(arr.shape[:2]) > kill_prob
        arr *= mask[:, :, np.newaxis]

    # 4. Block corruption (datamosh) every ~25 views
    if count % 25 == 0 and count > 0 and progress < 0.9:
        num_blocks = max(1, int(progress * 8))
        pre_mean = arr.mean()
        arr = block_corrupt(arr, num_blocks)
        post_mean = arr.mean()
        if post_mean > pre_mean and pre_mean > 0:
            arr = arr * (pre_mean / post_mean)

    # 5. JPEG corruption every ~35 views (blocky artifacts)
    if count % 35 == 0 and count > 0:
        quality = max(1, int(50 - progress * 45))
        pre_mean = arr.mean()
        arr = jpeg_corrupt(arr, quality)
        post_mean = arr.mean()
        if post_mean > pre_mean and pre_mean > 0:
            arr = arr * (pre_mean / post_mean)

    # 6. Apply Pillow filter every ~80 views
    if count % 80 == 0 and count > 0:
        pre_mean = arr.mean()
        img = arr_to_pil(arr)
        if progress < 0.5:
            img = img.filter(random.choice(GENTLE_FILTERS))
        else:
            img = img.filter(random.choice(HARSH_FILTERS))
        arr = pil_to_arr(img)
        post_mean = arr.mean()
        if post_mean > pre_mean and pre_mean > 0:
            arr = arr * (pre_mean / post_mean)

    # 7. Glitch-this effect every ~40 views (frequent and wild)
    if count % 40 == 0 and count > 0 and progress < 0.85:
        pre_mean = arr.mean()
        intensity = min(10.0, 1.0 + progress * 9.0)
        use_scan_lines = progress > 0.3
        try:
            img = arr_to_pil(arr)
            glitched = glitcher.glitch_image(
                img,
                glitch_amount=intensity,
                color_offset=True,
                scan_lines=use_scan_lines,
            )
            glitched = glitched.convert("RGB")
            if glitched.size != img.size:
                glitched = glitched.resize(img.size, Image.LANCZOS)
            arr = pil_to_arr(glitched)
            post_mean = arr.mean()
            if post_mean > pre_mean and pre_mean > 0:
                arr = arr * (pre_mean / post_mean)
        except Exception:
            pass

    # 5. Extra darkening in final stretch (views 850+)
    if count > 850:
        extra = (count - 850) / (MAX_VIEWS - 850) * 0.03
        arr = arr * (1.0 - extra)

    arr = np.clip(arr, 0.0, 255.0)
    return arr


def image_to_bytes(arr):
    """Convert float32 array to PNG bytes."""
    img = arr_to_pil(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# --- Routes ---
@app.route("/")
def index():
    """Each page request = one degradation step. The image is degraded here."""
    global current_image_f, view_count

    with state_lock:
        view_count += 1
        current_image_f = degrade(current_image_f, view_count)

        if view_count % SAVE_INTERVAL == 0:
            save_image(current_image_f)
            save_snapshot(current_image_f, view_count)
            save_state(view_count)

        count = view_count

    response = app.make_response(
        render_template("index.html", view_count=count, max_views=MAX_VIEWS)
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/image")
def serve_image():
    """Serve the current degraded image (read-only, no degradation)."""
    with state_lock:
        buf = image_to_bytes(current_image_f)

    response = send_file(buf, mimetype="image/png", download_name="image.png")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/snapshots")
def list_snapshots():
    """Return JSON list of available snapshot view numbers."""
    return jsonify(get_snapshot_list())


@app.route("/snapshot/<int:view_num>")
def serve_snapshot(view_num):
    """Serve a specific snapshot image."""
    path = os.path.join(SNAPSHOTS_DIR, f"snap_{view_num:04d}.png")
    if not os.path.exists(path):
        return "Snapshot not found", 404
    response = send_file(path, mimetype="image/png")
    response.headers["Cache-Control"] = "public, max-age=31536000"
    return response


@app.route("/reset", methods=["POST"])
def reset():
    """Reset to the original pristine image."""
    global current_image_f, view_count

    with state_lock:
        # Copy original back to source
        if os.path.exists(ORIGINAL_PATH):
            shutil.copy2(ORIGINAL_PATH, IMAGE_PATH)

        # Delete all snapshots
        if os.path.exists(SNAPSHOTS_DIR):
            shutil.rmtree(SNAPSHOTS_DIR)
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

        # Reset state
        view_count = 0
        current_image_f = load_image()
        save_state(0)

    return redirect(url_for("index"))


# --- Startup ---
def init():
    global current_image_f, image_size, view_count
    view_count = load_state()
    current_image_f = load_image()
    image_size = (current_image_f.shape[1], current_image_f.shape[0])

    # Preserve the original pristine image
    ensure_original()

    # Ensure snapshots dir and snap_0000 exist
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    snap_0 = os.path.join(SNAPSHOTS_DIR, "snap_0000.png")
    if not os.path.exists(snap_0):
        orig = Image.open(ORIGINAL_PATH).convert("RGB")
        orig.save(snap_0, format="PNG")

    print(f"Loaded state: view_count={view_count}")
    print(f"Image size: {image_size}")
    print(f"Snapshots available: {get_snapshot_list()}")


init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5001)
