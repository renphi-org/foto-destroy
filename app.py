import io
import json
import math
import os
import random
import threading

import numpy as np
from flask import Flask, render_template, send_file
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from glitch_this import ImageGlitcher

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "images", "source.png")
STATE_PATH = os.path.join(BASE_DIR, "state.json")
MAX_VIEWS = 1024
SAVE_INTERVAL = 10

# --- Global State ---
# We store image data as float32 numpy array to avoid 8-bit rounding losses.
# Only converted to uint8 PIL Image when serving, saving, or applying PIL filters.
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
    """Load source.png from disk or generate a placeholder. Returns float32 array."""
    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH).convert("RGB")
    else:
        img = generate_placeholder()
        os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
        img.save(IMAGE_PATH, format="PNG")
    return np.array(img, dtype=np.float32)


def save_image(arr):
    """Overwrite source.png with the current degraded image."""
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img.save(IMAGE_PATH, format="PNG")


def arr_to_pil(arr):
    """Convert float32 array to PIL Image."""
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def pil_to_arr(img):
    """Convert PIL Image to float32 array."""
    return np.array(img.convert("RGB"), dtype=np.float32)


def degrade(arr, count):
    """Apply one step of degradation to a float32 numpy array. Returns new float32 array."""
    if count >= MAX_VIEWS:
        return np.zeros_like(arr)

    progress = count / MAX_VIEWS  # 0.0 to 1.0

    # 1. Darken: multiply by a factor that smoothly approaches 0
    alpha = 0.001 + 0.007 * (progress ** 2)
    arr = arr * (1.0 - alpha)

    # 2. Add pixel noise every 8 views (destructive — randomly zero out pixels)
    if count % 8 == 0 and progress > 0.05:
        # Randomly zero out a fraction of pixels (dead pixels effect)
        kill_prob = progress * 0.003
        mask = np.random.random(arr.shape[:2]) > kill_prob
        arr *= mask[:, :, np.newaxis]

        # Also add subtle color distortion (shift channels differently)
        if progress > 0.15:
            shift = int(progress * 3)
            if shift > 0:
                channel = random.randint(0, 2)
                arr[:, shift:, channel] = arr[:, :-shift, channel]

    # 3. Apply Pillow filter every ~120 views (structural degradation)
    if count % 120 == 0 and count > 0:
        pre_mean = arr.mean()
        img = arr_to_pil(arr)
        if progress < 0.6:
            img = img.filter(random.choice(GENTLE_FILTERS))
        else:
            img = img.filter(random.choice(HARSH_FILTERS))
        arr = pil_to_arr(img)
        # Ensure filter didn't brighten the image
        post_mean = arr.mean()
        if post_mean > pre_mean and pre_mean > 0:
            arr = arr * (pre_mean / post_mean)

    # 4. Apply glitch effect every ~150 views
    if count % 150 == 0 and count > 0 and progress < 0.85:
        pre_mean = arr.mean()
        intensity = min(10.0, 0.5 + progress * 8.0)
        use_scan_lines = progress > 0.5
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
            # Normalize brightness to not exceed pre-glitch level
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
    with state_lock:
        count = view_count
    return render_template("index.html", view_count=count)


@app.route("/image")
def serve_image():
    global current_image_f, view_count

    with state_lock:
        view_count += 1
        current_image_f = degrade(current_image_f, view_count)

        if view_count % SAVE_INTERVAL == 0:
            save_image(current_image_f)
            save_state(view_count)

        buf = image_to_bytes(current_image_f)

    response = send_file(buf, mimetype="image/png", download_name="image.png")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# --- Startup ---
def init():
    global current_image_f, image_size, view_count
    view_count = load_state()
    current_image_f = load_image()
    image_size = (current_image_f.shape[1], current_image_f.shape[0])
    print(f"Loaded state: view_count={view_count}")
    print(f"Image size: {image_size}")


init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5001)
