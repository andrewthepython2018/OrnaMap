from io import BytesIO
from math import cos, pi, sin
import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
from sklearn.cluster import KMeans

st.set_page_config(page_title="OrnaMap Vision", page_icon="OM", layout="wide")

RED = (228, 31, 49)
BLUE = (29, 68, 154)
INK = (14, 16, 15)
PAPER = (250, 249, 243)

SAMPLE_SOURCES = {
    "s1.jpg": Path("sample/s3.jpg"),
    "s2.jpg": Path("sample/s2.jpg"),
    "s3.jpg": Path("sample/s1.jpg"),
}

st.markdown(
    """
<style>
.stApp {
background:
radial-gradient(circle at 8% 8%, rgba(228,31,49,.13), transparent 24%),
radial-gradient(circle at 94% 10%, rgba(29,68,154,.14), transparent 25%),
linear-gradient(135deg, #f8f2e8 0%, #eef5f3 52%, #fffaf2 100%);
}
.block-container { max-width: 1340px; padding-top: 1.25rem; }
h1 { color: #172b35; font-size: 4.1rem !important; line-height: .96 !important; letter-spacing: 0 !important; }
h2, h3 { color: #172b35; letter-spacing: 0 !important; }
[data-testid="stSidebar"] { background: #f4efe5; border-right: 1px solid rgba(23,43,53,.13); }
div[data-testid="stImage"] img { border-radius: 18px; box-shadow: 0 20px 54px rgba(24,36,42,.15); }
.lead { color: #52646c; max-width: 900px; font-size: 1.05rem; line-height: 1.52; }
.chip { display: inline-block; margin: 0 7px 7px 0; padding: 7px 11px; border-radius: 999px; background: rgba(255,255,255,.68); border: 1px solid rgba(23,43,53,.12); color: #344851; font-size: .92rem; }
.metric { background: rgba(255,255,255,.68); border: 1px solid rgba(23,43,53,.11); border-radius: 16px; padding: 14px 16px; min-height: 84px; }
.metric b { color: #172b35; font-size: 1.3rem; }
.metric span { color: #64747b; font-size: .9rem; }
</style>
""",
    unsafe_allow_html=True,
)


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def rgba(color, alpha=255):
    return int(color[0]), int(color[1]), int(color[2]), alpha


def symbol(draw: ImageDraw.ImageDraw, box, kind: int) -> None:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = 8
    if kind == 0:
        draw.ellipse((cx - 17, y1 + 10, cx + 17, y1 + 44), fill=INK)
        draw.line((cx, y1 + 44, cx, y2 - 34), fill=INK, width=w)
        draw.line((cx, y1 + 72, x1 + 16, cy), fill=INK, width=w)
        draw.line((cx, y1 + 72, x2 - 16, cy + 8), fill=INK, width=w)
        draw.line((cx, y2 - 34, x1 + 32, y2 - 5), fill=INK, width=w)
        draw.line((cx, y2 - 34, x2 - 28, y2 - 5), fill=INK, width=w)
    elif kind == 1:
        draw.ellipse((cx - 14, y1 + 12, cx + 14, y1 + 40), fill=INK)
        draw.polygon((cx, y1 + 42, x1 + 30, y2 - 32, x2 - 30, y2 - 32), fill=INK)
        draw.line((x1 + 18, y1 + 62, x2 - 18, y1 + 54), fill=INK, width=w)
        draw.line((cx - 18, y2 - 32, x1 + 44, y2), fill=INK, width=w)
        draw.line((cx + 18, y2 - 32, x2 - 44, y2), fill=INK, width=w)
    elif kind == 2:
        draw.polygon((cx, y1 + 12, x1 + 28, y2 - 18, x2 - 28, y2 - 18), outline=INK)
        draw.line((cx, y1 + 12, cx, y2 - 18), fill=INK, width=w)
        draw.line((x1 + 38, cy, x2 - 38, cy), fill=INK, width=w)
    elif kind == 3:
        draw.polygon((cx, y1 + 8, x1 + 20, y2 - 18, x2 - 20, y2 - 18), outline=INK)
        draw.arc((x1 + 44, cy - 10, x2 - 44, y2 + 14), 205, 335, fill=INK, width=w)
        draw.line((x1 + 14, y2 - 10, x2 - 14, y2 - 10), fill=INK, width=w)
    elif kind == 4:
        draw.arc((x1 + 12, cy - 30, x2 - 12, y2), 15, 165, fill=INK, width=w)
        draw.line((cx, y1 + 10, cx, cy + 20), fill=INK, width=w)
        draw.line((cx, y1 + 38, x1 + 38, cy), fill=INK, width=w)
        draw.line((cx, y1 + 38, x2 - 38, cy), fill=INK, width=w)
    else:
        draw.line((cx, y1 + 10, cx, y2 - 10), fill=INK, width=w)
        draw.line((x1 + 20, cy, x2 - 20, cy), fill=INK, width=w)
        draw.line((x1 + 32, y1 + 34, x2 - 32, y2 - 34), fill=INK, width=w)
        draw.line((x2 - 32, y1 + 34, x1 + 32, y2 - 34), fill=INK, width=w)


def demo_symbols() -> Image.Image:
    image = Image.new("RGBA", (1000, 620), (*PAPER, 255))
    draw = ImageDraw.Draw(image)
    for index in range(12):
        col, row = index % 4, index // 4
        x = 72 + col * 225
        y = 48 + row * 178
        symbol(draw, (x, y, x + 145, y + 120), index % 6)
    return image


def demo_strip() -> Image.Image:
    image = Image.new("RGBA", (1120, 520), (*PAPER, 255))
    draw = ImageDraw.Draw(image)
    s = 8

    def pix(x, y, cells, color):
        for cx, cy in cells:
            draw.rectangle((x + cx * s, y + cy * s, x + (cx + 1) * s - 1, y + (cy + 1) * s - 1), fill=rgba(color))

    def band(y, step, cells, color):
        for x in range(-step, 1120 + step, step):
            pix(x, y, cells, color)

    band(22, 92, [(0, 2), (1, 2), (2, 2), (3, 2), (1, 1), (2, 0), (3, 1), (6, 2), (7, 2), (8, 2), (9, 2), (7, 1), (8, 0), (9, 1)], BLUE)
    draw.rectangle((0, 61, 1120, 68), fill=rgba(BLUE))
    band(92, 196, [(4, 6), (5, 5), (6, 4), (7, 5), (8, 6), (6, 6), (6, 7), (5, 8), (7, 8), (14, 1), (15, 1), (16, 1), (15, 2), (15, 3), (14, 4), (16, 4), (13, 5), (17, 5), (12, 6), (18, 6), (11, 7), (19, 7), (12, 8), (18, 8), (13, 9), (17, 9), (14, 10), (16, 10)], RED)
    draw.rectangle((0, 180, 1120, 188), fill=rgba(BLUE))
    band(214, 78, [(0, 0), (1, 0), (2, 0), (2, 1), (3, 2), (4, 3), (5, 2), (6, 1), (6, 0), (7, 0), (8, 0)], BLUE)
    band(284, 78, [(0, 8), (1, 7), (2, 6), (3, 5), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (1, 8), (2, 8), (6, 8), (7, 8)], RED)
    draw.rectangle((0, 365, 1120, 373), fill=rgba(BLUE))
    band(394, 214, [(0, 4), (1, 3), (2, 2), (3, 1), (4, 2), (5, 3), (6, 4), (12, 1), (12, 2), (12, 3), (11, 4), (10, 5), (11, 6), (12, 6), (13, 6), (14, 5), (13, 4), (19, 4), (20, 3), (21, 2), (22, 1), (23, 2), (24, 3), (25, 4)], RED)
    return image


def load_source(uploaded, mode: str) -> Image.Image:
    if uploaded:
        image = Image.open(uploaded).convert("RGBA")
    elif mode in SAMPLE_SOURCES and SAMPLE_SOURCES[mode].exists():
        image = Image.open(SAMPLE_SOURCES[mode]).convert("RGBA")
    elif mode == "s1.jpg":
        image = demo_strip()
    else:
        image = demo_symbols()
    scale = max(image.size) / 1200
    if scale > 1:
        image = image.resize((int(image.width / scale), int(image.height / scale)), Image.Resampling.LANCZOS)
    return image


def make_mask(image: Image.Image, sensitivity: int, close_size: int) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = ((gray < sensitivity) | ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 252))).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((close_size, close_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)


def extract_elements(image: Image.Image, mask: np.ndarray, min_area: int, padding: int) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elements = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 16 or h < 16 or (w > image.width * .94 and h > image.height * .94):
            continue
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(image.width, x + w + padding), min(image.height, y + h + padding)
        crop = image.crop((x1, y1, x2, y2)).convert("RGBA")
        alpha = Image.fromarray(mask[y1:y2, x1:x2], "L").filter(ImageFilter.GaussianBlur(.7))
        crop.putalpha(alpha)
        elements.append({"image": crop, "box": (x1, y1, x2, y2), "area": area})
    elements.sort(key=lambda item: (item["box"][1], item["box"][0]))
    return elements[:48]


def palette(image: Image.Image, mask: np.ndarray, count: int = 5) -> list[tuple[int, int, int]]:
    rgb = np.array(image.convert("RGB"))
    pixels = rgb[mask > 0]
    if len(pixels) == 0:
        pixels = rgb.reshape(-1, 3)
    pixels = pixels[(pixels.max(axis=1) < 238) | (np.ptp(pixels, axis=1) > 30)]
    if len(pixels) == 0:
        return [INK]
    unique = np.unique(pixels, axis=0)
    clusters = min(count, len(unique))
    if clusters <= 1:
        return [tuple(int(c) for c in unique[0])]
    model = KMeans(n_clusters=clusters, n_init=10, random_state=12).fit(unique)
    centers = model.cluster_centers_.astype(int)
    labels, counts = np.unique(model.labels_, return_counts=True)
    order = labels[np.argsort(counts)[::-1]]
    return [tuple(int(c) for c in centers[label]) for label in order]


def mask_view(mask: np.ndarray) -> Image.Image:
    bg = np.full((mask.shape[0], mask.shape[1], 4), 247, dtype=np.uint8)
    bg[:, :, 3] = 255
    red = np.zeros_like(bg)
    red[:, :, :3] = (228, 31, 49)
    red[:, :, 3] = (mask > 0).astype(np.uint8) * 220
    return Image.alpha_composite(Image.fromarray(bg, "RGBA"), Image.fromarray(red, "RGBA"))


def contour_view(image: Image.Image, mask: np.ndarray) -> Image.Image:
    canvas = np.array(image.convert("RGB"))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (12, 172, 172), 3)
    return Image.fromarray(canvas).convert("RGBA")


def element_board(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    board = Image.new("RGBA", (980, 620), (248, 245, 236, 255))
    draw = ImageDraw.Draw(board)
    draw.rounded_rectangle((18, 18, 962, 602), radius=26, fill=(255, 253, 246, 255), outline=(210, 202, 188, 255), width=3)
    for i, color in enumerate(colors):
        draw.rounded_rectangle((42 + i * 62, 42, 92 + i * 62, 78), radius=9, fill=rgba(color), outline=(180, 170, 155, 255))
    draw.text((42, 98), f"Найдено элементов: {len(elements)}", fill=(28, 44, 52, 255))
    for index, item in enumerate(elements[:18]):
        x = 48 + (index % 6) * 148
        y = 142 + (index // 6) * 140
        draw.rounded_rectangle((x, y, x + 116, y + 108), radius=16, fill=(246, 244, 236, 255), outline=(214, 207, 192, 255), width=2)
        icon = item["image"].copy()
        icon.thumbnail((92, 82), Image.Resampling.LANCZOS)
        board.alpha_composite(icon, (x + (116 - icon.width) // 2, y + (108 - icon.height) // 2))
    return board


def place(canvas: Image.Image, image: Image.Image, center, size: int, angle: float = 0) -> None:
    icon = image.copy()
    icon.thumbnail((size, size), Image.Resampling.LANCZOS)
    if angle:
        icon = icon.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(icon, (int(center[0] - icon.width / 2), int(center[1] - icon.height / 2)))


def prepare_element(image: Image.Image, size: int, angle: float = 0, enhance: bool = True) -> Image.Image:
    """Resize + optionally sharpen, keep original colours, then rotate."""
    icon = image.copy().convert("RGBA")
    icon.thumbnail((size, size), Image.Resampling.LANCZOS)
    if enhance:
        rgb = icon.convert("RGB")
        rgb = ImageEnhance.Contrast(rgb).enhance(1.15)
        rgb = ImageEnhance.Color(rgb).enhance(1.1)
        result = rgb.convert("RGBA")
        result.putalpha(icon.getchannel("A"))
        icon = result
    if angle:
        icon = icon.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    return icon


def draw_diamond(draw, cx, cy, r, color, width=2):
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    draw.polygon(pts, outline=rgba(color, 200), width=width)


def draw_cross(draw, cx, cy, r, color, width=2):
    draw.line((cx - r, cy, cx + r, cy), fill=rgba(color, 160), width=width)
    draw.line((cx, cy - r, cx, cy + r), fill=rgba(color, 160), width=width)


def fallback_elements() -> list[dict]:
    image = demo_symbols()
    return extract_elements(image, make_mask(image, 232, 11), 520, 12)


# ─────────────────────────────────────────────
#  RIBBON  — horizontal decorative band
# ─────────────────────────────────────────────
def generate_ribbon(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    elements = elements or fallback_elements()
    colors = colors or [INK]
    W, H = 1280, 360
    c0 = colors[0]
    c1 = colors[min(1, len(colors) - 1)]
    c2 = colors[min(2, len(colors) - 1)]

    img = Image.new("RGBA", (W, H), rgba(PAPER))
    draw = ImageDraw.Draw(img)

    # warm parchment background with subtle stripe
    for y in range(H):
        t = y / H
        r = int(250 - t * 6)
        g = int(248 - t * 4)
        b = int(238 - t * 8)
        draw.line((0, y, W, y), fill=(r, g, b, 255))

    # outer frame lines
    border_h = 18
    draw.rectangle((0, 0, W, border_h), fill=rgba(c0))
    draw.rectangle((0, H - border_h, W, H), fill=rgba(c0))

    # inner accent lines
    draw.line((0, border_h + 8, W, border_h + 8), fill=rgba(c1, 180), width=3)
    draw.line((0, H - border_h - 8, W, H - border_h - 8), fill=rgba(c1, 180), width=3)
    draw.line((0, border_h + 18, W, border_h + 18), fill=rgba(c2, 120), width=2)
    draw.line((0, H - border_h - 18, W, H - border_h - 18), fill=rgba(c2, 120), width=2)

    # zigzag row top
    step = 40
    for x in range(0, W, step * 2):
        pts = [(x, border_h + 28), (x + step, border_h + 44), (x + step * 2, border_h + 28)]
        draw.line(pts, fill=rgba(c1, 200), width=3)
    # zigzag row bottom (mirrored)
    for x in range(0, W, step * 2):
        pts = [(x, H - border_h - 28), (x + step, H - border_h - 44), (x + step * 2, H - border_h - 28)]
        draw.line(pts, fill=rgba(c1, 200), width=3)

    # small diamond row top & bottom
    for x in range(20, W, step):
        draw_diamond(draw, x, border_h + 36, 6, c2)
    for x in range(20, W, step):
        draw_diamond(draw, x, H - border_h - 36, 6, c2)

    # main element strip — centred
    cy = H // 2
    n_elem = min(len(elements), 10)
    spacing = W // (n_elem + 1)
    for i in range(n_elem):
        elem = elements[i % len(elements)]
        cx = spacing * (i + 1)
        angle = (-12 if i % 2 == 0 else 12)
        sz = 120 if i % 3 == 0 else 96
        icon = prepare_element(elem["image"], sz, angle)
        img.alpha_composite(icon, (cx - icon.width // 2, cy - icon.height // 2))

        # decorative dot above and below each element
        draw.ellipse((cx - 5, cy - sz // 2 - 20, cx + 5, cy - sz // 2 - 10), fill=rgba(colors[i % len(colors)]))
        draw.ellipse((cx - 5, cy + sz // 2 + 10, cx + 5, cy + sz // 2 + 20), fill=rgba(colors[(i + 1) % len(colors)]))

    # vertical dividers between elements
    for i in range(1, n_elem):
        x = spacing * i + spacing // 2
        draw.line((x, cy - 48, x, cy + 48), fill=rgba(c2, 80), width=1)

    return img


# ─────────────────────────────────────────────
#  FABRIC / GRID
# ─────────────────────────────────────────────
def generate_fabric(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    elements = elements or fallback_elements()
    colors = colors or [INK]
    W, H = 1080, 720
    CELL = 120
    cols = W // CELL
    rows = H // CELL

    img = Image.new("RGBA", (W, H), rgba(PAPER))
    draw = ImageDraw.Draw(img)

    # checkerboard background
    bg_a = (252, 250, 243, 255)
    bg_b = (240, 246, 242, 255)
    for row in range(rows):
        for col in range(cols):
            fill = bg_a if (row + col) % 2 == 0 else bg_b
            draw.rectangle((col * CELL, row * CELL, (col + 1) * CELL - 1, (row + 1) * CELL - 1), fill=fill)

    # draw subtle grid lines
    c_grid = colors[min(1, len(colors) - 1)]
    for x in range(0, W + 1, CELL):
        draw.line((x, 0, x, H), fill=rgba(c_grid, 40), width=1)
    for y in range(0, H + 1, CELL):
        draw.line((0, y, W, y), fill=rgba(c_grid, 40), width=1)

    # place elements — every cell gets one
    for row in range(rows):
        for col in range(cols):
            cx = col * CELL + CELL // 2
            cy = row * CELL + CELL // 2
            idx = (row * cols + col) % len(elements)
            color_idx = (row + col) % len(colors)
            alt = (row + col) % 4

            # every other cell: place element; the rest: decorative motif
            if (row + col) % 2 == 0:
                angle = (col - row) * 15 % 360
                icon = prepare_element(elements[idx]["image"], CELL - 16, angle)
                img.alpha_composite(icon, (cx - icon.width // 2, cy - icon.height // 2))
            else:
                # decorative fill: nested diamonds / crosses
                r = CELL // 2 - 12
                draw_diamond(draw, cx, cy, r, colors[color_idx], width=2)
                draw_diamond(draw, cx, cy, r - 10, colors[(color_idx + 1) % len(colors)], width=1)
                draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=rgba(colors[color_idx], 180))

    # multi-layer border
    c0 = colors[0]
    c1 = colors[min(1, len(colors) - 1)]
    for offset, color, w in [(0, c0, 6), (8, c1, 3), (16, c0, 2)]:
        draw.rectangle((offset, offset, W - offset, H - offset), outline=rgba(color, 200), width=w)

    return img


# ─────────────────────────────────────────────
#  CIRCLE / MANDALA
# ─────────────────────────────────────────────
def generate_circle(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    elements = elements or fallback_elements()
    colors = colors or [INK]
    SIZE = 840
    cx = cy = SIZE // 2

    img = Image.new("RGBA", (SIZE, SIZE), rgba(PAPER))
    draw = ImageDraw.Draw(img)

    # soft radial background gradient
    for r in range(SIZE // 2, 0, -1):
        t = r / (SIZE // 2)
        val = int(250 - t * 10)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(val, val + 2, val - 4, 255))

    c0 = colors[0]
    c1 = colors[min(1, len(colors) - 1)]
    c2 = colors[min(2, len(colors) - 1)]
    c3 = colors[min(3, len(colors) - 1)]

    # concentric decorative rings
    ring_specs = [
        (380, c0, 7),
        (340, c1, 3),
        (295, c0, 5),
        (238, c2, 3),
        (180, c1, 4),
        (120, c0, 3),
        (60,  c2, 5),
    ]
    for r, color, w in ring_specs:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=rgba(color, 200), width=w)

    # tick marks on outer ring
    for i in range(48):
        angle = 2 * pi * i / 48
        r_outer, r_inner = 388, (375 if i % 4 == 0 else 382)
        x1 = cx + cos(angle) * r_outer
        y1 = cy + sin(angle) * r_outer
        x2 = cx + cos(angle) * r_inner
        y2 = cy + sin(angle) * r_inner
        col = c0 if i % 4 == 0 else c1
        draw.line((x1, y1, x2, y2), fill=rgba(col, 200), width=(3 if i % 4 == 0 else 1))

    # diamond accents on ring 340
    for i in range(16):
        angle = 2 * pi * i / 16
        px = cx + cos(angle) * 340
        py = cy + sin(angle) * 340
        draw_diamond(draw, int(px), int(py), 7, colors[i % len(colors)], width=2)

    # ── three rings of elements ──
    ring_data = [
        (6,  115, 84),   # innermost: 6 elements, r=115, size=84
        (10, 198, 68),   # middle:    10 elements
        (16, 295, 52),   # outer:     16 elements
    ]
    for ring_idx, (count, radius, sz) in enumerate(ring_data):
        for i in range(count):
            angle = 2 * pi * i / count + ring_idx * pi / count  # offset each ring
            px = cx + cos(angle) * radius
            py = cy + sin(angle) * radius
            elem_idx = (ring_idx * 7 + i) % len(elements)
            rotation = angle * 180 / pi + 90
            icon = prepare_element(elements[elem_idx]["image"], sz, rotation)
            img.alpha_composite(icon, (int(px - icon.width / 2), int(py - icon.height / 2)))

            # small dot between elements on outer ring
            if ring_idx == 2:
                half_angle = 2 * pi * (i + 0.5) / count + ring_idx * pi / count
                dx = cx + cos(half_angle) * radius
                dy = cy + sin(half_angle) * radius
                r_dot = 4
                draw.ellipse((dx - r_dot, dy - r_dot, dx + r_dot, dy + r_dot),
                             fill=rgba(colors[(ring_idx + i) % len(colors)], 200))

    # centre ornament
    draw.ellipse((cx - 38, cy - 38, cx + 38, cy + 38), fill=rgba(c0))
    draw.ellipse((cx - 26, cy - 26, cx + 26, cy + 26), fill=rgba(PAPER))
    draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=rgba(c1))
    draw.ellipse((cx - 5,  cy - 5,  cx + 5,  cy + 5),  fill=rgba(c0))

    return img


# ─────────────────────────────────────────────
#  POSTER — showcase of all three
# ─────────────────────────────────────────────
def generate_poster(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    elements = elements or fallback_elements()
    colors = colors or [INK]
    W, H = 1320, 860
    c0 = colors[0]
    c1 = colors[min(1, len(colors) - 1)]

    poster = Image.new("RGBA", (W, H), (252, 250, 243, 255))
    draw = ImageDraw.Draw(poster)

    # subtle background gradient
    for y in range(H):
        t = y / H
        val = int(252 - t * 8)
        draw.line((0, y, W, y), fill=(val, val, val - 4, 255))

    # decorative frame
    for offset, color, w in [(0, c0, 8), (10, c1, 3), (20, c0, 2)]:
        draw.rectangle((offset, offset, W - offset, H - offset), outline=rgba(color, 220), width=w)

    # ribbon at top
    ribbon = generate_ribbon(elements, colors).resize((W - 80, 230), Image.Resampling.LANCZOS)
    poster.alpha_composite(ribbon, (40, 40))

    # divider after ribbon
    draw.line((60, 285, W - 60, 285), fill=rgba(c1, 120), width=2)

    # fabric on left bottom
    fabric = generate_fabric(elements, colors).resize((520, 520), Image.Resampling.LANCZOS)
    poster.alpha_composite(fabric, (40, 310))

    # circle on right bottom
    circle = generate_circle(elements, colors).resize((520, 520), Image.Resampling.LANCZOS)
    poster.alpha_composite(circle, (W - 560, 310))

    # centre column: a few free-floating elements
    mid_x = W // 2
    for i, item in enumerate(elements[:6]):
        x = mid_x + ((-1) ** i) * 40
        y = 330 + i * 82
        angle = 20 * ((-1) ** i)
        sz = 88 if i % 2 == 0 else 70
        icon = prepare_element(item["image"], sz, angle)
        poster.alpha_composite(icon, (x - icon.width // 2, y))

    # vertical accent lines flanking the centre column
    draw.line((mid_x - 72, 300, mid_x - 72, 820), fill=rgba(c0, 80), width=3)
    draw.line((mid_x + 72, 300, mid_x + 72, 820), fill=rgba(c1, 80), width=3)
    draw.line((mid_x - 82, 300, mid_x - 82, 820), fill=rgba(c1, 40), width=1)
    draw.line((mid_x + 82, 300, mid_x + 82, 820), fill=rgba(c0, 40), width=1)

    # caption
    caption = "Витрина: лента · сетка · круговой орнамент"
    draw.text((mid_x - 170, 828), caption, fill=rgba(INK, 160))

    return poster


def period(mask: np.ndarray) -> int:
    rows = np.where(mask.sum(axis=1) > mask.shape[1] * 2)[0]
    if len(rows) == 0:
        return 0
    band = mask[max(0, rows.min() - 5): min(mask.shape[0], rows.max() + 6), :]
    signal = band.sum(axis=0).astype(np.float32)
    signal -= signal.mean()
    corr = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
    corr[: max(20, mask.shape[1] // 30)] = 0
    return int(np.argmax(corr[: mask.shape[1] // 2])) if corr.max() > 0 else 0


# ═══════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════
with st.sidebar:
    st.subheader("Источник")
    source_mode = st.radio("Изображение", [*SAMPLE_SOURCES.keys(), "Загрузить свое"])
    upload = st.file_uploader("PNG, JPG, WEBP", type=["png", "jpg", "jpeg", "webp"], disabled=source_mode != "Загрузить свое")

    st.subheader("Распознавание")
    sensitivity = st.slider("Чувствительность к фону", 170, 248, 232)
    close_size  = st.slider("Склеивание линий", 3, 25, 3, step=2)
    min_area    = st.slider("Минимальный размер фрагмента", 80, 5000, 520)
    padding     = st.slider("Отступ вокруг фрагмента", 2, 32, 12)


# ═══════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════
st.markdown(
    """
<h1>OrnaMap Vision</h1>
<p class="lead">Лаборатория компьютерного зрения для орнаментов: программа отделяет узор от фона,
находит контуры, вырезает отдельные элементы, определяет палитру и собирает новые композиции.</p>
<span class="chip">OpenCV</span><span class="chip">контуры</span><span class="chip">сегментация</span>
<span class="chip">KMeans</span><span class="chip">генерация паттернов</span>
""",
    unsafe_allow_html=True,
)

source   = load_source(upload, source_mode)
mask     = make_mask(source, sensitivity, close_size)
elements = extract_elements(source, mask, min_area, padding)
colors   = palette(source, mask)
repeat   = period(mask)
fill     = int((mask > 0).mean() * 100)
board    = element_board(elements, colors)
poster   = generate_poster(elements, colors)
ribbon   = generate_ribbon(elements, colors)
fabric   = generate_fabric(elements, colors)
circle   = generate_circle(elements, colors)

# ═══════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════
left, right = st.columns([1, 1.2])

with left:
    st.subheader("Исходное изображение")
    st.image(source, width="stretch")
    st.markdown(
        f"""
<div class="metric"><b>{len(elements)}</b><br><span>найденных фрагментов</span></div>
<div class="metric"><b>{fill}%</b><br><span>площадь узора на изображении</span></div>
<div class="metric"><b>{repeat if repeat else "нет"}</b><br><span>примерный период по горизонтали</span></div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.subheader("Найденные элементы")
    st.image(board, width="stretch")
    st.download_button("Скачать карточку элементов", data=png_bytes(board), file_name="ornamap_elements.png", mime="image/png")

    st.subheader("Новые орнаменты из этих элементов")
    st.image(poster, caption="Витрина: лента, сетка и круговой орнамент", width="stretch")
    st.download_button("Скачать витрину PNG", data=png_bytes(poster), file_name="ornamap_showcase.png", mime="image/png")

    gen_a, gen_b, gen_c = st.tabs(["Лента", "Сетка", "Круг"])
    with gen_a:
        st.image(ribbon, width="stretch")
        st.download_button("Скачать ленту", data=png_bytes(ribbon), file_name="ornamap_ribbon.png", mime="image/png")
    with gen_b:
        st.image(fabric, width="stretch")
        st.download_button("Скачать сетку", data=png_bytes(fabric), file_name="ornamap_fabric.png", mime="image/png")
    with gen_c:
        st.image(circle, width="stretch")
        st.download_button("Скачать круг", data=png_bytes(circle), file_name="ornamap_circle.png", mime="image/png")

st.divider()

tab_steps, tab_science = st.tabs(["Как видит программа", "Для посетителей моего сайта"])

with tab_steps:
    c1_col, c2_col, c3_col = st.columns(3)
    with c1_col:
        st.image(mask_view(mask), caption="1. Маска: узор отделен от фона", width="stretch")
    with c2_col:
        st.image(contour_view(source, mask), caption="2. Контуры найденных фрагментов", width="stretch")
    with c3_col:
        swatches = Image.new("RGBA", (820, 240), (255, 253, 246, 255))
        draw_sw = ImageDraw.Draw(swatches)
        for i, color in enumerate(colors):
            draw_sw.rounded_rectangle((42 + i * 145, 58, 142 + i * 145, 158), radius=20, fill=rgba(color), outline=(190, 180, 165, 255), width=2)
            draw_sw.text((42 + i * 145, 174), f"RGB {color}", fill=(43, 58, 65, 255))
        st.image(swatches, caption="3. Палитра KMeans", width="stretch")

with tab_science:
    st.markdown(
        """
**Что происходит внутри:** изображение превращается в массив пикселей — и дальше в дело вступает алгоритм. OpenCV строит маску, отделяя узор от фона, 
затем находит контуры каждого элемента, вырезает его с прозрачным фоном и передаёт в следующий шаг. KMeans анализирует цвета и собирает палитру орнамента.

**Что вы видите на экране:** весь путь от исходного изображения до результата — маска, контуры, отдельные фрагменты, палитра и новые орнаменты, собранные 
из найденных элементов.

**Идея проекта:** орнамент — это не просто картинка. Это структура, которую можно распознать, разобрать на части и пересобрать заново. Программа делает именно 
это: находит элементы народного узора и использует их как цифровой материал для создания новых композиций.
"""
    )
