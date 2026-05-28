from io import BytesIO
from math import cos, pi, sin
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
from sklearn.cluster import KMeans


st.set_page_config(page_title="OrnaMap Vision", page_icon="OM", layout="wide")

RED = (228, 31, 49)
BLUE = (29, 68, 154)
INK = (14, 16, 15)
PAPER = (250, 249, 243)

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
    else:
        image = demo_strip() if mode == "Полосной орнамент" else demo_symbols()
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


def fallback_elements() -> list[dict]:
    image = demo_symbols()
    return extract_elements(image, make_mask(image, 232, 11), 520, 12)


def generate_ribbon(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    elements = elements or fallback_elements()
    image = Image.new("RGBA", (1200, 300), rgba(PAPER))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1200, 18), fill=rgba(colors[0]))
    draw.rectangle((0, 282, 1200, 300), fill=rgba(colors[min(1, len(colors) - 1)]))
    for i, x in enumerate(range(65, 1200, 130)):
        place(image, elements[i % len(elements)]["image"], (x, 150), 112, 0 if i % 2 == 0 else 180)
    return image


def generate_fabric(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    elements = elements or fallback_elements()
    image = Image.new("RGBA", (920, 620), (248, 246, 239, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, 620, 116):
        for x in range(0, 920, 116):
            fill = (255, 255, 250, 255) if (x + y) // 116 % 2 == 0 else (235, 241, 239, 255)
            draw.rectangle((x, y, x + 116, y + 116), fill=fill)
    for row, y in enumerate(range(58, 620, 116)):
        for col, x in enumerate(range(58, 920, 116)):
            place(image, elements[(row * 7 + col) % len(elements)]["image"], (x, y), 88, 0 if (row + col) % 2 == 0 else 180)
    draw.rounded_rectangle((18, 18, 902, 602), radius=20, outline=rgba(colors[0]), width=5)
    return image


def generate_circle(elements: list[dict], colors: list[tuple[int, int, int]]) -> Image.Image:
    elements = elements or fallback_elements()
    image = Image.new("RGBA", (760, 760), rgba(PAPER))
    draw = ImageDraw.Draw(image)
    draw.ellipse((72, 72, 688, 688), outline=rgba(colors[0]), width=7)
    draw.ellipse((180, 180, 580, 580), outline=rgba(colors[min(1, len(colors) - 1)]), width=5)
    for ring, radius in enumerate([116, 220, 306]):
        count = 8 + ring * 4
        for i in range(count):
            angle = 2 * pi * i / count
            x = 380 + int(cos(angle) * radius)
            y = 380 + int(sin(angle) * radius)
            place(image, elements[(i + ring * 3) % len(elements)]["image"], (x, y), 78 - ring * 6, angle * 180 / pi + 90)
    draw.ellipse((344, 344, 416, 416), fill=rgba(colors[0]))
    return image


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


with st.sidebar:
    st.subheader("Источник")
    source_mode = st.radio("Изображение", ["Лист знаков", "Полосной орнамент", "Загрузить свое"])
    upload = st.file_uploader("PNG, JPG, WEBP", type=["png", "jpg", "jpeg", "webp"], disabled=source_mode != "Загрузить свое")
    st.subheader("Распознавание")
    sensitivity = st.slider("Чувствительность к фону", 170, 248, 232)
    close_size = st.slider("Склеивание линий", 3, 25, 11, step=2)
    min_area = st.slider("Минимальный размер фрагмента", 80, 5000, 520)
    padding = st.slider("Отступ вокруг фрагмента", 2, 32, 12)


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

source = load_source(upload, source_mode)
mask = make_mask(source, sensitivity, close_size)
elements = extract_elements(source, mask, min_area, padding)
colors = palette(source, mask)
repeat = period(mask)
fill = int((mask > 0).mean() * 100)

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
    board = element_board(elements, colors)
    st.image(board, width="stretch")
    st.download_button("Скачать карточку элементов", data=png_bytes(board), file_name="ornamap_elements.png", mime="image/png")

st.divider()
tab_steps, tab_generate, tab_science = st.tabs(["Как видит программа", "Новые орнаменты", "Для защиты"])

with tab_steps:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(mask_view(mask), caption="1. Маска: узор отделен от фона", width="stretch")
    with c2:
        st.image(contour_view(source, mask), caption="2. Контуры найденных фрагментов", width="stretch")
    with c3:
        swatches = Image.new("RGBA", (820, 240), (255, 253, 246, 255))
        draw = ImageDraw.Draw(swatches)
        for i, color in enumerate(colors):
            draw.rounded_rectangle((42 + i * 145, 58, 142 + i * 145, 158), radius=20, fill=rgba(color), outline=(190, 180, 165, 255), width=2)
            draw.text((42 + i * 145, 174), f"RGB {color}", fill=(43, 58, 65, 255))
        st.image(swatches, caption="3. Палитра KMeans", width="stretch")

with tab_generate:
    g1, g2 = st.columns([1.2, 1])
    with g1:
        ribbon = generate_ribbon(elements, colors)
        fabric = generate_fabric(elements, colors)
        st.image(ribbon, caption="Лента из найденных элементов", width="stretch")
        st.download_button("Скачать ленту PNG", data=png_bytes(ribbon), file_name="ornamap_ribbon.png", mime="image/png")
        st.image(fabric, caption="Повторяющийся паттерн для ткани", width="stretch")
    with g2:
        circle = generate_circle(elements, colors)
        st.image(circle, caption="Круговая композиция", width="stretch")
        st.download_button("Скачать круговую композицию", data=png_bytes(circle), file_name="ornamap_circle.png", mime="image/png")

with tab_science:
    st.markdown(
        """
        **Что происходит внутри:** изображение переводится в массив пикселей, затем OpenCV строит маску объектов
        на светлом фоне. После этого программа ищет внешние контуры, обрезает каждый найденный фрагмент и делает
        его прозрачным по маске.

        **Почему это выглядит убедительно:** на экране виден весь путь исследования: исходник, маска, контуры,
        найденные элементы, палитра и новые орнаменты из этих элементов.

        **Фраза для защиты:** я не рисую орнамент вручную в редакторе, а показываю, как алгоритм компьютерного
        зрения может распознать элементы народного узора и использовать их как цифровой материал для нового дизайна.
        """
    )
