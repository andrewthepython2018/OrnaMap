from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans


st.set_page_config(page_title="OrnaMap", page_icon="OM", layout="wide")

CANVAS_SIZE = (980, 680)

TEMPLATES = {
    "Шоппер": {"box": (300, 175, 680, 555), "kind": "bag"},
    "Футболка": {"box": (355, 195, 625, 445), "kind": "shirt"},
    "Обложка блокнота": {"box": (350, 115, 655, 590), "kind": "notebook"},
    "Плакат": {"box": (230, 85, 750, 600), "kind": "poster"},
    "Кружка: развертка": {"box": (205, 205, 775, 465), "kind": "mug_wrap"},
}

SYMBOLS = [
    "Фигура с посохом",
    "Танцующая фигура",
    "Треугольная фигура",
    "Чум",
    "Лодка",
    "Знак солнца",
    "Олень",
    "След",
]


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 8%, rgba(219, 35, 52, .08), transparent 27%),
            linear-gradient(135deg, #f8f3ea 0%, #eef5f3 48%, #fbfaf5 100%);
    }
    .block-container { padding-top: 2rem; max-width: 1220px; }
    h1 { color: #203945; font-size: 3.35rem !important; line-height: 1 !important; }
    h2, h3 { color: #203945; }
    [data-testid="stSidebar"] {
        background: #f5f0e8;
        border-right: 1px solid rgba(56, 73, 81, .12);
    }
    div[data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 18px 45px rgba(40, 48, 52, .12);
    }
    .note { color: #66737a; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def textured_background(size: tuple[int, int] = CANVAS_SIZE) -> Image.Image:
    width, height = size
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0] = (244 - 12 * y + 5 * x).astype(np.uint8)
    base[..., 1] = (242 - 7 * y + 6 * x).astype(np.uint8)
    base[..., 2] = (234 - 8 * y + 10 * x).astype(np.uint8)
    noise = np.random.default_rng(11).normal(0, 1.5, base.shape).astype(np.int16)
    return Image.fromarray(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)).convert("RGBA")


def shadow(size: tuple[int, int], box: tuple[int, int, int, int], radius: int = 30) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(box, radius=radius, fill=(23, 34, 38, 58))
    return layer.filter(ImageFilter.GaussianBlur(24))


def pixel_rect(draw: ImageDraw.ImageDraw, x: int, y: int, cells: list[tuple[int, int]], color, s: int) -> None:
    for cx, cy in cells:
        draw.rectangle((x + cx * s, y + cy * s, x + (cx + 1) * s - 1, y + (cy + 1) * s - 1), fill=color)


def make_strip_ornament(width: int = 900, height: int = 560) -> Image.Image:
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    red = (232, 28, 48, 255)
    blue = (29, 67, 154, 255)
    s = 8

    def band(y: int, motif_width: int, color, cells: list[tuple[int, int]]) -> None:
        for x in range(-motif_width, width + motif_width, motif_width):
            pixel_rect(draw, x, y, cells, color, s)

    band(18, 88, blue, [(0, 2), (1, 2), (2, 2), (3, 2), (1, 1), (2, 0), (3, 1), (6, 2), (7, 2), (8, 2), (9, 2), (7, 1), (8, 0), (9, 1)])
    draw.rectangle((0, 53, width, 58), fill=blue)
    band(76, 190, red, [(5, 1), (6, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (8, 6), (9, 7), (6, 5), (6, 6), (6, 7), (5, 8), (7, 8), (4, 9), (8, 9), (14, 1), (15, 1), (16, 1), (15, 2), (15, 3), (14, 4), (16, 4), (13, 5), (17, 5), (12, 6), (18, 6), (11, 7), (19, 7), (12, 8), (18, 8), (13, 9), (17, 9), (14, 10), (16, 10)])
    draw.rectangle((0, 154, width, 160), fill=blue)
    band(176, 74, blue, [(0, 0), (1, 0), (2, 0), (2, 1), (3, 2), (4, 3), (5, 2), (6, 1), (6, 0), (7, 0), (8, 0)])
    band(240, 76, red, [(0, 8), (1, 7), (2, 6), (3, 5), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (1, 8), (2, 8), (6, 8), (7, 8)])
    draw.rectangle((0, 312, width, 318), fill=blue)
    band(336, 40, blue, [(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)])
    band(386, 210, red, [(0, 4), (1, 3), (2, 2), (3, 1), (4, 2), (5, 3), (6, 4), (12, 1), (12, 2), (12, 3), (11, 4), (10, 5), (11, 6), (12, 6), (13, 6), (14, 5), (13, 4), (19, 4), (20, 3), (21, 2), (22, 1), (23, 2), (24, 3), (25, 4)])
    draw.rectangle((0, 468, width, 474), fill=blue)
    band(495, 70, blue, [(0, 0), (1, 1), (2, 2), (3, 2), (4, 1), (5, 0), (2, 3), (3, 3)])
    return image


def draw_symbol(name: str, size: int = 260) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    black = (15, 16, 14, 255)
    w = max(5, size // 32)
    c = size // 2

    if name == "Фигура с посохом":
        draw.ellipse((c - 22, 28, c + 22, 72), fill=black)
        draw.line((c, 72, c, 155), fill=black, width=w + 2)
        draw.line((c, 96, c - 62, 122), fill=black, width=w)
        draw.line((c, 96, c + 54, 125), fill=black, width=w)
        draw.line((c, 155, c - 44, 220), fill=black, width=w)
        draw.line((c, 155, c + 46, 220), fill=black, width=w)
        draw.line((215, 35, 238, 225), fill=black, width=max(3, w - 2))
        draw.line((42, 224, 215, 224), fill=black, width=w)
    elif name == "Танцующая фигура":
        draw.ellipse((c - 20, 24, c + 20, 64), fill=black)
        draw.polygon((c, 64, c - 42, 170, c + 42, 170), fill=black)
        draw.line((c - 22, 88, c - 82, 70), fill=black, width=w)
        draw.line((c + 22, 88, c + 82, 70), fill=black, width=w)
        draw.line((c - 16, 170, c - 38, 224), fill=black, width=w)
        draw.line((c + 16, 170, c + 44, 224), fill=black, width=w)
    elif name == "Треугольная фигура":
        draw.ellipse((c - 15, 28, c + 15, 58), outline=black, width=w)
        draw.polygon((c, 70, c - 54, 205, c + 54, 205), outline=black)
        draw.line((c - 54, 205, c + 54, 205), fill=black, width=w)
        draw.line((c - 82, 95, c - 38, 128), fill=black, width=w)
        draw.line((c + 40, 128, c + 78, 96), fill=black, width=w)
        draw.line((c - 28, 140, c + 28, 140), fill=black, width=w)
    elif name == "Чум":
        draw.polygon((c, 45, c - 76, 205, c + 76, 205), outline=black)
        draw.line((c, 45, c, 205), fill=black, width=w)
        draw.line((c - 52, 150, c + 52, 150), fill=black, width=w)
        draw.arc((c - 46, 120, c + 46, 214), 200, 340, fill=black, width=w)
        draw.line((45, 222, 215, 222), fill=black, width=w)
    elif name == "Лодка":
        draw.arc((42, 124, 220, 230), 12, 168, fill=black, width=w + 3)
        draw.line((58, 172, 210, 172), fill=black, width=w)
        draw.line((c, 55, c, 172), fill=black, width=w)
        draw.line((c, 82, c - 52, 132), fill=black, width=w)
        draw.line((c, 82, c + 44, 132), fill=black, width=w)
        draw.line((72, 214, 210, 214), fill=black, width=w)
    elif name == "Знак солнца":
        draw.line((c, 32, c, 222), fill=black, width=w)
        draw.line((55, 125, 205, 125), fill=black, width=w)
        draw.line((80, 62, 180, 188), fill=black, width=w)
        draw.line((180, 62, 80, 188), fill=black, width=w)
        draw.polygon((c, 52, c - 26, 116, c + 26, 116), outline=black)
    elif name == "Олень":
        draw.line((65, 152, 168, 152), fill=black, width=w + 2)
        draw.line((168, 152, 205, 112), fill=black, width=w)
        draw.line((94, 152, 74, 210), fill=black, width=w)
        draw.line((140, 152, 154, 210), fill=black, width=w)
        draw.line((200, 112, 230, 92), fill=black, width=w)
        draw.line((205, 112, 232, 132), fill=black, width=w)
        draw.line((220, 92, 230, 62), fill=black, width=max(3, w - 1))
    else:
        draw.line((42, 210, 218, 210), fill=black, width=w)
        draw.line((c, 50, c, 210), fill=black, width=w)
        draw.line((c, 50, c - 35, 92), fill=black, width=w)
        draw.line((c, 50, c + 36, 92), fill=black, width=w)
        draw.line((c - 34, 138, c + 34, 138), fill=black, width=w)
    return image.filter(ImageFilter.GaussianBlur(0.35))


def open_uploaded_or_fallback(uploaded_file, fallback: Image.Image) -> Image.Image:
    if uploaded_file is None:
        return fallback
    return Image.open(uploaded_file).convert("RGBA")


def fit_to_square(image: Image.Image, size: int = 480) -> Image.Image:
    image = image.convert("RGBA")
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def find_palette(image: Image.Image, color_count: int) -> list[tuple[int, int, int]]:
    rgb = image.convert("RGB").resize((150, 150))
    pixels = np.array(rgb).reshape(-1, 3)
    model = KMeans(n_clusters=color_count, n_init=10, random_state=42)
    model.fit(pixels)
    centers = model.cluster_centers_.astype(int)
    labels, counts = np.unique(model.labels_, return_counts=True)
    order = labels[np.argsort(counts)[::-1]]
    return [tuple(centers[label]) for label in order]


def edge_map(image: Image.Image, strength: int) -> Image.Image:
    gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
    gray = ImageEnhance.Contrast(gray).enhance(1 + strength / 18)
    alpha = gray.point(lambda value: 210 if value > 32 else 0)
    edges = Image.new("RGBA", image.size, (20, 45, 54, 0))
    edges.putalpha(alpha)
    return edges


def recolor_image(image: Image.Image, palette: list[tuple[int, int, int]], opacity: int) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    gray = np.mean(arr, axis=2)
    levels = np.linspace(0, 256, len(palette) + 1)
    result = np.zeros_like(arr)

    for index, color in enumerate(palette):
        mask = (gray >= levels[index]) & (gray < levels[index + 1])
        result[mask] = color

    recolored = Image.fromarray(result.astype("uint8")).convert("RGBA")
    source_alpha = np.array(image.getchannel("A"), dtype=np.float32)
    alpha = (source_alpha * (opacity / 100)).clip(0, 255).astype("uint8")
    recolored.putalpha(Image.fromarray(alpha))
    return recolored


def make_pattern_tile(image: Image.Image, repeats: int, mirror: bool) -> Image.Image:
    tile_size = 220
    tile = image.resize((tile_size, tile_size), Image.LANCZOS)
    pattern = Image.new("RGBA", (tile_size * repeats, tile_size * repeats), (255, 255, 255, 0))
    for y in range(repeats):
        for x in range(repeats):
            current = tile
            if mirror and (x + y) % 2 == 1:
                current = current.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pattern.alpha_composite(current, (x * tile_size, y * tile_size))
    return pattern


def make_symbol_pattern(symbol: Image.Image, repeats: int, opacity: int) -> Image.Image:
    tile_size = 190
    symbol = symbol.resize((tile_size, tile_size), Image.LANCZOS)
    alpha = np.array(symbol.getchannel("A"), dtype=np.float32)
    symbol.putalpha(Image.fromarray((alpha * (opacity / 100)).clip(0, 255).astype("uint8")))
    pattern = Image.new("RGBA", (tile_size * repeats, tile_size * repeats), (255, 255, 255, 0))
    for y in range(repeats):
        for x in range(repeats):
            pattern.alpha_composite(symbol, (x * tile_size, y * tile_size))
    return pattern


def draw_template(name: str) -> Image.Image:
    template = TEMPLATES[name]
    kind = template["kind"]
    box = template["box"]
    image = textured_background()
    draw = ImageDraw.Draw(image)
    image.alpha_composite(shadow(CANVAS_SIZE, box, 26))

    if kind == "bag":
        product = (250, 110, 730, 625)
        image.alpha_composite(shadow(CANVAS_SIZE, product, 34))
        draw.rounded_rectangle(product, radius=34, fill=(239, 232, 214), outline=(186, 174, 150), width=5)
        draw.arc((365, 25, 615, 250), 180, 360, fill=(126, 104, 76), width=15)
        draw.line((365, 120, 365, 180), fill=(126, 104, 76), width=15)
        draw.line((615, 120, 615, 180), fill=(126, 104, 76), width=15)
    elif kind == "shirt":
        shape = (285, 120, 400, 80, 490, 155, 580, 80, 695, 120, 765, 270, 685, 320, 670, 620, 310, 620, 295, 320, 215, 270)
        image.alpha_composite(shadow(CANVAS_SIZE, (215, 80, 765, 625), 45))
        draw.polygon(shape, fill=(246, 246, 239), outline=(176, 178, 168))
        draw.arc((430, 105, 550, 215), 0, 180, fill=(176, 178, 168), width=6)
    elif kind == "notebook":
        product = (320, 80, 685, 625)
        image.alpha_composite(shadow(CANVAS_SIZE, product, 22))
        draw.rounded_rectangle(product, radius=22, fill=(248, 246, 238), outline=(194, 184, 166), width=5)
        draw.rectangle((340, 80, 358, 625), fill=(220, 188, 96))
        for y in range(125, 590, 48):
            draw.ellipse((304, y, 334, y + 30), fill=(38, 47, 52))
    elif kind == "mug_wrap":
        product = (170, 170, 810, 500)
        image.alpha_composite(shadow(CANVAS_SIZE, product, 30))
        draw.rounded_rectangle(product, radius=30, fill=(250, 250, 245), outline=(178, 185, 184), width=5)
        draw.text((210, 505), "Развертка для печати на кружке", fill=(96, 105, 108))
    else:
        product = (190, 65, 790, 625)
        image.alpha_composite(shadow(CANVAS_SIZE, product, 18))
        draw.rectangle(product, fill=(250, 248, 240), outline=(112, 96, 75), width=12)
        draw.rectangle((216, 92, 764, 598), outline=(196, 188, 170), width=4)

    draw.rounded_rectangle(box, radius=16, outline=(32, 57, 69, 115), width=3)
    return image


def fit_custom_mockup(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    image.thumbnail(CANVAS_SIZE, Image.LANCZOS)
    canvas = textured_background()
    canvas.alpha_composite(image, ((CANVAS_SIZE[0] - image.width) // 2, (CANVAS_SIZE[1] - image.height) // 2))
    return canvas


def place_pattern_on_mockup(mockup: Image.Image, pattern: Image.Image, box: tuple[int, int, int, int], opacity: int) -> Image.Image:
    left, top, right, bottom = box
    fitted = pattern.resize((right - left, bottom - top), Image.LANCZOS)
    alpha = np.array(fitted.getchannel("A"), dtype=np.float32)
    fitted.putalpha(Image.fromarray((alpha * (opacity / 100)).clip(0, 255).astype("uint8")))
    result = mockup.copy()
    result.alpha_composite(fitted, (left, top))
    overlay = Image.new("RGBA", result.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((left, top, right, bottom), radius=16, outline=(255, 255, 255, 95), width=2)
    result.alpha_composite(overlay)
    return result


st.title("OrnaMap")
st.caption("Рабочие плоские макеты с точной областью печати")

with st.sidebar:
    st.subheader("Орнамент")
    ornament_type = st.radio("Тип узора", ["Полосной орнамент", "Отдельный рисунок", "Загрузить свой"])
    selected_symbol = st.selectbox("Рисунок из второго изображения", SYMBOLS, disabled=ornament_type != "Отдельный рисунок")
    uploaded_ornament = st.file_uploader("Загрузите свой орнамент", type=["png", "jpg", "jpeg", "webp"], disabled=ornament_type != "Загрузить свой")

    st.subheader("Макет")
    mockup_mode = st.radio("Источник макета", ["Готовый плоский", "Загрузить свой"], horizontal=True)
    template_name = st.selectbox("Готовый макет", list(TEMPLATES.keys()), disabled=mockup_mode != "Готовый плоский")
    uploaded_mockup = st.file_uploader("Загрузите свой макет", type=["png", "jpg", "jpeg", "webp"], disabled=mockup_mode != "Загрузить свой")

    color_count = st.slider("Сколько основных цветов найти", 2, 6, 4)
    repeats = st.slider("Повторы узора", 1, 8, 3 if ornament_type == "Отдельный рисунок" else 4)
    pattern_opacity = st.slider("Прозрачность узора", 35, 100, 92)
    edge_strength = st.slider("Сила выделения контура", 1, 12, 5)
    mirror = st.toggle("Чередовать зеркальные плитки", value=True, disabled=ornament_type == "Отдельный рисунок")

    custom_box = None
    if mockup_mode == "Загрузить свой":
        st.subheader("Область наложения")
        x = st.slider("Смещение по горизонтали", 0, 760, 300)
        y = st.slider("Смещение по вертикали", 0, 520, 180)
        w = st.slider("Ширина области", 120, 720, 360)
        h = st.slider("Высота области", 120, 500, 300)
        custom_box = (x, y, min(CANVAS_SIZE[0], x + w), min(CANVAS_SIZE[1], y + h))


strip = make_strip_ornament()
symbol = draw_symbol(selected_symbol)

if ornament_type == "Полосной орнамент":
    source = fit_to_square(strip)
    palette = find_palette(source, color_count)
    recolored = recolor_image(source, palette, 100)
    tile = Image.alpha_composite(recolored, edge_map(source, edge_strength))
    pattern = make_pattern_tile(tile, repeats, mirror)
    source_caption = "Полосной орнамент"
elif ornament_type == "Отдельный рисунок":
    source = fit_to_square(symbol)
    palette = find_palette(source, color_count)
    pattern = make_symbol_pattern(symbol, repeats, 100)
    source_caption = f"Отдельный рисунок: {selected_symbol}"
else:
    source = fit_to_square(open_uploaded_or_fallback(uploaded_ornament, strip))
    palette = find_palette(source, color_count)
    recolored = recolor_image(source, palette, 100)
    tile = Image.alpha_composite(recolored, edge_map(source, edge_strength))
    pattern = make_pattern_tile(tile, repeats, mirror)
    source_caption = "Пользовательский орнамент"

if mockup_mode == "Загрузить свой":
    mockup = fit_custom_mockup(open_uploaded_or_fallback(uploaded_mockup, draw_template("Плакат")))
    box = custom_box or (300, 180, 660, 480)
    result_caption = "Пользовательский макет"
else:
    mockup = draw_template(template_name)
    box = TEMPLATES[template_name]["box"]
    result_caption = f"Макет: {template_name}"

result = place_pattern_on_mockup(mockup, pattern, box, pattern_opacity)

left, right = st.columns([1, 1.25])
with left:
    st.subheader("Источник узора")
    st.image(source, caption=source_caption, width="stretch")
    swatches = "".join(
        f"<span style='display:inline-block;width:54px;height:38px;background:rgb{color};"
        "border-radius:10px;border:1px solid rgba(32,57,69,.18);margin-right:8px'></span>"
        for color in palette
    )
    st.markdown(swatches, unsafe_allow_html=True)
    st.markdown("<p class='note'>В этой версии макет не имитирует фото, а показывает точную область печати. Так узор не съезжает и всегда читается.</p>", unsafe_allow_html=True)

with right:
    st.subheader("Примерка на макете")
    st.image(result, caption=result_caption, width="stretch")
    st.download_button("Скачать результат PNG", data=image_to_png_bytes(result), file_name="ornamap_result.png", mime="image/png")

st.divider()
st.subheader("Отдельные рисунки")
cols = st.columns(4)
for index, name in enumerate(SYMBOLS):
    with cols[index % 4]:
        st.image(draw_symbol(name), caption=name, width="stretch")

tab1, tab2, tab3 = st.tabs(["Как работает", "Для защиты", "Что дальше"])
with tab1:
    st.markdown(
        """
        1. Пользователь выбирает полосной орнамент, отдельный знак или загружает свое изображение.
        2. Программа строит повторяющийся паттерн.
        3. Макет содержит точную область печати, поэтому узор накладывается ровно.
        4. Результат можно скачать как PNG.
        """
    )
with tab2:
    st.markdown(
        """
        **Как объяснить:** я отказалась от случайных фото-мокапов, потому что на них сложно точно определить
        место печати. Вместо этого приложение показывает плоский дизайн-макет, похожий на рабочий шаблон
        для дизайнера.

        **Информатика:** изображение анализируется как массив пикселей, цвета группируются KMeans,
        а итоговый макет собирается программно из слоев.
        """
    )
with tab3:
    st.markdown(
        """
        - добавить автоматическое определение области печати на загруженном макете;
        - сделать экспорт PDF для печати;
        - сохранять библиотеку пользовательских знаков;
        - добавить режим симметрии для полосных орнаментов.
        """
    )
