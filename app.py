from io import BytesIO
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from sklearn.cluster import KMeans


st.set_page_config(page_title="OrnaMap", page_icon="OM", layout="wide")

CANVAS_SIZE = (980, 680)

MOCKUPS = {
    "Шоппер": {
        "box": (355, 215, 625, 505),
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Tote%20Bag.jpeg?width=900",
        "source": "Wikimedia Commons: Tote Bag.jpeg",
    },
    "Блокнот": {
        "box": (210, 145, 765, 520),
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Blank%20Notebook.jpg?width=900",
        "source": "Wikimedia Commons: Blank Notebook.jpg",
    },
    "Плакат": {
        "box": (304, 88, 668, 635),
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Blank%20poster%20for%20Fedoruk%20screenings%20%2816739030627%29.jpg?width=900",
        "source": "Wikimedia Commons: Blank poster for Fedoruk screenings",
    },
    "Футболка": {
        "box": (378, 230, 600, 430),
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/White%20T%20shirt.jpg?width=900",
        "source": "Wikimedia Commons: White T shirt.jpg",
    },
    "Кружка": {
        "box": (360, 250, 625, 450),
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Taza%20Blanca.jpg?width=900",
        "source": "Wikimedia Commons: Taza Blanca.jpg",
    },
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
            radial-gradient(circle at 12% 5%, rgba(190, 47, 54, 0.08), transparent 28%),
            linear-gradient(135deg, #f7f2e8 0%, #edf3f2 48%, #f9f7f0 100%);
    }
    .block-container { padding-top: 2.1rem; max-width: 1220px; }
    h1 { font-size: 3.4rem !important; line-height: 1 !important; color: #203945; }
    h2, h3 { color: #203945; }
    [data-testid="stSidebar"] {
        background: #f4f0e8;
        border-right: 1px solid rgba(56, 73, 81, 0.12);
    }
    div[data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 18px 45px rgba(40, 48, 52, 0.12);
    }
    .source-note { color: #637077; font-size: 0.92rem; margin-top: -0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def textured_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0] = (238 - 15 * y + 8 * x).astype(np.uint8)
    base[..., 1] = (241 - 7 * y + 5 * x).astype(np.uint8)
    base[..., 2] = (235 - 11 * y + 11 * x).astype(np.uint8)
    noise = np.random.default_rng(7).normal(0, 2, base.shape).astype(np.int16)
    return Image.fromarray(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)).convert("RGBA")


def shadow_layer(size: tuple[int, int], box: tuple[int, int, int, int], radius: int = 36) -> Image.Image:
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.rounded_rectangle(box, radius=radius, fill=(25, 34, 37, 70))
    return shadow.filter(ImageFilter.GaussianBlur(22))


def pixel_rect(draw: ImageDraw.ImageDraw, x: int, y: int, cells: list[tuple[int, int]], color: tuple[int, int, int, int], s: int) -> None:
    for cx, cy in cells:
        draw.rectangle((x + cx * s, y + cy * s, x + (cx + 1) * s - 1, y + (cy + 1) * s - 1), fill=color)


def make_strip_ornament(width: int = 900, height: int = 560) -> Image.Image:
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    red = (232, 28, 48, 255)
    blue = (29, 67, 154, 255)
    s = 8

    def repeat_band(y: int, motif_width: int, color: tuple[int, int, int, int], cells: list[tuple[int, int]]) -> None:
        for x in range(-motif_width, width + motif_width, motif_width):
            pixel_rect(draw, x, y, cells, color, s)

    blue_top = [(0, 2), (1, 2), (2, 2), (3, 2), (1, 1), (2, 0), (3, 1), (6, 2), (7, 2), (8, 2), (9, 2), (7, 1), (8, 0), (9, 1)]
    repeat_band(18, 88, blue, blue_top)
    draw.rectangle((0, 53, width, 58), fill=blue)

    red_people = [
        (5, 1), (6, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (8, 6), (9, 7),
        (6, 5), (6, 6), (6, 7), (5, 8), (7, 8), (4, 9), (8, 9),
        (14, 1), (15, 1), (16, 1), (15, 2), (15, 3), (14, 4), (16, 4),
        (13, 5), (17, 5), (12, 6), (18, 6), (11, 7), (19, 7), (12, 8), (18, 8), (13, 9), (17, 9), (14, 10), (16, 10),
    ]
    repeat_band(76, 190, red, red_people)
    draw.rectangle((0, 154, width, 160), fill=blue)

    blue_branch = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 2), (4, 3), (5, 2), (6, 1), (6, 0), (7, 0), (8, 0)]
    repeat_band(176, 74, blue, blue_branch)

    red_mountains = [(0, 8), (1, 7), (2, 6), (3, 5), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (1, 8), (2, 8), (6, 8), (7, 8)]
    repeat_band(240, 76, red, red_mountains)
    draw.rectangle((0, 312, width, 318), fill=blue)

    blue_crosses = [(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)]
    repeat_band(336, 40, blue, blue_crosses)

    red_birds = [
        (0, 4), (1, 3), (2, 2), (3, 1), (4, 2), (5, 3), (6, 4),
        (12, 1), (12, 2), (12, 3), (11, 4), (10, 5), (11, 6), (12, 6), (13, 6), (14, 5), (13, 4),
        (19, 4), (20, 3), (21, 2), (22, 1), (23, 2), (24, 3), (25, 4),
    ]
    repeat_band(386, 210, red, red_birds)
    draw.rectangle((0, 468, width, 474), fill=blue)
    repeat_band(495, 70, blue, [(0, 0), (1, 1), (2, 2), (3, 2), (4, 1), (5, 0), (2, 3), (3, 3)])

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


def make_symbol_sheet() -> Image.Image:
    cell_w, cell_h = 220, 170
    sheet = Image.new("RGBA", (cell_w * 4, cell_h * 2), (255, 255, 255, 255))
    for index, name in enumerate(SYMBOLS):
        symbol = draw_symbol(name, 155)
        x = (index % 4) * cell_w + 34
        y = (index // 4) * cell_h + 8
        sheet.alpha_composite(symbol, (x, y))
    return sheet


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


def fit_to_canvas(image: Image.Image, size: tuple[int, int] = CANVAS_SIZE) -> Image.Image:
    image = image.convert("RGBA")
    image.thumbnail(size, Image.LANCZOS)
    canvas = textured_background(size)
    canvas.alpha_composite(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
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


@st.cache_data(show_spinner=False, ttl=3600)
def load_mockup_from_url(url: str) -> Image.Image:
    request = Request(url, headers={"User-Agent": "OrnaMap school research app"})
    with urlopen(request, timeout=12) as response:
        data = response.read()
    return Image.open(BytesIO(data)).convert("RGBA")


def fit_photo_mockup(image: Image.Image, size: tuple[int, int] = CANVAS_SIZE) -> Image.Image:
    image = ImageOps.contain(image.convert("RGBA"), size, Image.LANCZOS)
    canvas = Image.new("RGBA", size, (246, 244, 238, 255))
    canvas.alpha_composite(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_fallback_mockup(name: str) -> Image.Image:
    image = textured_background(CANVAS_SIZE)
    draw = ImageDraw.Draw(image)

    if name == "Шоппер":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (275, 145, 705, 610), 32))
        draw.rounded_rectangle((275, 145, 705, 610), radius=32, fill=(238, 231, 213), outline=(185, 172, 148), width=5)
        draw.arc((375, 40, 605, 250), 180, 360, fill=(126, 104, 76), width=14)
        draw.line((375, 145, 375, 195), fill=(126, 104, 76), width=14)
        draw.line((605, 145, 605, 195), fill=(126, 104, 76), width=14)
        draw.rounded_rectangle((310, 188, 670, 552), radius=18, outline=(255, 255, 255, 120), width=2)
    elif name == "Блокнот":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (220, 130, 760, 555), 20))
        draw.rounded_rectangle((220, 130, 760, 555), radius=20, fill=(248, 246, 238), outline=(198, 190, 176), width=5)
        draw.line((245, 130, 245, 555), fill=(190, 182, 168), width=8)
    elif name == "Футболка":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (265, 125, 715, 622), 40))
        draw.polygon((320, 140, 415, 96, 490, 166, 565, 96, 660, 140, 735, 275, 670, 320, 670, 620, 310, 620, 310, 320, 245, 275), fill=(245, 245, 239), outline=(164, 166, 157))
        draw.arc((430, 112, 550, 220), 0, 180, fill=(164, 166, 157), width=6)
        draw.line((430, 165, 490, 205, 550, 165), fill=(220, 220, 212), width=4)
    elif name == "Кружка":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (305, 215, 690, 542), 45))
        draw.rounded_rectangle((305, 215, 690, 542), radius=50, fill=(248, 249, 245), outline=(178, 185, 184), width=6)
        draw.arc((630, 280, 830, 485), 270, 90, fill=(178, 185, 184), width=20)
        draw.rectangle((330, 224, 668, 286), fill=(252, 253, 249))
        draw.rounded_rectangle((340, 295, 655, 505), radius=24, outline=(223, 229, 225), width=2)
    else:
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (190, 95, 790, 610), 16))
        draw.rectangle((190, 95, 790, 610), fill=(250, 248, 240), outline=(112, 96, 75), width=12)
        draw.rectangle((212, 117, 768, 588), outline=(196, 188, 170), width=5)
    return image


def get_ready_mockup(name: str) -> tuple[Image.Image, bool]:
    try:
        return fit_photo_mockup(load_mockup_from_url(MOCKUPS[name]["url"])), True
    except (OSError, URLError, TimeoutError, ValueError):
        return make_fallback_mockup(name), False


def place_pattern_on_mockup(mockup: Image.Image, pattern: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    fitted = pattern.resize((right - left, bottom - top), Image.LANCZOS)
    result = mockup.copy()
    result.alpha_composite(fitted, (left, top))
    shine = Image.new("RGBA", result.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(shine)
    draw.rounded_rectangle((left, top, right, bottom), radius=18, outline=(255, 255, 255, 80), width=2)
    result.alpha_composite(shine)
    return result


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("OrnaMap")
st.caption("Полосные северные орнаменты и отдельные знаки для примерки на макетах")

with st.sidebar:
    st.subheader("Орнамент")
    ornament_type = st.radio("Тип узора", ["Полосной орнамент", "Отдельный рисунок", "Загрузить свой"], horizontal=False)
    selected_symbol = st.selectbox("Рисунок из второго изображения", SYMBOLS, disabled=ornament_type != "Отдельный рисунок")
    uploaded_ornament = st.file_uploader(
        "Загрузите свой орнамент",
        type=["png", "jpg", "jpeg", "webp"],
        disabled=ornament_type != "Загрузить свой",
    )

    st.subheader("Макет")
    mockup_mode = st.radio("Источник макета", ["Готовый", "Загрузить свой"], horizontal=True)
    mockup_name = st.selectbox("Готовый макет", list(MOCKUPS.keys()), disabled=mockup_mode != "Готовый")
    uploaded_mockup = st.file_uploader(
        "Загрузите изображение макета",
        type=["png", "jpg", "jpeg", "webp"],
        disabled=mockup_mode != "Загрузить свой",
    )

    color_count = st.slider("Сколько основных цветов найти", 2, 6, 4)
    repeats = st.slider("Повторы узора", 1, 7, 3 if ornament_type == "Отдельный рисунок" else 4)
    opacity = st.slider("Прозрачность наложения", 35, 100, 88)
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
    recolored = recolor_image(source, palette, opacity)
    tile = Image.alpha_composite(recolored, edge_map(source, edge_strength))
    pattern = make_pattern_tile(tile, repeats, mirror)
    source_caption = "Полосной орнамент по первому изображению"
elif ornament_type == "Отдельный рисунок":
    source = fit_to_square(symbol)
    palette = find_palette(source, color_count)
    pattern = make_symbol_pattern(symbol, repeats, opacity)
    source_caption = f"Вырезанный рисунок: {selected_symbol}"
else:
    fallback = strip
    source = fit_to_square(open_uploaded_or_fallback(uploaded_ornament, fallback))
    palette = find_palette(source, color_count)
    recolored = recolor_image(source, palette, opacity)
    tile = Image.alpha_composite(recolored, edge_map(source, edge_strength))
    pattern = make_pattern_tile(tile, repeats, mirror)
    source_caption = "Пользовательский орнамент"

if mockup_mode == "Загрузить свой":
    mockup_base = fit_to_canvas(open_uploaded_or_fallback(uploaded_mockup, make_fallback_mockup("Плакат")))
    placement_box = custom_box or (300, 180, 660, 480)
    result_caption = "Пользовательский макет"
    mockup_loaded = True
else:
    mockup_base, mockup_loaded = get_ready_mockup(mockup_name)
    placement_box = MOCKUPS[mockup_name]["box"]
    result_caption = f"Макет: {mockup_name}"

result = place_pattern_on_mockup(mockup_base, pattern, placement_box)

left, right = st.columns([1, 1.25])

with left:
    st.subheader("Источник узора")
    st.image(source, caption=source_caption, width="stretch")
    swatches = "".join(
        f"<span style='display:inline-block;width:54px;height:38px;background:rgb{color};"
        "border-radius:10px;border:1px solid rgba(32,57,69,0.18);margin-right:8px'></span>"
        for color in palette
    )
    st.markdown(swatches, unsafe_allow_html=True)
    st.markdown(
        "<p class='source-note'>Для полосы строится повторяющийся паттерн. Для второго изображения отдельные знаки уже разнесены по списку и накладываются как самостоятельные мотивы.</p>",
        unsafe_allow_html=True,
    )

with right:
    st.subheader("Примерка на макете")
    st.image(result, caption=result_caption, width="stretch")
    if mockup_mode == "Готовый" and not mockup_loaded:
        st.warning("Готовое фото макета не загрузилось, поэтому показана запасная светлая схема.")
    st.download_button(
        "Скачать результат PNG",
        data=image_to_png_bytes(result),
        file_name="ornamap_result.png",
        mime="image/png",
    )

st.divider()

st.subheader("Разрезанные рисунки из второго изображения")
symbol_cols = st.columns(4)
for index, name in enumerate(SYMBOLS):
    with symbol_cols[index % 4]:
        st.image(draw_symbol(name), caption=name, width="stretch")

tab1, tab2, tab3 = st.tabs(["Как работает", "Для защиты", "Что улучшить дальше"])

with tab1:
    st.markdown(
        """
        1. Первый референс используется как полосной орнамент: программа повторяет его как декоративную ленту.
        2. Второй референс представлен как набор отдельных рисунков: каждый знак можно выбрать отдельно.
        3. KMeans находит основные цвета, а фильтр контуров усиливает линии.
        4. Итоговый паттерн накладывается на выбранный макет и скачивается как PNG.
        """
    )

with tab2:
    st.markdown(
        """
        **Фраза для выступления:** я взяла два типа северных визуальных мотивов: полосный орнамент
        и отдельные знаки. В приложении они превращаются в цифровые элементы дизайна, которые можно
        примерить на предметы.

        **Информатика здесь:** изображение представлено пикселями, цвета анализируются алгоритмом KMeans,
        а макет собирается программно из нескольких слоев.
        """
    )

with tab3:
    st.markdown(
        """
        - добавить автоматическое разрезание загруженной фотографии с отдельными знаками;
        - сохранять библиотеку пользовательских мотивов;
        - распознавать тип узора: полоса, одиночный знак, сетка;
        - добавить экспорт не только PNG, но и PDF-лист для печати.
        """
    )

st.caption(
    "Готовые фото макетов загружаются с Wikimedia Commons: Tote Bag.jpeg, Blank Notebook.jpg, "
    "Blank poster for Fedoruk screenings, White T shirt.jpg, Taza Blanca.jpg."
)
