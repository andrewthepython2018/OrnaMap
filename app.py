from io import BytesIO
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="OrnaMap",
    page_icon="OM",
    layout="wide",
)


CANVAS_SIZE = (980, 680)

MOCKUPS = {
    "Шоппер": {"box": (318, 195, 662, 530), "color": (238, 231, 213), "accent": (126, 104, 76)},
    "Обложка блокнота": {"box": (335, 135, 630, 555), "color": (52, 67, 75), "accent": (226, 180, 78)},
    "Плакат": {"box": (210, 112, 770, 590), "color": (250, 247, 238), "accent": (112, 96, 75)},
    "Футболка": {"box": (350, 210, 630, 460), "color": (243, 243, 235), "accent": (164, 166, 157)},
    "Кружка": {"box": (355, 265, 625, 455), "color": (248, 249, 245), "accent": (178, 185, 184)},
}

ORNAMENTS = {
    "Северные ромбы": "rhombs",
    "Полярная звезда": "star",
    "Лента тундры": "ribbon",
    "Снежная сетка": "snow",
    "Каменный берег": "shore",
}

ONLINE_ORNAMENTS = {
    "Саамские костюмы, Ловозеро": "https://commons.wikimedia.org/wiki/Special:FilePath/S%C3%A1mi%20traditional%20costumes%2C%20Lovozero%2C%20Kola%20Peninsula%2C%20Russia.jpg?width=900",
    "Саамская деталь одежды": "https://commons.wikimedia.org/wiki/Special:FilePath/Sami%2C%20colletto%20di%20giubba%20con%20pendenti%20in%20argento%2C%20xviii%20secolo.jpg?width=900",
}

ONLINE_MOCKUPS = {
    "Фото шоппера": "https://commons.wikimedia.org/wiki/Special:FilePath/Tote%20Bag.jpeg?width=900",
    "Фото белой кружки": "https://commons.wikimedia.org/wiki/Special:FilePath/Taza%20Blanca.jpg?width=900",
}


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 5%, rgba(190, 47, 54, 0.08), transparent 28%),
            linear-gradient(135deg, #f7f2e8 0%, #edf3f2 48%, #f9f7f0 100%);
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }
    h1 {
        font-size: 3.4rem !important;
        line-height: 1 !important;
        color: #203945;
    }
    h2, h3 {
        color: #203945;
    }
    [data-testid="stSidebar"] {
        background: #f4f0e8;
        border-right: 1px solid rgba(56, 73, 81, 0.12);
    }
    div[data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 18px 45px rgba(40, 48, 52, 0.12);
    }
    .source-note {
        color: #637077;
        font-size: 0.92rem;
        margin-top: -0.35rem;
    }
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


def make_ornament(name: str, size: int = 480) -> Image.Image:
    kind = ORNAMENTS[name]
    image = Image.new("RGBA", (size, size), (246, 241, 230, 255))
    draw = ImageDraw.Draw(image)
    navy = (26, 66, 82, 255)
    red = (172, 43, 50, 255)
    gold = (229, 178, 75, 255)
    ice = (218, 229, 224, 255)

    if kind == "star":
        step = size // 6
        colors = [navy, red, gold]
        for row in range(6):
            for col in range(6):
                x = col * step + step // 2
                y = row * step + step // 2
                r = step // 3
                color = colors[(row * 2 + col) % 3]
                draw.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=color)
                draw.line((x - r, y, x + r, y), fill=(246, 241, 230, 255), width=5)
                draw.line((x, y - r, x, y + r), fill=(246, 241, 230, 255), width=5)

    elif kind == "ribbon":
        band = size // 8
        colors = [navy, red, gold, ice]
        for y in range(0, size, band):
            draw.rectangle((0, y, size, y + band), fill=colors[(y // band) % len(colors)])
            for x in range(-band, size + band, band):
                draw.line(
                    [(x, y + band), (x + band // 2, y + band // 4), (x + band, y + band)],
                    fill=(246, 241, 230, 255),
                    width=7,
                    joint="curve",
                )

    elif kind == "snow":
        step = size // 9
        for offset in range(-size, size * 2, step):
            draw.line((offset, 0, offset + size, size), fill=navy, width=5)
            draw.line((offset + size, 0, offset, size), fill=ice, width=5)
        for row in range(1, 9, 2):
            for col in range(1, 9, 2):
                x = col * step
                y = row * step
                draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=red)

    elif kind == "shore":
        colors = [navy, (58, 105, 106, 255), red, gold, ice]
        step = size // 10
        for y in range(0, size, step):
            color = colors[(y // step) % len(colors)]
            draw.rectangle((0, y, size, y + step), fill=color)
            for x in range(0, size, step):
                if (x // step + y // step) % 2 == 0:
                    draw.polygon([(x, y), (x + step, y), (x + step // 2, y + step)], fill=(246, 241, 230, 90))

    else:
        step = size // 7
        colors = [navy, red, gold]
        for row in range(7):
            for col in range(7):
                x = col * step + step // 2
                y = row * step + step // 2
                r = step // 3
                draw.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=colors[(row + col) % 3])
                if (row + col) % 2 == 0:
                    draw.line((x - r, y, x + r, y), fill=(246, 241, 230, 255), width=5)
        for offset in range(0, size, step):
            draw.line((0, offset, size, offset), fill=(26, 66, 82, 60), width=2)
            draw.line((offset, 0, offset, size), fill=(26, 66, 82, 60), width=2)

    return image


@st.cache_data(show_spinner=False, ttl=3600)
def load_online_image(url: str) -> Image.Image:
    request = Request(url, headers={"User-Agent": "OrnaMap school research app"})
    with urlopen(request, timeout=12) as response:
        data = response.read()
    return Image.open(BytesIO(data)).convert("RGBA")


def load_online_or_fallback(url: str, fallback: Image.Image) -> tuple[Image.Image, bool]:
    try:
        return load_online_image(url), True
    except (OSError, URLError, TimeoutError, ValueError):
        return fallback, False


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


def draw_mockup(name: str) -> Image.Image:
    data = MOCKUPS[name]
    image = textured_background(CANVAS_SIZE)
    draw = ImageDraw.Draw(image)

    if name == "Шоппер":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (275, 145, 705, 610), 32))
        draw.rounded_rectangle((275, 145, 705, 610), radius=32, fill=data["color"], outline=(185, 172, 148), width=5)
        draw.arc((375, 40, 605, 250), 180, 360, fill=data["accent"], width=14)
        draw.line((375, 145, 375, 195), fill=data["accent"], width=14)
        draw.line((605, 145, 605, 195), fill=data["accent"], width=14)
        draw.rounded_rectangle((310, 188, 670, 552), radius=18, outline=(255, 255, 255, 120), width=2)

    elif name == "Обложка блокнота":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (330, 105, 650, 610), 20))
        draw.rounded_rectangle((330, 105, 650, 610), radius=20, fill=data["color"])
        draw.rectangle((365, 105, 386, 610), fill=data["accent"])
        draw.rounded_rectangle((348, 130, 632, 585), radius=10, outline=(255, 255, 255, 45), width=2)
        for y in range(148, 575, 48):
            draw.ellipse((315, y, 342, y + 27), fill=(30, 37, 42))

    elif name == "Футболка":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (265, 125, 715, 622), 40))
        draw.polygon((320, 140, 415, 96, 490, 166, 565, 96, 660, 140, 735, 275, 670, 320, 670, 620, 310, 620, 310, 320, 245, 275), fill=data["color"], outline=data["accent"])
        draw.arc((430, 112, 550, 220), 0, 180, fill=data["accent"], width=6)
        draw.line((430, 165, 490, 205, 550, 165), fill=(220, 220, 212), width=4)

    elif name == "Кружка":
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (305, 215, 690, 542), 45))
        draw.rounded_rectangle((305, 215, 690, 542), radius=50, fill=data["color"], outline=data["accent"], width=6)
        draw.arc((630, 280, 830, 485), 270, 90, fill=data["accent"], width=20)
        draw.rectangle((330, 224, 668, 286), fill=(252, 253, 249))
        draw.rounded_rectangle((340, 295, 655, 505), radius=24, outline=(223, 229, 225), width=2)

    else:
        image.alpha_composite(shadow_layer(CANVAS_SIZE, (190, 95, 790, 610), 16))
        draw.rectangle((190, 95, 790, 610), fill=data["color"], outline=(112, 96, 75), width=12)
        draw.rectangle((212, 117, 768, 588), outline=(196, 188, 170), width=5)

    return image


def place_pattern_on_mockup(mockup: Image.Image, pattern: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    target_size = (right - left, bottom - top)
    fitted = pattern.resize(target_size, Image.LANCZOS)
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
st.caption("Исследуем северный орнамент и сразу примеряем его на дизайнерский макет")

with st.sidebar:
    st.subheader("Орнамент")
    ornament_mode = st.radio(
        "Источник орнамента",
        ["Готовый", "Из интернета", "Загрузить свой"],
        horizontal=False,
    )

    ornament_name = st.selectbox("Готовый орнамент", list(ORNAMENTS.keys()), disabled=ornament_mode != "Готовый")
    online_ornament_name = st.selectbox(
        "Онлайн-изображение",
        list(ONLINE_ORNAMENTS.keys()),
        disabled=ornament_mode != "Из интернета",
    )
    uploaded_ornament = st.file_uploader(
        "Загрузите изображение орнамента",
        type=["png", "jpg", "jpeg", "webp"],
        disabled=ornament_mode != "Загрузить свой",
    )

    st.subheader("Макет")
    mockup_mode = st.radio(
        "Источник макета",
        ["Готовый", "Из интернета", "Загрузить свой"],
        horizontal=False,
    )

    mockup_name = st.selectbox("Готовый макет", list(MOCKUPS.keys()), disabled=mockup_mode != "Готовый")
    online_mockup_name = st.selectbox("Онлайн-макет", list(ONLINE_MOCKUPS.keys()), disabled=mockup_mode != "Из интернета")
    uploaded_mockup = st.file_uploader(
        "Загрузите изображение макета",
        type=["png", "jpg", "jpeg", "webp"],
        disabled=mockup_mode != "Загрузить свой",
    )

    color_count = st.slider("Сколько основных цветов найти", 2, 6, 4)
    repeats = st.slider("Повторы узора", 2, 7, 4)
    opacity = st.slider("Прозрачность наложения", 35, 100, 88)
    edge_strength = st.slider("Сила выделения контура", 1, 12, 5)
    mirror = st.toggle("Чередовать зеркальные плитки", value=True)

    custom_box = None
    if mockup_mode != "Готовый":
        st.subheader("Область наложения")
        x = st.slider("Смещение по горизонтали", 0, 760, 300)
        y = st.slider("Смещение по вертикали", 0, 520, 180)
        w = st.slider("Ширина области", 120, 720, 360)
        h = st.slider("Высота области", 120, 500, 300)
        custom_box = (x, y, min(CANVAS_SIZE[0], x + w), min(CANVAS_SIZE[1], y + h))


offline_ornament = make_ornament(ornament_name)
online_loaded = None
if ornament_mode == "Из интернета":
    source_raw, online_loaded = load_online_or_fallback(ONLINE_ORNAMENTS[online_ornament_name], offline_ornament)
elif ornament_mode == "Загрузить свой":
    source_raw = open_uploaded_or_fallback(uploaded_ornament, offline_ornament)
else:
    source_raw = offline_ornament
source = fit_to_square(source_raw)

if mockup_mode == "Из интернета":
    fallback_mockup = draw_mockup("Кружка" if "кружки" in online_mockup_name else "Шоппер")
    mockup_raw, mockup_online_loaded = load_online_or_fallback(ONLINE_MOCKUPS[online_mockup_name], fallback_mockup)
    mockup_base = fit_to_canvas(mockup_raw)
    placement_box = custom_box or (300, 180, 660, 480)
    result_caption = f"Онлайн-макет: {online_mockup_name}"
elif mockup_mode == "Загрузить свой":
    mockup_base = fit_to_canvas(open_uploaded_or_fallback(uploaded_mockup, draw_mockup("Плакат")))
    placement_box = custom_box or (300, 180, 660, 480)
    result_caption = "Пользовательский макет"
else:
    mockup_online_loaded = True
    mockup_base = draw_mockup(mockup_name)
    placement_box = MOCKUPS[mockup_name]["box"]
    result_caption = f"Макет: {mockup_name}"

palette = find_palette(source, color_count)
recolored = recolor_image(source, palette, opacity)
edges = edge_map(source, edge_strength)
tile = Image.alpha_composite(recolored, edges)
pattern = make_pattern_tile(tile, repeats, mirror)
result = place_pattern_on_mockup(mockup_base, pattern, placement_box)

left, right = st.columns([1, 1.25])

with left:
    st.subheader("Анализ орнамента")
    st.image(source, caption="Источник узора", width="stretch")
    swatches = "".join(
        f"<span style='display:inline-block;width:54px;height:38px;background:rgb{color};"
        "border-radius:10px;border:1px solid rgba(32,57,69,0.18);margin-right:8px'></span>"
        for color in palette
    )
    st.markdown(swatches, unsafe_allow_html=True)
    st.markdown("<p class='source-note'>KMeans находит основные цветовые группы, а фильтр контуров усиливает форму узора.</p>", unsafe_allow_html=True)
    if ornament_mode == "Из интернета" and not online_loaded:
        st.warning("Онлайн-изображение не загрузилось, поэтому показан встроенный орнамент.")

with right:
    st.subheader("Примерка на макете")
    st.image(result, caption=result_caption, width="stretch")
    if mockup_mode == "Из интернета" and not mockup_online_loaded:
        st.warning("Онлайн-макет не загрузился, поэтому показан встроенный макет.")
    st.download_button(
        "Скачать результат PNG",
        data=image_to_png_bytes(result),
        file_name="ornamap_result.png",
        mime="image/png",
    )

st.divider()

gallery_cols = st.columns(5)
for index, name in enumerate(MOCKUPS):
    with gallery_cols[index % 5]:
        st.image(draw_mockup(name), caption=name, width="stretch")

tab1, tab2, tab3 = st.tabs(["Как работает", "Источники", "Для защиты"])

with tab1:
    st.markdown(
        """
        1. Пользователь выбирает готовый орнамент, онлайн-изображение или загружает свой файл.
        2. Программа переводит изображение в массив пикселей и ищет основные цвета методом KMeans.
        3. Фильтр контуров выделяет линии, чтобы узор читался лучше.
        4. Из фрагмента создается повторяющийся паттерн.
        5. Паттерн накладывается на готовый, онлайн или пользовательский макет.
        """
    )

with tab2:
    st.markdown(
        """
        Онлайн-режим использует открытые изображения с Wikimedia Commons. Если сеть недоступна,
        приложение автоматически возвращается к встроенным изображениям, чтобы демонстрация не сорвалась.

        Для школьной работы важно проговорить: реальные культурные орнаменты нужно использовать бережно,
        с указанием источника и без искажения смысла.
        """
    )
    st.markdown("- Wikimedia Commons: Kola Peninsula, White mugs, Tote bags")

with tab3:
    st.markdown(
        """
        **Коротко для рассказа:** я создала веб-приложение, где можно выбрать или загрузить орнамент,
        программа находит его палитру, усиливает контуры и показывает, как узор будет выглядеть на макете.

        **Что здесь от информатики:** изображение хранится как пиксели, цвета анализируются алгоритмом KMeans,
        а результат собирается программно в новом изображении.
        """
    )
