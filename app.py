from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="OrnaMap",
    page_icon="OM",
    layout="wide",
)


MOCKUPS = {
    "Шоппер": {"size": (900, 620), "box": (255, 155, 645, 500), "color": (239, 232, 213)},
    "Обложка блокнота": {"size": (900, 620), "box": (300, 105, 610, 520), "color": (55, 67, 77)},
    "Плакат": {"size": (900, 620), "box": (185, 90, 715, 535), "color": (244, 241, 232)},
}


def make_default_ornament(size: int = 420) -> Image.Image:
    """Создает простой северный орнамент, если пользователь не загрузил свой."""
    image = Image.new("RGBA", (size, size), (246, 241, 230, 255))
    draw = ImageDraw.Draw(image)
    colors = [(32, 72, 94, 255), (168, 42, 48, 255), (229, 178, 75, 255)]
    step = size // 7

    for row in range(7):
        for col in range(7):
            x = col * step + step // 2
            y = row * step + step // 2
            color = colors[(row + col) % len(colors)]
            r = step // 3
            points = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
            draw.polygon(points, fill=color)
            if (row + col) % 2 == 0:
                draw.line((x - r, y, x + r, y), fill=(246, 241, 230, 255), width=4)

    for offset in range(0, size, step):
        draw.line((0, offset, size, offset), fill=(32, 72, 94, 70), width=2)
        draw.line((offset, 0, offset, size), fill=(32, 72, 94, 70), width=2)

    return image


def open_uploaded_image(uploaded_file) -> Image.Image:
    if uploaded_file is None:
        return make_default_ornament()
    return Image.open(uploaded_file).convert("RGBA")


def resize_to_square(image: Image.Image, size: int = 420) -> Image.Image:
    image = image.convert("RGBA")
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.alpha_composite(image, (x, y))
    return canvas


def find_palette(image: Image.Image, color_count: int) -> list[tuple[int, int, int]]:
    rgb = image.convert("RGB").resize((140, 140))
    pixels = np.array(rgb).reshape(-1, 3)

    model = KMeans(n_clusters=color_count, n_init=10, random_state=42)
    model.fit(pixels)

    centers = model.cluster_centers_.astype(int)
    labels, counts = np.unique(model.labels_, return_counts=True)
    order = labels[np.argsort(counts)[::-1]]
    return [tuple(centers[label]) for label in order]


def edge_map(image: Image.Image, strength: int) -> Image.Image:
    gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
    gray = ImageEnhance.Contrast(gray).enhance(1 + strength / 20)
    alpha = gray.point(lambda value: 255 if value > 32 else 0)
    edges = Image.new("RGBA", image.size, (23, 48, 60, 0))
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

    recolored = Image.fromarray(result.astype("uint8"), "RGB").convert("RGBA")
    source_alpha = np.array(image.getchannel("A"))
    recolored.putalpha((source_alpha * (opacity / 100)).astype("uint8"))
    return recolored


def make_pattern_tile(image: Image.Image, repeats: int, mirror: bool) -> Image.Image:
    tile_size = 210
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
    width, height = data["size"]
    image = Image.new("RGBA", (width, height), (229, 234, 232, 255))
    draw = ImageDraw.Draw(image)

    if name == "Шоппер":
        draw.rounded_rectangle((245, 145, 655, 535), radius=24, fill=data["color"], outline=(188, 178, 154), width=4)
        draw.arc((350, 45, 550, 230), 180, 360, fill=(129, 115, 92), width=12)
        draw.line((350, 140, 350, 175), fill=(129, 115, 92), width=12)
        draw.line((550, 140, 550, 175), fill=(129, 115, 92), width=12)
    elif name == "Обложка блокнота":
        draw.rounded_rectangle((295, 100, 615, 525), radius=12, fill=data["color"])
        draw.rectangle((327, 100, 345, 525), fill=(230, 196, 94))
        for y in range(135, 500, 45):
            draw.ellipse((285, y, 305, y + 20), fill=(28, 34, 39))
    else:
        draw.rectangle((175, 80, 725, 545), fill=(250, 248, 240), outline=(196, 188, 170), width=6)
        draw.rectangle((165, 70, 735, 555), outline=(104, 91, 75), width=10)

    return image


def place_pattern_on_mockup(mockup: Image.Image, pattern: Image.Image, name: str) -> Image.Image:
    left, top, right, bottom = MOCKUPS[name]["box"]
    target_size = (right - left, bottom - top)
    fitted = pattern.resize(target_size, Image.LANCZOS)
    result = mockup.copy()
    result.alpha_composite(fitted, (left, top))
    return result


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("OrnaMap")
st.caption("Веб-сервис для исследования орнамента и примерки узора на дизайнерские макеты")

with st.sidebar:
    uploaded = st.file_uploader("Загрузите изображение орнамента", type=["png", "jpg", "jpeg", "webp"])
    color_count = st.slider("Сколько основных цветов найти", 2, 6, 4)
    repeats = st.slider("Повторы узора", 2, 6, 4)
    opacity = st.slider("Прозрачность наложения", 35, 100, 88)
    edge_strength = st.slider("Сила выделения контура", 1, 12, 5)
    mirror = st.toggle("Чередовать зеркальные плитки", value=True)
    mockup_name = st.selectbox("Макет", list(MOCKUPS.keys()))

source = resize_to_square(open_uploaded_image(uploaded))
palette = find_palette(source, color_count)
recolored = recolor_image(source, palette, opacity)
edges = edge_map(source, edge_strength)
tile = Image.alpha_composite(recolored, edges)
pattern = make_pattern_tile(tile, repeats, mirror)
mockup = place_pattern_on_mockup(draw_mockup(mockup_name), pattern, mockup_name)

left, right = st.columns([1, 1.25])

with left:
    st.subheader("Анализ изображения")
    st.image(source, caption="Исходный орнамент", use_container_width=True)
    swatches = "".join(
        f"<span style='display:inline-block;width:52px;height:36px;background:rgb{color};"
        "border-radius:6px;border:1px solid #d0d0d0;margin-right:8px'></span>"
        for color in palette
    )
    st.markdown(swatches, unsafe_allow_html=True)
    st.caption("Алгоритм KMeans группирует похожие пиксели и находит главные цвета.")

with right:
    st.subheader("Примерка орнамента")
    st.image(mockup, caption=f"Макет: {mockup_name}", use_container_width=True)
    st.download_button(
        "Скачать результат PNG",
        data=image_to_png_bytes(mockup),
        file_name="ornamap_result.png",
        mime="image/png",
    )

st.divider()

tab1, tab2, tab3 = st.tabs(["Как работает", "Для научной работы", "Идеи развития"])

with tab1:
    st.markdown(
        """
        1. Пользователь загружает фотографию или рисунок орнамента.
        2. Программа уменьшает изображение до удобного размера и переводит его в массив пикселей.
        3. Метод KMeans ищет группы похожих цветов. Так получается палитра узора.
        4. Фильтр контуров подчеркивает линии, чтобы орнамент был заметнее.
        5. Из одного фрагмента создается повторяющийся паттерн и накладывается на макет.
        """
    )

with tab2:
    st.markdown(
        """
        **Гипотеза:** если автоматизировать подбор палитры и наложение орнамента, дизайнеру будет проще
        быстро проверить, как национальный узор смотрится на разных предметах.

        **Объект исследования:** цифровые изображения орнаментов.

        **Предмет исследования:** способы анализа цвета и повторения узора в веб-приложении.

        **Практическая польза:** приложение можно использовать на уроках информатики, технологии,
        в школьных проектах о культуре Кольского полуострова и в первых дизайнерских макетах.
        """
    )

with tab3:
    st.markdown(
        """
        - добавить библиотеку орнаментов народов Севера;
        - научить программу аккуратно вырезать фон;
        - сохранять историю созданных макетов;
        - добавить распознавание типа симметрии узора;
        - сделать отдельный режим для печати на ткани.
        """
    )
