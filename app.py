from io import BytesIO
from math import cos, pi, sin

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans


st.set_page_config(page_title="OrnaMap AR", page_icon="OM", layout="wide")

CANVAS = (1280, 820)
SURFACE_QUAD = ((420, 185), (945, 248), (872, 657), (330, 570))

PALETTES = {
    "Красный, синий, снег": [(225, 28, 48), (28, 68, 154), (255, 255, 250), (24, 42, 55)],
    "Полярная ночь": [(18, 31, 45), (47, 101, 128), (232, 247, 244), (205, 42, 68)],
    "Морошка и лед": [(230, 83, 53), (249, 179, 69), (235, 247, 244), (30, 67, 89)],
    "Черный знак": [(12, 14, 16), (248, 247, 241), (210, 219, 214), (168, 33, 49)],
}

MOTIFS = ["Ромб", "Чум", "Солнце", "Лодка", "След", "Фигура"]


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(229, 40, 63, .15), transparent 22%),
            radial-gradient(circle at 86% 4%, rgba(39, 119, 145, .16), transparent 24%),
            linear-gradient(135deg, #f7f2e8 0%, #edf4f2 48%, #fffaf1 100%);
    }
    .block-container { max-width: 1320px; padding-top: 1.3rem; }
    h1 {
        color: #172b35;
        font-size: 4.2rem !important;
        line-height: .95 !important;
        letter-spacing: 0 !important;
        margin-bottom: .2rem !important;
    }
    h2, h3 { color: #172b35; letter-spacing: 0 !important; }
    [data-testid="stSidebar"] {
        background: #f4efe4;
        border-right: 1px solid rgba(23, 43, 53, .12);
    }
    div[data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 22px 60px rgba(24, 36, 42, .16);
    }
    .hero {
        padding: 24px 0 10px 0;
        border-bottom: 1px solid rgba(23, 43, 53, .12);
        margin-bottom: 22px;
    }
    .lead {
        max-width: 850px;
        color: #52646c;
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .chip {
        display: inline-block;
        padding: 7px 11px;
        margin: 0 7px 7px 0;
        border-radius: 999px;
        background: rgba(255, 255, 255, .62);
        border: 1px solid rgba(23, 43, 53, .12);
        color: #33464f;
        font-size: .92rem;
    }
    .metric {
        padding: 14px 16px;
        background: rgba(255, 255, 255, .68);
        border: 1px solid rgba(23, 43, 53, .1);
        border-radius: 16px;
    }
    .small { color: #66757c; font-size: .92rem; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return color[0], color[1], color[2], alpha


def paper_texture(size: tuple[int, int], base=(248, 246, 236)) -> Image.Image:
    width, height = size
    yy = np.linspace(0, 1, height)[:, None]
    xx = np.linspace(0, 1, width)[None, :]
    arr = np.zeros((height, width, 3), dtype=np.float32)
    arr[..., 0] = base[0] - 10 * yy + 4 * xx
    arr[..., 1] = base[1] - 8 * yy + 5 * xx
    arr[..., 2] = base[2] - 7 * yy + 8 * xx
    noise = np.random.default_rng(42).normal(0, 2.1, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8)).convert("RGBA")


def draw_motif(draw: ImageDraw.ImageDraw, kind: str, cx: int, cy: int, cell: int, color) -> None:
    w = max(3, cell // 12)
    r = cell // 2 - 5
    if kind == "Ромб":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], outline=color, width=w)
        draw.polygon([(cx, cy - r // 2), (cx + r // 2, cy), (cx, cy + r // 2), (cx - r // 2, cy)], fill=color)
    elif kind == "Чум":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], outline=color, width=w)
        draw.line((cx, cy - r, cx, cy + r), fill=color, width=w)
        draw.arc((cx - r // 2, cy, cx + r // 2, cy + r + 8), 205, 335, fill=color, width=w)
    elif kind == "Солнце":
        draw.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), outline=color, width=w)
        for i in range(8):
            a = i * pi / 4
            draw.line((cx + cos(a) * r * .62, cy + sin(a) * r * .62, cx + cos(a) * r, cy + sin(a) * r), fill=color, width=w)
    elif kind == "Лодка":
        draw.arc((cx - r, cy - r // 4, cx + r, cy + r), 15, 165, fill=color, width=w + 1)
        draw.line((cx, cy - r, cx, cy + r // 3), fill=color, width=w)
        draw.line((cx, cy - r // 2, cx - r // 2, cy), fill=color, width=w)
        draw.line((cx, cy - r // 2, cx + r // 2, cy), fill=color, width=w)
    elif kind == "След":
        draw.line((cx - r, cy + r, cx + r, cy + r), fill=color, width=w)
        draw.line((cx, cy - r, cx, cy + r), fill=color, width=w)
        draw.line((cx, cy - r, cx - r // 2, cy - r // 3), fill=color, width=w)
        draw.line((cx, cy - r, cx + r // 2, cy - r // 3), fill=color, width=w)
    else:
        draw.ellipse((cx - r // 3, cy - r, cx + r // 3, cy - r // 3), fill=color)
        draw.line((cx, cy - r // 3, cx, cy + r // 2), fill=color, width=w)
        draw.line((cx - r, cy, cx + r, cy), fill=color, width=w)
        draw.line((cx, cy + r // 2, cx - r // 2, cy + r), fill=color, width=w)
        draw.line((cx, cy + r // 2, cx + r // 2, cy + r), fill=color, width=w)


def generate_ornament(seed: int, palette_name: str, style: str, density: int, symmetry: bool, accent: str, size: int = 920) -> Image.Image:
    rng = np.random.default_rng(seed)
    palette = PALETTES[palette_name]
    bg = palette[2]
    primary = palette[0]
    secondary = palette[1]
    dark = palette[3]
    image = paper_texture((size, size), bg).convert("RGBA")
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cell = max(54, size // density)
    rows = size // cell + 2
    cols = size // cell + 2

    if style == "Пиксельная вышивка":
        pixel = max(8, cell // 8)
        for y in range(rows):
            for x in range(cols):
                cx = x * cell + cell // 2
                cy = y * cell + cell // 2
                color = rgba(primary if (x + y + seed) % 2 == 0 else secondary)
                for step in range(-3, 4):
                    draw.rectangle((cx + step * pixel, cy + abs(step) * pixel, cx + (step + 1) * pixel - 2, cy + (abs(step) + 1) * pixel - 2), fill=color)
                    draw.rectangle((cx + step * pixel, cy - abs(step) * pixel, cx + (step + 1) * pixel - 2, cy - (abs(step) - 1) * pixel - 2), fill=color)
                if rng.random() > .58:
                    draw.rectangle((cx - pixel, cy - pixel, cx + pixel, cy + pixel), fill=rgba(dark))
    elif style == "Петроглифы":
        for y in range(rows):
            for x in range(cols):
                if rng.random() < .18:
                    continue
                kind = rng.choice(MOTIFS if accent == "Микс" else [accent])
                jitter_x = int(rng.integers(-cell // 7, cell // 7))
                jitter_y = int(rng.integers(-cell // 7, cell // 7))
                color = rgba(dark if rng.random() > .25 else primary)
                draw_motif(draw, kind, x * cell + cell // 2 + jitter_x, y * cell + cell // 2 + jitter_y, cell, color)
    else:
        for y in range(rows):
            for x in range(cols):
                cx = x * cell + cell // 2
                cy = y * cell + cell // 2
                color = rgba(primary if x % 2 == 0 else secondary)
                draw.polygon([(cx, cy - cell // 2), (cx + cell // 2, cy), (cx, cy + cell // 2), (cx - cell // 2, cy)], outline=color, width=max(4, cell // 12))
                draw.line((cx - cell // 2, cy, cx + cell // 2, cy), fill=rgba(dark, 160), width=max(3, cell // 18))
                draw.line((cx, cy - cell // 2, cx, cy + cell // 2), fill=rgba(dark, 160), width=max(3, cell // 18))

    if symmetry:
        mirror = layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        layer = Image.blend(layer, mirror, .35)

    image.alpha_composite(layer)
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))


def make_ribbon(pattern: Image.Image, palette_name: str) -> Image.Image:
    palette = PALETTES[palette_name]
    ribbon = Image.new("RGBA", (1180, 260), rgba(palette[2]))
    draw = ImageDraw.Draw(ribbon)
    crop = pattern.resize((260, 260), Image.LANCZOS)
    for x in range(0, 1180, 260):
        tile = crop if (x // 260) % 2 == 0 else crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        ribbon.alpha_composite(tile, (x, 0))
    draw.rectangle((0, 0, 1180, 12), fill=rgba(palette[1]))
    draw.rectangle((0, 248, 1180, 260), fill=rgba(palette[0]))
    return ribbon


def find_palette(image: Image.Image, count: int = 5) -> list[tuple[int, int, int]]:
    sample = image.convert("RGB").resize((160, 160))
    pixels = np.array(sample).reshape(-1, 3)
    model = KMeans(n_clusters=count, n_init=10, random_state=8)
    model.fit(pixels)
    centers = model.cluster_centers_.astype(int)
    labels, counts = np.unique(model.labels_, return_counts=True)
    order = labels[np.argsort(counts)[::-1]]
    return [tuple(centers[label]) for label in order]


def edge_preview(image: Image.Image) -> Image.Image:
    gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
    gray = ImageEnhance.Contrast(gray).enhance(2.2)
    result = Image.new("RGBA", image.size, (18, 35, 44, 0))
    result.putalpha(gray.point(lambda value: 230 if value > 28 else 0))
    return Image.alpha_composite(image.convert("RGBA"), result)


def showroom_scene() -> Image.Image:
    image = paper_texture(CANVAS, (234, 238, 234))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 520, CANVAS[0], CANVAS[1]), fill=(218, 213, 201, 255))
    draw.polygon([(150, 620), (1065, 620), (1205, 790), (20, 790)], fill=(197, 190, 175, 255))
    draw.rounded_rectangle((260, 110, 1040, 705), radius=34, fill=(242, 240, 232, 255), outline=(170, 168, 158, 255), width=4)
    draw.rounded_rectangle((315, 155, 985, 660), radius=20, fill=(250, 248, 239, 255), outline=(206, 201, 188, 255), width=4)
    draw.polygon(SURFACE_QUAD, fill=(255, 254, 246, 255), outline=(155, 154, 147, 255))
    draw.rounded_rectangle((65, 290, 310, 710), radius=30, fill=(235, 224, 203, 255), outline=(155, 130, 98, 255), width=5)
    draw.arc((112, 170, 262, 395), 180, 360, fill=(126, 96, 65, 255), width=12)
    draw.rounded_rectangle((1040, 330, 1205, 610), radius=26, fill=(249, 249, 244, 255), outline=(160, 170, 170, 255), width=5)
    draw.arc((1142, 395, 1265, 545), 270, 90, fill=(160, 170, 170, 255), width=14)
    return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=105, threshold=3))


def transform_to_quad(pattern: Image.Image, quad: tuple[tuple[int, int], ...], size: tuple[int, int]) -> Image.Image:
    texture = pattern.resize((640, 640), Image.LANCZOS)
    alpha = np.array(texture.getchannel("A"), dtype=np.float32)
    texture.putalpha(Image.fromarray((alpha * .92).clip(0, 255).astype(np.uint8)))
    source = [(0, 0), (texture.width, 0), (texture.width, texture.height), (0, texture.height)]
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(quad, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    coeffs = np.linalg.solve(np.array(matrix, dtype=float), np.array(vector, dtype=float))
    return texture.transform(size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def ar_overlay(base: Image.Image, pattern: Image.Image, quad: tuple[tuple[int, int], ...], tint: int) -> Image.Image:
    base = base.convert("RGBA")
    base.thumbnail(CANVAS, Image.LANCZOS)
    canvas = Image.new("RGBA", CANVAS, (235, 234, 226, 255))
    canvas.alpha_composite(base, ((CANVAS[0] - base.width) // 2, (CANVAS[1] - base.height) // 2))
    overlay = transform_to_quad(pattern, quad, CANVAS)
    if tint < 100:
        alpha = np.array(overlay.getchannel("A"), dtype=np.float32)
        overlay.putalpha(Image.fromarray((alpha * (tint / 100)).astype(np.uint8)))
    result = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(result)
    draw.line([*quad, quad[0]], fill=(255, 255, 255, 170), width=3)
    return result


def generate_collection(pattern: Image.Image, ribbon: Image.Image) -> Image.Image:
    image = showroom_scene()
    image = ar_overlay(image, pattern, SURFACE_QUAD, 88)
    bag_pattern = pattern.resize((175, 220), Image.LANCZOS)
    mug_pattern = ribbon.resize((150, 66), Image.LANCZOS)
    image.alpha_composite(bag_pattern, (98, 390))
    image.alpha_composite(mug_pattern, (1055, 437))
    return image


def open_user_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGBA")


with st.sidebar:
    st.subheader("Генератор")
    palette_name = st.selectbox("Палитра", list(PALETTES.keys()))
    style = st.radio("Стиль", ["Пиксельная вышивка", "Петроглифы", "Северная мозаика"])
    accent = st.selectbox("Главный мотив", ["Микс", *MOTIFS], disabled=style != "Петроглифы")
    seed = st.slider("Номер варианта", 1, 999, 124)
    density = st.slider("Плотность узора", 5, 14, 9)
    symmetry = st.toggle("Симметрия", value=True)
    opacity = st.slider("Яркость AR-наложения", 35, 100, 88)

    st.subheader("AR-сцена")
    scene_mode = st.radio("Фон", ["Демо-сцена", "Загрузить фото", "Камера"], horizontal=False)
    uploaded_scene = st.file_uploader("Фото поверхности", type=["png", "jpg", "jpeg", "webp"], disabled=scene_mode != "Загрузить фото")
    camera_scene = st.camera_input("Снимок с камеры", disabled=scene_mode != "Камера")

    st.subheader("Плоскость")
    top_shift = st.slider("Верхняя перспектива", -120, 120, 0)
    side_shift = st.slider("Боковая перспектива", -120, 120, 0)


st.markdown(
    """
    <div class="hero">
      <h1>OrnaMap AR</h1>
      <p class="lead">
        Генератор северных орнаментов и примерка узора на поверхности в стиле дополненной реальности.
        Проект показывает не просто готовую картинку, а понятный алгоритм: генерация мотива, анализ цветов,
        построение повторяющегося паттерна и перспективное наложение на сцену.
      </p>
      <span class="chip">Streamlit</span>
      <span class="chip">Pillow</span>
      <span class="chip">KMeans</span>
      <span class="chip">AR-перспектива</span>
      <span class="chip">PNG экспорт</span>
    </div>
    """,
    unsafe_allow_html=True,
)

pattern = generate_ornament(seed, palette_name, style, density, symmetry, accent)
ribbon = make_ribbon(pattern, palette_name)
palette = find_palette(pattern)

quad = (
    (SURFACE_QUAD[0][0] + side_shift, SURFACE_QUAD[0][1] + top_shift),
    (SURFACE_QUAD[1][0] - side_shift, SURFACE_QUAD[1][1] - top_shift),
    (SURFACE_QUAD[2][0] - side_shift, SURFACE_QUAD[2][1]),
    (SURFACE_QUAD[3][0] + side_shift, SURFACE_QUAD[3][1]),
)

if scene_mode == "Загрузить фото":
    source_scene = open_user_image(uploaded_scene) or showroom_scene()
elif scene_mode == "Камера":
    source_scene = open_user_image(camera_scene) or showroom_scene()
else:
    source_scene = showroom_scene()

ar_result = ar_overlay(source_scene, pattern, quad, opacity)
collection = generate_collection(pattern, ribbon)

main_left, main_right = st.columns([.9, 1.35])
with main_left:
    st.subheader("Сгенерированный орнамент")
    st.image(pattern, caption="Вариант узора", width="stretch")
    st.image(ribbon, caption="Полосная версия для бордюра или ткани", width="stretch")
    swatches = "".join(
        f"<span style='display:inline-block;width:56px;height:36px;border-radius:12px;margin-right:8px;"
        f"background:rgb{color};border:1px solid rgba(23,43,53,.14)'></span>"
        for color in palette
    )
    st.markdown(swatches, unsafe_allow_html=True)

with main_right:
    st.subheader("AR-примерка")
    st.image(ar_result, caption="Перспективное наложение на плоскость", width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Скачать AR PNG", data=png_bytes(ar_result), file_name="ornamap_ar.png", mime="image/png")
    with c2:
        st.download_button("Скачать орнамент PNG", data=png_bytes(pattern), file_name="ornamap_pattern.png", mime="image/png")

st.divider()

tab_gallery, tab_vision, tab_research = st.tabs(["Витрина", "Компьютерное зрение", "Для защиты"])

with tab_gallery:
    st.image(collection, caption="Один орнамент сразу на нескольких носителях", width="stretch")

with tab_vision:
    v1, v2 = st.columns(2)
    with v1:
        st.image(edge_preview(pattern), caption="Выделение контуров", width="stretch")
    with v2:
        st.markdown(
            """
            <div class="metric"><b>Что делает программа</b><br>
            <span class="small">1. Создает узор из простых геометрических мотивов.<br>
            2. Повторяет его по сетке и добавляет симметрию.<br>
            3. Находит основные цвета алгоритмом KMeans.<br>
            4. Преобразует квадратный паттерн в четырехугольник, как при AR-наложении.</span></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="metric"><b>Почему это выглядит как дополненная реальность</b><br>
            <span class="small">Узор не просто вставляется прямоугольником. Он искажается по четырем точкам,
            поэтому повторяет перспективу листа, стены, стола или другой поверхности на фото.</span></div>
            """,
            unsafe_allow_html=True,
        )

with tab_research:
    st.markdown(
        """
        **Короткая формулировка проекта:** я создала веб-приложение, которое генерирует северные орнаменты
        и показывает, как они могут выглядеть на реальной поверхности через AR-примерку.

        **Что можно показать жюри:** менять стиль, палитру, номер варианта, плотность узора, включать
        симметрию, делать снимок с камеры и накладывать орнамент на выбранную плоскость.

        **Где здесь информатика:** изображение хранится как массив пикселей, цвета анализируются методом
        KMeans, контуры выделяются фильтром, а перспектива строится программным преобразованием изображения.
        """
    )
