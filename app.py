from io import BytesIO
from math import cos, pi, sin
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

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
            radial-gradient(circle at 8% 8%, rgba(228, 31, 49, .13), transparent 24%),
            radial-gradient(circle at 94% 10%, rgba(29, 68, 154, .14), transparent 25%),
            linear-gradient(135deg, #f8f2e8 0%, #eef5f3 52%, #fffaf2 100%);
    }
    .block-container { max-width: 1340px; padding-top: 1.25rem; }
    h1 {
        color: #172b35;
        font-size: 4.1rem !important;
        line-height: .96 !important;
        letter-spacing: 0 !important;
        margin-bottom: .25rem !important;
    }
    h2, h3 { color: #172b35; letter-spacing: 0 !important; }
    [data-testid="stSidebar"] {
        background: #f4efe5;
        border-right: 1px solid rgba(23, 43, 53, .13);
    }
    div[data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 20px 54px rgba(24, 36, 42, .15);
    }
    .hero {
        padding: 20px 0 14px 0;
        border-bottom: 1px solid rgba(23, 43, 53, .12);
        margin-bottom: 22px;
    }
    .lead {
        color: #52646c;
        max-width: 900px;
        font-size: 1.05rem;
        line-height: 1.52;
    }
    .chip {
        display: inline-block;
        margin: 0 7px 7px 0;
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(255,255,255,.68);
        border: 1px solid rgba(23,43,53,.12);
        color: #344851;
        font-size: .92rem;
    }
    .metric {
        background: rgba(255,255,255,.68);
        border: 1px solid rgba(23,43,53,.11);
        border-radius: 16px;
        padding: 14px 16px;
        min-height: 84px;
    }
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


def rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return color[0], color[1], color[2], alpha


def make_reference_strip(width: int = 1120, height: int = 520) -> Image.Image:
    image = Image.new("RGBA", (width, height), (*PAPER, 255))
    draw = ImageDraw.Draw(image)
    s = 8

    def pixel_cells(x: int, y: int, cells: list[tuple[int, int]], color) -> None:
        for cx, cy in cells:
            draw.rectangle((x + cx * s, y + cy * s, x + (cx + 1) * s - 1, y + (cy + 1) * s - 1), fill=rgba(color))

    def band(y: int, step: int, cells: list[tuple[int, int]], color) -> None:
        for x in range(-step, width + step, step):
            pixel_cells(x, y, cells, color)

    band(22, 92, BLUE, [])
    return image
