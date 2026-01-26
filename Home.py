# -*- coding: utf-8 -*-
"""
Home2 page for Linear Transformation Playground
(uses the same autoplay GIF logic as Home.py)
"""

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Linear Transformation Playground", layout="wide")
st.title("Linear Transformation Playground (Extended)")

st.write(
    """
    Created by BM Studio  

    January, 2026  

    Choose a visualization mode:
    """
)

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"


# ----------------------------
# Caching helper (GIF)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_gif_b64(path_str: str, file_mtime: float) -> str:
    with open(path_str, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def autoplay_gif_panel(gif_path: Path, title: str, height: int = 400) -> None:
    """
    IDENTICAL logic to Home.py
    """
    if gif_path.exists():
        st.markdown(title)
        b64 = load_gif_b64(str(gif_path), gif_path.stat().st_mtime)
        gif_html = (
            f'<img src="data:image/gif;base64,{b64}" '
            f'style="width: 100%; max-width: 520px; height: auto; margin: 0 auto; display: block;" />'
        )
        components.html(gif_html, height=height)
    else:
        st.info(f"{gif_path.as_posix()} not found.")


# ============================================================
# Row 1: 2D & 3D Transforms
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("2D Transform")
    st.write("2×2 matrix, eigenvectors, SVD geometry.")
    if st.button("Go to 2D Transform", key="h2_2d"):
        st.switch_page("pages/1_2D_Transform.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD2x2Demo.gif",
        "##### 2D SVD demo (auto-playing)",
        height=400,
    )

with col2:
    st.subheader("3D Transform")
    st.write("3×3 matrix, eigenvectors, 3D SVD path.")
    if st.button("Go to 3D Transform", key="h2_3d"):
        st.switch_page("pages/2_3D_Transform.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD3x3Demo.gif",
        "##### 3D SVD demo (auto-playing)",
        height=400,
    )

st.markdown("---")

# ============================================================
# Row 2: 2×3 Projection & 3×2 Lifting
# ============================================================
col3, col4 = st.columns(2)

with col3:
    st.subheader("2×3 Projection (R³ → R²)")
    st.write("Project 3D points onto a 2D plane using SVD.")
    if st.button("Go to 2×3 Projection", key="h2_2x3"):
        st.switch_page("pages/3_2x3_Projection.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD2x3ProjectionDemo.gif",
        "##### 2×3 SVD demo (auto-playing)",
        height=400,
    )

with col4:
    st.subheader("3×2 Lifting (R² → R³)")
    st.write("Lift 2D points into 3D space using SVD.")
    if st.button("Go to 3×2 Lifting", key="h2_3x2"):
        st.switch_page("pages/4_3x2_Lifting.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD3x2LiftingDemo.gif",
        "##### 3×2 SVD demo (auto-playing)",
        height=400,
    )

st.markdown("---")

# ============================================================
# Row 3: PCA Demo & SVD Image Compression
# ============================================================
col5, col6 = st.columns(2)

with col5:
    st.subheader("PCA Demo")
    st.write(
        """
        Understand PCA as a rotation plus projection.
        Includes 2D and 3D intuition and links to SVD.
        """
    )
    if st.button("Go to PCA Demo", key="h2_pca"):
        st.switch_page("pages/5_PCA_Demo.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "PCACartoon2D.gif",
        "##### PCA demo (auto-playing)",
        height=400,
    )

with col6:
    st.subheader("SVD Image Compression")
    st.write(
        """
        Compress grayscale images using SVD.
        Keep top-k singular values and inspect reconstruction.
        """
    )
    if st.button("Go to SVD Image Compression", key="h2_svd_img"):
        st.switch_page("pages/6_SVDImgCompression.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "SVD_Img.gif",
        "##### SVD Image Reconstrction (auto-playing)",
        height=400,
    )

st.markdown("---")

# ============================================================
# Row 4: PCA Image Compression
# ============================================================
col7, col8 = st.columns(2)

with col7:
    st.subheader("PCA Image Compression")
    st.write(
        """
        PCA-based image compression:
        mean image, eigen images, reconstruction, and error.
        """
    )
    if st.button("Go to PCA Image Compression", key="h2_pca_img"):
        st.switch_page("pages/7_PCAImgCompression.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "PCAFacesDemo.gif",
        "##### PCA Faces Reconstrction (auto-playing)",
        height=400,
    )

with col8:
    st.empty()
