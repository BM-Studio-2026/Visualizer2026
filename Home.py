# -*- coding: utf-8 -*-
"""
Home2 page for Linear Transformation Playground
(uses the same autoplay GIF logic as Home.py)
"""

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

LOGO_PATH = Path("videos/BM_Logo.png")  
# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Linear Transformation Playground", layout="wide")
st.title("Seeing Linear Algebra: An Interactive Journey from Data to Geometry")

st.markdown(
    """
    <div style="font-size:32px; font-weight:650; margin-top:-10px; margin-left: 200px">
              -- Interactive Linear Algebra Visualizer
    </div>
    """,
    unsafe_allow_html=True,
)


col_text, col_logo = st.columns([6, 1])  # adjust ratio if needed

with col_text:
    st.markdown(
        """
        <div style="margin-top:20px; margin-left:250px; font-size:22px; line-height:1.6;">
            <div style="font-weight:600;">Created by BM Studio</div>
            <div>January, 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_logo:
    st.markdown("<div style='padding-left:200px;'>", unsafe_allow_html=True)
    st.image(LOGO_PATH, width=120)
    st.markdown("</div>", unsafe_allow_html=True)
    
    
st.markdown("""
### About This Tool

This interactive linear algebra visualizer demonstrates:
- 2D and 3D matrix transformations
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Projection and lifting between dimensions
- Image compression using SVD and PCA
- Least Square Estimation (LSE)

Designed for students learning linear algebra, machine learning, and data science.
""")
    
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
# Row 4: PCA Image Compression & Least Squares (LSE)
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
    st.subheader("Least Squares (LSE)")
    st.write(
        """
        3D least squares demo:
        planes (equations), intersections, LS point, and residuals.
        """
    )
    if st.button("Go to Least Squares (LSE)", key="h2_lse"):
        st.switch_page("pages/8_LSE.py")
    autoplay_gif_panel(
        VIDEOS_DIR / "LSE3D_Demo.gif",
        "##### LSE 3D Demo (auto-playing)",
        height=400,
    )
    