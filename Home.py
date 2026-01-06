# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 13:07:51 2026

@author: Brayden Miao
"""



# Home page for Linear Transformation Playground

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Linear Transformation Playground",
    layout="wide"
)

st.title("Linear Transformation Playground")

st.write(
    """
    Created by Brayden Miao
    
    Janunary, 2026
    
    Choose a visualization mode:

    - **2D Transform**: random points in the plane, 2×2 matrix, eigenvectors, SVD cartoon.
    - **3D Transform**: random points + cube in 3D, 3×3 matrix, eigenvectors, 3D SVD path + GIF/MP4.
    - **2×3 Projection**: 2×3 matrix mapping 3D points onto a tilted 2D plane using SVD, with cartoon + GIF.
    - **3×2 Lifting**: 3×2 matrix mapping 2D points into 3D space using SVD, with cartoon + GIF.
    """
)

# ----------------------------
# Caching helper (key change)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_video_b64(path_str: str, file_mtime: float) -> str:
    """
    Read a local MP4 and return base64 string.
    Cached across Streamlit reruns. Cache invalidates automatically when the file changes
    because we include file_mtime in the cache key.
    """
    with open(path_str, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def autoplay_video_panel(video_path: Path, title: str, height: int = 400) -> None:
    """
    Render an autoplay/muted/loop video using base64 data URL.
    Uses cached base64 so reruns don't re-encode.
    """
    if video_path.exists():
        st.markdown(title)
        b64 = load_video_b64(str(video_path), video_path.stat().st_mtime)

        video_html = f"""
        <video width="100%" autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        components.html(video_html, height=height)
    else:
        st.info(f"{video_path.name} not found in the app folder.")


# First row: 2D & 3D
col1, col2 = st.columns(2)

with col1:
    st.subheader("2D Transform")
    st.write(
        """
        Explore how a 2×2 matrix stretches, shears, and rotates a cloud of points
        in the plane, and how its eigenvectors and SVD explain the motion.
        """
    )
    if st.button("Go to 2D Transform"):
        st.switch_page("pages/1_2D_Transform.py")

    autoplay_video_panel(
        Path(__file__).parent / "SVD2x2Demo.mp4",
        "##### 2D SVD demo (auto-playing)",
        height=400
    )

with col2:
    st.subheader("3D Transform")
    st.write(
        """
        See a 3×3 matrix transform a 3D point cloud and cube, with eigenvectors in 3D
        and a rotation–stretch–rotation SVD path (plus an animation).
        """
    )
    if st.button("Go to 3D Transform"):
        st.switch_page("pages/2_3D_Transform.py")

    autoplay_video_panel(
        Path(__file__).parent / "SVD3x3Demo.mp4",
        "##### 3D SVD demo (auto-playing)",
        height=400
    )

st.markdown("---")

# Second row: 2×3 Projection & 3×2 Lifting
col3, col4 = st.columns(2)

with col3:
    st.subheader("2×3 Projection (R³ → R²)")
    st.write(
        """
        Visualize how a 2×3 matrix projects a 3D point cloud onto a tilted 2D plane.
        The app shows the SVD-based interpretation (rotation by V, stretching by Σ,
        in-plane rotation by U) with a 3-stage cartoon and GIF animation.
        """
    )
    if st.button("Go to 2×3 Projection"):
        st.switch_page("pages/3_2x3_Projection.py")

    autoplay_video_panel(
        Path(__file__).parent / "SVD2x3ProjectionDemo.mp4",
        "##### 2x3 Matrix SVD demo (auto-playing)",
        height=400
    )

with col4:
    st.subheader("3×2 Lifting (R² → R³)")
    st.write(
        """
        See how a 3×2 matrix lifts 2D points into 3D space. The SVD cartoon shows
        in-plane rotation by V, stretching by Σ, and a 3D rotation by U. A red–blue
        gradient square tracks how an entire region moves, with an optional GIF animation.
        """
    )
    if st.button("Go to 3×2 Lifting"):
        st.switch_page("pages/4_3x2_Lifting.py")

    autoplay_video_panel(
        Path(__file__).parent / "SVD3x2LiftingDemo.mp4",
        "##### 3x2 Matrix SVD demo (auto-playing)",
        height=400
    )
