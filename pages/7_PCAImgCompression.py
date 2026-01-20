# pages/6_Eigenfaces_PCA.py
# PCA Demo with two modes:
#
# Mode A) Built-in sample (faces): classic eigenfaces PCA on a face dataset (grayscale).
#   - Built-in sample path DISPLAY is relative: \assets\faces
#   - PCA uses ORIGINAL dataset pixels (no resizing before PCA)
#   - Display scale is display-only and keeps ratio (default 2x)
#
# Mode B) User image (RGB): PCA on pixels in RGB space (pixels are samples, RGB are features).
#   - Dataset source is either Built-in sample OR user image (no separate upload section)
#   - Show original + RGB channels in a 2x2 panel grid
#   - Show eigen images (PC score maps) in a 2x2 grid too (PC1, PC2, PC3, EVR plot)
#   - Display resize slider WORKS:
#       slider controls desired scale; we prevent cropping by capping render width to PANEL_MAX_W
#   - Reconstruct color image using k = 1,2,3 components and show difference image
#
# Dependencies: streamlit, numpy, plotly, pillow

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


# -----------------------------
# Repo root finder (portable)
# -----------------------------
def find_repo_root() -> Path:
    """
    Try to locate repo root robustly:
    1) If current working directory contains assets/faces, use it.
    2) Else walk upward from this file until we find assets/faces.
    3) Else fall back to a known absolute path.
    """
    cwd = Path.cwd()
    if (cwd / "assets" / "faces").is_dir():
        return cwd

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "assets" / "faces").is_dir():
            return parent

    return Path(r"D:\Visualizer2026")


REPO_ROOT = find_repo_root()
DEFAULT_FACES_DIR = str((REPO_ROOT / "assets" / "faces").resolve())
BUILTIN_FACES_DISPLAY = r"\assets\faces"


# -----------------------------
# PCA helpers
# -----------------------------
@dataclass
class PCAResult:
    mean: np.ndarray          # (p,)
    U: np.ndarray             # (n, r)
    s: np.ndarray             # (r,)
    Vt: np.ndarray            # (r, p)
    Xc: np.ndarray            # (n, p)
    scores: np.ndarray        # (n, r)


def pca_via_svd(X: np.ndarray) -> PCAResult:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt.T
    return PCAResult(mean=mean, U=U, s=s, Vt=Vt, Xc=Xc, scores=scores)


def explained_variance_ratio(p: PCAResult) -> np.ndarray:
    n = p.Xc.shape[0]
    if n <= 1:
        return np.zeros_like(p.s)
    vals = (p.s ** 2) / (n - 1)
    total = vals.sum() if vals.sum() > 0 else 1.0
    return vals / total


def reconstruct_one_vector(p: PCAResult, x: np.ndarray, k: int) -> np.ndarray:
    k = int(np.clip(k, 0, p.s.shape[0]))
    xc = x - p.mean
    if k == 0:
        return p.mean.copy()
    V = p.Vt.T
    z = xc @ V[:, :k]
    xhat = z @ V[:, :k].T + p.mean
    return xhat


def reconstruct_matrix(p: PCAResult, X: np.ndarray, k: int) -> np.ndarray:
    k = int(np.clip(k, 0, p.s.shape[0]))
    Xc = X - p.mean
    if k == 0:
        return np.tile(p.mean, (X.shape[0], 1))
    V = p.Vt.T
    Z = Xc @ V[:, :k]
    Xhat = Z @ V[:, :k].T + p.mean
    return Xhat


# -----------------------------
# Face dataset IO (no resize)
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def list_images_recursive(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def load_gray_vec_no_resize(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(path).convert("L")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten(), (w, h)


def vec_to_gray_img(vec: np.ndarray, size_wh: Tuple[int, int]) -> Image.Image:
    w, h = size_wh
    arr = vec.reshape((h, w))
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="L")


def normalize_for_display(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vmin = float(np.min(vec))
    vmax = float(np.max(vec))
    if vmax - vmin < eps:
        return np.zeros_like(vec)
    return (vec - vmin) / (vmax - vmin)


# -----------------------------
# Plots
# -----------------------------
def fig_evr(evr: np.ndarray, cumulative: bool = True, height: int = 320) -> go.Figure:
    xs = np.arange(1, len(evr) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=evr, name="EVR"))
    if cumulative:
        fig.add_trace(go.Scatter(x=xs, y=np.cumsum(evr), mode="lines+markers", name="Cumulative"))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="component",
        yaxis_title="explained variance",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PCA Demo", layout="wide")
st.title("PCA Demo")
st.caption("Built-in sample: eigenfaces. User image: RGB PCA on pixels (PC score maps, color reconstruction).")

with st.sidebar:
    st.subheader("Dataset source")
    mode = st.radio("Choose", ["Built-in sample (faces)", "User image (RGB)"], index=0)
    st.caption(f"Repo root detected: {REPO_ROOT}")

    st.divider()
    st.subheader("Display resize (keeps ratio)")
    # For faces: default 2x; for user image: default 1.0 (panel cap handles large images)
    display_scale = st.slider(
        "Scale",
        min_value=0.1,
        max_value=6.0,
        value=2.0 if mode == "Built-in sample (faces)" else 1.0,
        step=0.1,
    )

    if mode == "Built-in sample (faces)":
        st.divider()
        st.subheader("Built-in sample path")
        st.text_input("Built-in path", value=BUILTIN_FACES_DISPLAY, disabled=True)

        st.divider()
        st.subheader("Loading")
        max_imgs = st.slider("Max images to load", 20, 200, 100, 10)
        seed = st.number_input("Shuffle seed", value=7, step=1)
        shuffle = st.checkbox("Shuffle images", value=True)

        st.divider()
        st.subheader("Eigenfaces view")
        eig_cols = st.slider("Eigenfaces columns", 2, 8, 4, 1)
        show_eigs = st.slider("How many eigenfaces to show", 4, 64, 16, 4)

        st.divider()
        st.subheader("Reconstruction")
        k = st.slider("Keep k components", 0, 200, 25, 1)

    else:
        st.divider()
        st.subheader("User image input")
        uploaded = st.file_uploader("Upload a color image (png/jpg/webp)", type=["png", "jpg", "jpeg", "webp"])

        st.divider()
        st.subheader("RGB PCA reconstruction")
        k_rgb = st.slider("Keep k components (RGB)", 1, 3, 2, 1)
        show_evr_rgb = st.checkbox("Show explained variance", value=True)

        st.divider()
        st.subheader("Auto-fit panels")
        PANEL_MAX_W = st.slider("Max panel width (px)", 220, 520, 360, 10)
        st.caption("No cropping: image render width is capped by this value in 2×2 grids.")


# -----------------------------
# Mode A: Built-in eigenfaces
# -----------------------------
if mode == "Built-in sample (faces)":
    root = DEFAULT_FACES_DIR
    if not root or not os.path.isdir(root):
        st.warning(f"Built-in dataset folder not found: {root}")
        st.stop()

    paths_all = list_images_recursive(root)
    if len(paths_all) == 0:
        st.warning(f"No images found under: {root}")
        st.stop()

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(paths_all))
    if shuffle:
        rng.shuffle(idx)
    idx = idx[: int(max_imgs)]
    paths = [paths_all[i] for i in idx]

    @st.cache_data(show_spinner=False)
    def load_pack_no_resize(paths_in: List[str], root_dir: str) -> Tuple[np.ndarray, List[str], Tuple[int, int]]:
        X_list: List[np.ndarray] = []
        labels: List[str] = []
        base_size: Optional[Tuple[int, int]] = None

        for pth in paths_in:
            vec, sz = load_gray_vec_no_resize(pth)
            if base_size is None:
                base_size = sz
            if sz != base_size:
                raise ValueError(
                    f"Image size mismatch: {pth} is {sz}, expected {base_size}. "
                    "PCA requires all images have the same resolution."
                )
            X_list.append(vec)
            labels.append(os.path.relpath(pth, start=root_dir))

        if base_size is None or len(X_list) == 0:
            raise ValueError("No readable images loaded.")
        X = np.vstack(X_list).astype(np.float32)
        return X, labels, base_size

    try:
        X, labels, size_wh = load_pack_no_resize(paths, root)
    except Exception as e:
        st.error(str(e))
        st.stop()

    n, _p = X.shape
    pca = pca_via_svd(X)
    evr = explained_variance_ratio(pca)

    base_w, base_h = size_wh
    display_w = max(32, int(round(base_w * float(display_scale))))
    st.sidebar.caption(f"Dataset size: {base_w}×{base_h}")
    st.sidebar.caption(f"Display width: {display_w}px")

    left, right = st.columns([1.0, 1.8], gap="large")

    with left:
        st.subheader("Mean face")
        st.image(vec_to_gray_img(pca.mean, size_wh), caption=f"Mean face (n={n})", width=display_w)
        st.subheader("Explained variance (first 50)")
        st.plotly_chart(fig_evr(evr[: min(len(evr), 50)]), use_container_width=True)

    with right:
        st.subheader("Eigenfaces")
        num_show = int(min(show_eigs, pca.Vt.shape[0]))
        rows = int(math.ceil(num_show / int(eig_cols)))

        k0 = 0
        for _ in range(rows):
            cols = st.columns(int(eig_cols))
            for c in cols:
                if k0 >= num_show:
                    break
                v = pca.Vt[k0, :]
                v_disp = normalize_for_display(v)
                c.image(
                    vec_to_gray_img(v_disp, size_wh),
                    caption=f"PC{k0+1}  EVR {evr[k0]:.3f}",
                    width=display_w
                )
                k0 += 1

    st.divider()
    st.subheader("Face reconstruction from k eigenfaces")

    sel = st.slider("Select dataset face index", 0, n - 1, 0, 1)
    x = X[sel, :]
    xhat = reconstruct_one_vector(pca, x, k=int(k))

    diff = np.abs(x - xhat)
    diff_disp = normalize_for_display(diff)

    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    c1.image(vec_to_gray_img(x, size_wh), caption=f"Original face (index {sel})", use_container_width=True)
    c2.image(vec_to_gray_img(xhat, size_wh), caption=f"Reconstructed face (k={int(k)})", use_container_width=True)
    c3.image(vec_to_gray_img(diff_disp, size_wh), caption="Difference (absolute error, rescaled)", use_container_width=True)

    mse_val = float(np.mean((x - xhat) ** 2))
    st.write(f"**Reconstruction MSE:** {mse_val:.6f}")
    st.stop()


# -----------------------------
# Mode B: User image (RGB) PCA on pixels
# -----------------------------
if uploaded is None:
    st.info("Upload a color image in the sidebar to run RGB PCA on pixels.")
    st.stop()

try:
    img_rgb = Image.open(uploaded).convert("RGB")
except Exception as e:
    st.error(f"Failed to open image: {e}")
    st.stop()

arr = np.asarray(img_rgb, dtype=np.uint8)  # H x W x 3
H, W, _ = arr.shape

# Split channels
r_ch = arr[:, :, 0]
g_ch = arr[:, :, 1]
b_ch = arr[:, :, 2]

# Display sizing (NO CROPPING) + slider works:
# - slider controls desired_w
# - we cap actual rendered width to PANEL_MAX_W so it fits in 2×2 panels
panel_w = int(PANEL_MAX_W)
desired_w = int(round(W * float(display_scale)))
display_w_user = max(64, min(desired_w, panel_w))
effective_scale = display_w_user / float(W)

st.sidebar.caption(f"Image size: {W}×{H}")
st.sidebar.caption(f"Panel cap: {panel_w}px")
st.sidebar.caption(f"Desired scale: {float(display_scale):.2f}  Effective scale: {effective_scale:.3f}")
st.sidebar.caption(f"Panel display width: {display_w_user}px")

st.subheader("User image and RGB channels")
row1 = st.columns(2, gap="large")
row2 = st.columns(2, gap="large")
row1[0].image(img_rgb, caption="Original color image", width=display_w_user)
row1[1].image(Image.fromarray(r_ch, mode="L"), caption="R channel (grayscale)", width=display_w_user)
row2[0].image(Image.fromarray(g_ch, mode="L"), caption="G channel (grayscale)", width=display_w_user)
row2[1].image(Image.fromarray(b_ch, mode="L"), caption="B channel (grayscale)", width=display_w_user)

# PCA on pixels: X shape (N,3), N = H*W
Xpix = (arr.reshape(-1, 3).astype(np.float32)) / 255.0
pca_rgb = pca_via_svd(Xpix)
evr_rgb = explained_variance_ratio(pca_rgb)

scores = pca_rgb.scores  # (N,3)
pc1 = normalize_for_display(scores[:, 0]).reshape(H, W)
pc2 = normalize_for_display(scores[:, 1]).reshape(H, W)
pc3 = normalize_for_display(scores[:, 2]).reshape(H, W)

st.divider()
st.subheader("Eigen images (PC score maps)")
erow1 = st.columns(2, gap="large")
erow2 = st.columns(2, gap="large")

erow1[0].image(
    Image.fromarray((pc1 * 255).astype(np.uint8), mode="L"),
    caption=f"PC1 score map (EVR {evr_rgb[0]:.3f})",
    width=display_w_user
)
erow1[1].image(
    Image.fromarray((pc2 * 255).astype(np.uint8), mode="L"),
    caption=f"PC2 score map (EVR {evr_rgb[1]:.3f})",
    width=display_w_user
)
erow2[0].image(
    Image.fromarray((pc3 * 255).astype(np.uint8), mode="L"),
    caption=f"PC3 score map (EVR {evr_rgb[2]:.3f})",
    width=display_w_user
)

if show_evr_rgb:
    with erow2[1]:
        st.plotly_chart(fig_evr(evr_rgb, height=260), use_container_width=True)
else:
    erow2[1].empty()

# Reconstruct using k components (1..3), then reshape to image
Xhat = reconstruct_matrix(pca_rgb, Xpix, k=int(k_rgb))
arr_hat = np.clip(Xhat.reshape(H, W, 3) * 255.0, 0, 255).astype(np.uint8)

# Difference image (magnitude across channels)
diff = (arr.astype(np.int16) - arr_hat.astype(np.int16))
diff_mag = np.sqrt(np.sum(diff.astype(np.float32) ** 2, axis=2))  # HxW
diff_disp = normalize_for_display(diff_mag.flatten()).reshape(H, W)
diff_u8 = (diff_disp * 255).astype(np.uint8)

st.divider()
st.subheader(f"Reconstructed color image using first {int(k_rgb)} PCA component(s)")
rrow1 = st.columns(2, gap="large")
rrow2 = st.columns(2, gap="large")

rrow1[0].image(img_rgb, caption="Original color image", width=display_w_user)
rrow1[1].image(Image.fromarray(arr_hat, mode="RGB"), caption=f"Reconstructed (k={int(k_rgb)})", width=display_w_user)
rrow2[0].image(Image.fromarray(diff_u8, mode="L"), caption="Difference (RGB error magnitude, rescaled)", width=display_w_user)
rrow2[1].empty()

mse_rgb = float(np.mean((Xpix - Xhat) ** 2))
st.write(f"**Pixel-space reconstruction MSE (RGB):** {mse_rgb:.6f}")

with st.expander("What’s happening here?"):
    st.markdown(
        """
- PCA is applied to **pixel RGB vectors**.
- Each pixel is a sample with 3 features (R,G,B): **X is (H·W)×3**.
- The eigen images shown are **PC score maps** (projection values per pixel), reshaped back to H×W.
- Reconstruction with k=1..3 gives a low-dimensional color approximation plus a difference map.
        """.strip()
    )
