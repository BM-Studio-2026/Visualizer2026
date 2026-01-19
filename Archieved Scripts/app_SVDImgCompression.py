import io
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="SVD Image Compression (Grayscale)", layout="wide")

st.title("SVD Image Compression (Single Band Image)")
st.write(
    "Upload an image (color or grayscale). If it is color, it will be converted to grayscale. "
    "Then we compute SVD and reconstruct using the top k singular values."
)

# -----------------------------
# Helpers
# -----------------------------
def pil_to_float_gray(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to grayscale float32 array in [0, 255]."""
    gray = pil_img.convert("L")
    return np.asarray(gray).astype(np.float32)

def safe_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip to [0, 255] and convert to uint8 for display/export."""
    x = np.clip(arr, 0, 255)
    return x.astype(np.uint8)

def make_sample_image(size: int = 256) -> np.ndarray:
    """Synthetic grayscale image (gradient + shapes) for demo."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    grad = (x / (size - 1)) * 180.0 + (y / (size - 1)) * 60.0
    img = grad.copy()
    img[40:120, 30:190] += 40.0
    img[140:220, 80:140] -= 60.0
    cx, cy = size * 0.72, size * 0.35
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    img += 50.0 * np.exp(-r2 / (2 * (size * 0.08) ** 2))
    return np.clip(img, 0, 255)

def resize_pil_keep_aspect(pil_img: Image.Image, max_side: int) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

@st.cache_data(show_spinner=False)
def compute_svd(A: np.ndarray):
    # For images this can be heavy, so keep full_matrices=False
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt

def reconstruct(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    # Efficient truncated SVD reconstruction
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    return (Uk * sk) @ Vtk

def compression_stats(m: int, n: int, k: int):
    """
    Original parameters: m*n
    Truncated SVD parameters: k*(m + n + 1)  (U: m*k, V: n*k, s: k)
    """
    original = m * n
    approx = k * (m + n + 1)
    ratio = approx / original if original > 0 else 1.0
    return ratio, original, approx

def energy_kept(s: np.ndarray, k: int) -> float:
    denom = float(np.sum(s ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(s[:k] ** 2) / denom)

def diff_image(A: np.ndarray, Ak: np.ndarray) -> np.ndarray:
    """Return a displayable absolute-difference image in uint8."""
    d = np.abs(A - Ak)
    dmax = float(d.max())
    if dmax <= 1e-12:
        return np.zeros_like(safe_uint8(A))
    d_norm = 255.0 * (d / dmax)
    return safe_uint8(d_norm)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Input")
use_sample = st.sidebar.checkbox("Use sample image", value=True)
max_side = st.sidebar.slider("Resize max side (for speed)", 64, 2048, 512, 32)

uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image (PNG/JPG/TIF/BMP). Color will be converted to grayscale.",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

# k slider will be defined after we know matrix size

# -----------------------------
# Load image
# -----------------------------
src_label = ""
uploaded_pil_resized = None
uploaded_is_color = False

if use_sample:
    A = make_sample_image(size=max_side)
    src_label = f"Sample grayscale image ({A.shape[1]}×{A.shape[0]})"
else:
    if uploaded_file is None:
        st.info("Upload an image, or turn on 'Use sample image'.")
        st.stop()

    uploaded_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(uploaded_bytes))

    # Detect if original is color-ish (RGB/RGBA/etc.)
    uploaded_is_color = pil_img.mode not in ["L", "I;16", "I", "F"]
    uploaded_pil_resized = resize_pil_keep_aspect(pil_img, max_side)

    A = pil_to_float_gray(uploaded_pil_resized)
    src_label = f"Uploaded image resized to {A.shape[1]}×{A.shape[0]} (grayscale matrix used for SVD)"

m, n = A.shape
k_max = min(m, n)
default_k = min(30, k_max)
k = st.slider("k (number of singular values kept)", 1, k_max, default_k)

# -----------------------------
# Compute SVD and reconstruct
# -----------------------------
with st.spinner("Computing SVD and reconstruction..."):
    U, s, Vt = compute_svd(A)
    Ak = reconstruct(U, s, Vt, k)

A_u8 = safe_uint8(A)
Ak_u8 = safe_uint8(Ak)
D_u8 = diff_image(A, Ak)

ratio, original_params, approx_params = compression_stats(m, n, k)
if ratio > 1.0:
    st.warning(
        f"Not really compression at this k. "
        f"Truncated SVD needs ~{ratio*100:.1f}% as many numbers as the original "
        f"(U_k + Σ_k + V_k is larger than m×n). Try a smaller k."
    )
ek = energy_kept(s, k)


# -----------------------------
# Display
# -----------------------------
if (not use_sample) and uploaded_is_color:
    # 2x2 layout for user color image
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    # Top-left: color image
    with r1c1:
        st.subheader("Original (Color)")
        st.image(uploaded_pil_resized, caption="Uploaded image (resized preview)", clamp=True)

    # Top-right: intentionally empty
    with r1c2:
        st.empty()

    # Bottom-left: grayscale used for SVD
    with r2c1:
        st.subheader("Grayscale used for SVD")
        st.caption(src_label)
        st.image(A_u8, caption="Matrix A", clamp=True)

    # Bottom-right: compressed result
    with r2c2:
        st.subheader(f"Compressed (k = {k})")
        st.caption(f"Energy kept (Σσ²): {ek*100:.2f}%   |   Parameter ratio: {ratio*100:.2f}%")
        st.image(Ak_u8, caption="Reconstruction A_k", clamp=True)

else:
    # Original 2-column layout (sample image or grayscale upload)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grayscale used for SVD")
        st.caption(src_label)
        st.image(A_u8, caption="Matrix A", clamp=True)

    with col2:
        st.subheader(f"Compressed (k = {k})")
        st.caption(f"Energy kept (Σσ²): {ek*100:.2f}%   |   Parameter ratio: {ratio*100:.2f}%")
        st.image(Ak_u8, caption="Reconstruction A_k", clamp=True)



# Difference directly under the two panels
st.subheader("Absolute difference |A − A_k| (scaled for visibility)")
st.image(D_u8, clamp=True)

# Metrics
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matrix size", f"{m} × {n}")
c2.metric("Original params (m·n)", f"{original_params:,}")
c3.metric("Truncated SVD params", f"{approx_params:,}")
c4.metric("Param ratio", f"{ratio*100:.2f}%")

# Singular values plot
st.subheader("Singular values (sorted)")
st.line_chart(s)

# Download compressed grayscale image
st.subheader("Download")
out = Image.fromarray(Ak_u8, mode="L")
buf = io.BytesIO()
out.save(buf, format="PNG")
st.download_button(
    "Download compressed grayscale image (PNG)",
    data=buf.getvalue(),
    file_name=f"svd_compressed_k{k}.png",
    mime="image/png",
)
