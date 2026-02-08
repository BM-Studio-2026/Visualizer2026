import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------- Core Math & Utility Functions ----------

def hex_to_rgb(hex_str):
    """Converts #RRGGBB to (R, G, B) tuple."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def blend_colors(hex1, hex2):
    """Blends two hex colors for intersection lines."""
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    avg_rgb = tuple(int((c1 + c2) / 2) for c1, c2 in zip(rgb1, rgb2))
    return f'rgb({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})'

def solve_lse_least_squares(planes):
    """Calculates the LS point, individual distances, and RMS error."""
    A_mat, b_vec, normalized_normals = [], [], []
    for (A, B, C, D) in planes:
        n = np.array([A, B, C], dtype=float)
        norm_val = np.linalg.norm(n)
        if norm_val < 1e-10: continue
        # Normalize to ensure least squares minimizes geometric distance
        A_mat.append(n / norm_val)
        b_vec.append(-D / norm_val)
        normalized_normals.append(n / norm_val)
    
    if not A_mat: return None, None, None, None
    
    A_mat, b_vec = np.array(A_mat), np.array(b_vec)
    # Solve using standard least squares (Normal Equations)
    x_ls, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    
    # Residual distances d_i = |n_i . x + D_i|
    dists = [abs(np.dot(A_mat[i], x_ls) - b_vec[i]) for i in range(len(A_mat))]
    # RMS = sqrt(mean(d_i^2))
    rms_error = np.sqrt(np.mean(np.square(dists)))
    
    return x_ls, dists, rms_error, normalized_normals

def make_plane_mesh(A, B, C, D, extent=5, n=40):
    """Generates mesh grids for Plotly surfaces."""
    lin = np.linspace(-extent, extent, n)
    coeffs = np.array([abs(A), abs(B), abs(C)])
    k = np.argmax(coeffs)
    if k == 2:
        X, Y = np.meshgrid(lin, lin)
        Z = -(A*X + B*Y + D) / C
        return X, Y, Z
    elif k == 1:
        X, Z = np.meshgrid(lin, lin)
        Y = -(A*X + C*Z + D) / B
        return X, Y, Z
    else:
        Y, Z = np.meshgrid(lin, lin)
        X = -(B*Y + C*Z + D) / A
        return X, Y, Z

def get_intersection_line(p1, p2, extent):
    """Calculates the intersection line between two planes."""
    n1, n2 = np.array(p1[:3]), np.array(p2[:3])
    v = np.cross(n1, n2)
    if np.linalg.norm(v) < 1e-10: return None
    A_sys = np.vstack([n1, n2])
    b_sys = np.array([-p1[3], -p2[3]])
    p_line, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
    t = np.array([-extent * 2, extent * 2])
    return p_line + t[:, np.newaxis] * (v / np.linalg.norm(v))

# ---------- Streamlit Layout ----------

st.set_page_config(page_title="LSE Matrix Visualizer", layout="wide")
st.title("🛰️ 3D Linear Systems & Least Squares Visualizer")

# Sidebar setup
st.sidebar.header("1. Define Your Planes")
num_planes = st.sidebar.slider("Number of Equations", 2, 8, 4)

planes = []
defaults = [(1.0, 1.0, 1.0, -3.0), (1.0, -1.0, 0.0, 0.0), (0.0, 1.0, -1.0, 1.0), (1.0, 0.0, -2.0, 2.0)]

cols = st.sidebar.columns(4)
for idx, lbl in enumerate(["A", "B", "C", "D"]): cols[idx].markdown(f"**{lbl}**")

for i in range(num_planes):
    r_cols = st.sidebar.columns(4)
    dv = defaults[i] if i < len(defaults) else (1.0, 0.0, 0.0, 1.0)
    vals = [r_cols[j].number_input(f"{l}{i}", value=dv[j], key=f"{l}{i}", label_visibility="collapsed") for j, l in enumerate("abcd")]
    planes.append(tuple(vals))

st.sidebar.markdown("---")
show_intersections = st.sidebar.toggle("Show Pairwise Blended Intersections", value=True)
show_projections = st.sidebar.toggle("Show Residuals (Distances to LS)", value=True)
plot_extent = st.sidebar.slider("Plot Box Size", 1, 15, 5)

# Perform Computations
x_ls, dists, rms_error, normals = solve_lse_least_squares(planes)

# --- 3D Visualization ---
fig = go.Figure()
plane_colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6", "#1ABC9C", "#D35400", "#2C3E50"]

for i, p in enumerate(planes):
    try:
        X, Y, Z = make_plane_mesh(*p, extent=plot_extent)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, plane_colors[i % 8]], [1, plane_colors[i % 8]]],
                                 opacity=0.3, showscale=False, name=f"Plane {i+1}"))
    except: continue

if show_intersections:
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            pts = get_intersection_line(planes[i], planes[j], plot_extent)
            if pts is not None:
                line_color = blend_colors(plane_colors[i % 8], plane_colors[j % 8])
                fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines',
                                           line=dict(color=line_color, width=6), name=f"Int {i+1}&{j+1}"))

if x_ls is not None:
    if show_projections:
        for i, (p, n) in enumerate(zip(planes, normals)):
            d_val = np.dot(n, x_ls) + (p[3] / np.linalg.norm(p[:3]))
            p_on_plane = x_ls - d_val * n
            fig.add_trace(go.Scatter3d(x=[x_ls[0], p_on_plane[0]], y=[x_ls[1], p_on_plane[1]], z=[x_ls[2], p_on_plane[2]],
                                       mode='lines', line=dict(color=plane_colors[i % 8], width=4), showlegend=False))

    fig.add_trace(go.Scatter3d(x=[x_ls[0]], y=[x_ls[1]], z=[x_ls[2]], mode='markers+text',
                               marker=dict(size=10, color='black', symbol='diamond', line=dict(width=2, color='white')),
                               text=["LS Solution"], textposition="top center", name="LS Solution"))

fig.update_layout(scene=dict(xaxis=dict(range=[-plot_extent, plot_extent]),
                             yaxis=dict(range=[-plot_extent, plot_extent]),
                             zaxis=dict(range=[-plot_extent, plot_extent]), aspectmode='cube'),
                  height=750, margin=dict(l=0, r=0, b=0, t=0))

st.plotly_chart(fig, use_container_width=True)

# ---------- Results & Education Panel ----------

st.markdown("---")
st.markdown("""
    <style>
    .section-header {
        font-size: 36px !important;
        font-weight: bold;
    }
    .big-font {
        font-size: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.markdown('<p class="section-header">📊 Numerical Analysis</p>', unsafe_allow_html=True)
    if x_ls is not None:
        st.markdown('<p class="big-font">Optimal Point (<b>x</b><sub>LS</sub>):</p>', unsafe_allow_html=True)
        st.latex(rf"x={x_ls[0]:.4f}, \quad y={x_ls[1]:.4f}, \quad z={x_ls[2]:.4f}")
        st.metric("Total RMS Error", f"{rms_error:.5f}")
        with st.expander("Residuals Detail"):
            for i, d in enumerate(dists): 
                st.markdown(f'<p class="big-font">Plane {i+1}: {d:.5f}</p>', unsafe_allow_html=True)

with res_col2:
    st.markdown('<p class="section-header">🎓 Learn the Math: LSE via Matrices</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">The system is represented as <b>A x</b> = <b>y</b>. In an overdetermined system, we seek the solution that minimizes the sum of squared distances to all planes.</p>', unsafe_allow_html=True)
    
    st.latex(r"""
    A = \begin{bmatrix} 
    A_1 & B_1 & C_1 \\ 
    \vdots & \vdots & \vdots \\ 
    A_m & B_m & C_m 
    \end{bmatrix}, \quad
    \mathbf{y} = \begin{bmatrix} -D_1 \\ \vdots \\ -D_m \end{bmatrix}
    """)
    
    st.markdown('<p class="big-font">This leads to the <b>Normal Equation</b>, providing the unique Least Squares solution:</p>', unsafe_allow_html=True)
    st.latex(r"A^T A \mathbf{x} = A^T \mathbf{y} \quad \implies \quad \mathbf{x}_{LS} = (A^T A)^{-1} A^T \mathbf{y}")
    st.info("The short lines (residuals) visualize the perpendicular distance from the LS point to each plane.")