# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 12:28:36 2026

@author: Brayden Miao
"""

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from streamlit_plotly_events import plotly_events


# ---------- Core math functions ----------

def generate_random_cluster_3d(n_points=30, mean=(0.0, 0.0, 0.0), cov=None, seed=0):
    if cov is None:
        cov = np.array([
            [1.0, 0.4, 0.2],
            [0.4, 1.2, 0.3],
            [0.2, 0.3, 0.8],
        ])
    rng = np.random.default_rng(seed)
    pts = rng.multivariate_normal(mean, cov, size=n_points)
    return pts  # (n_points, 3)


def apply_linear_transform_3d(points, A):
    # Row-vector convention
    return points @ A.T


def rotation_matrix_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])
    return R


def axis_angle_from_rot(R):
    R = np.asarray(R, dtype=float)
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / axis_norm

    return axis, theta


def make_rotational_svd_3d(A):
    U_raw, s, Vt_raw = np.linalg.svd(A)
    Sigma = np.diag(s)
    V_raw = Vt_raw.T

    U_rot = U_raw.copy()
    V_rot = V_raw.copy()
    Sigma_signed = Sigma.copy()

    detU = np.linalg.det(U_rot)
    detV = np.linalg.det(V_rot)
    R_ref = np.diag([1.0, 1.0, -1.0])

    if detU < 0 and detV < 0:
        U_rot = U_rot @ R_ref
        V_rot = V_rot @ R_ref
    elif detU < 0 and detV >= 0:
        U_rot = U_rot @ R_ref
        Sigma_signed = R_ref @ Sigma_signed
    elif detU >= 0 and detV < 0:
        V_rot = V_rot @ R_ref
        Sigma_signed = Sigma_signed @ R_ref

    return U_rot, Sigma_signed, V_rot, s, Vt_raw


def svd_path_transform_3d(t, V_rot, Sigma_signed, U_rot):
    axis_V, theta_V = axis_angle_from_rot(V_rot)
    axis_U, theta_U = axis_angle_from_rot(U_rot)

    s1_signed = Sigma_signed[0, 0]
    s2_signed = Sigma_signed[1, 1]
    s3_signed = Sigma_signed[2, 2]

    if t <= 1.0:
        alpha = t
        Rv_t = rotation_matrix_axis_angle(axis_V, alpha * theta_V)
        M = Rv_t
    elif t <= 2.0:
        alpha = t - 1.0
        Rv = rotation_matrix_axis_angle(axis_V, theta_V)
        s1_t = 1.0 + alpha * (s1_signed - 1.0)
        s2_t = 1.0 + alpha * (s2_signed - 1.0)
        s3_t = 1.0 + alpha * (s3_signed - 1.0)
        S_t = np.diag([s1_t, s2_t, s3_t])
        M = Rv @ S_t
    else:
        alpha = t - 2.0
        Rv = rotation_matrix_axis_angle(axis_V, theta_V)
        S_final = Sigma_signed
        Ru_t = rotation_matrix_axis_angle(axis_U, -alpha * theta_U)
        M = Rv @ S_final @ Ru_t

    return M


# ---------- Geometry helpers ----------

def build_outer_cube(points):
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)

    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span_x, span_y, span_z = xmax - xmin, ymax - ymin, zmax - zmin
    half_side = 0.5 * max(span_x, span_y, span_z) * 1.3 + 1e-6

    corners = np.array([
        [cx - half_side, cy - half_side, cz - half_side],
        [cx + half_side, cy - half_side, cz - half_side],
        [cx + half_side, cy + half_side, cz - half_side],
        [cx - half_side, cy + half_side, cz - half_side],
        [cx - half_side, cy - half_side, cz + half_side],
        [cx + half_side, cy - half_side, cz + half_side],
        [cx + half_side, cy + half_side, cz + half_side],
        [cx - half_side, cy + half_side, cz + half_side],
    ])

    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return corners, edge_pairs


def build_edge_lines(corners, edge_pairs):
    xs, ys, zs = [], [], []
    for i, j in edge_pairs:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])
    return xs, ys, zs


def compute_bounds_3d(points_list, margin_factor=1.1):
    all_pts = np.vstack(points_list)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)

    center = (mins + maxs) / 2
    spans = maxs - mins
    half = 0.5 * spans.max() * margin_factor

    x_range = [center[0] - half, center[0] + half]
    y_range = [center[1] - half, center[1] + half]
    z_range = [center[2] - half, center[2] + half]

    return x_range, y_range, z_range, center, half


# ---------- Mesh primitives for robust point rendering ----------

def make_unit_sphere_mesh(n_theta=10, n_phi=18):
    thetas = np.linspace(0.0, np.pi, n_theta)
    phis = np.linspace(0.0, 2*np.pi, n_phi, endpoint=False)

    verts = []
    for th in thetas:
        for ph in phis:
            x = np.sin(th) * np.cos(ph)
            y = np.sin(th) * np.sin(ph)
            z = np.cos(th)
            verts.append((x, y, z))
    verts = np.array(verts, dtype=float)

    def idx(ti, pi):
        return ti * n_phi + (pi % n_phi)

    I, J, K = [], [], []
    for ti in range(n_theta - 1):
        for pi in range(n_phi):
            a = idx(ti, pi)
            b = idx(ti, pi + 1)
            c = idx(ti + 1, pi)
            d = idx(ti + 1, pi + 1)
            I.extend([a, b])
            J.extend([c, c])
            K.extend([b, d])

    vx, vy, vz = verts[:, 0].tolist(), verts[:, 1].tolist(), verts[:, 2].tolist()
    return vx, vy, vz, I, J, K


def pack_spheres(points, radius, base_mesh):
    vx0, vy0, vz0, I0, J0, K0 = base_mesh
    v0 = np.column_stack([vx0, vy0, vz0])
    V = v0.shape[0]

    all_v = []
    all_I, all_J, all_K = [], [], []
    offset = 0

    for p in points:
        v = radius * v0 + p[None, :]
        all_v.append(v)
        all_I.extend([ii + offset for ii in I0])
        all_J.extend([jj + offset for jj in J0])
        all_K.extend([kk + offset for kk in K0])
        offset += V

    all_v = np.vstack(all_v)
    return all_v[:, 0].tolist(), all_v[:, 1].tolist(), all_v[:, 2].tolist(), all_I, all_J, all_K


def build_3d_cross_segments(points, half_len):
    xs, ys, zs = [], [], []
    for p in points:
        x, y, z = p

        xs.extend([x - half_len, x + half_len, None])
        ys.extend([y, y, None])
        zs.extend([z, z, None])

        xs.extend([x, x, None])
        ys.extend([y - half_len, y + half_len, None])
        zs.extend([z, z, None])

        xs.extend([x, x, None])
        ys.extend([y, y, None])
        zs.extend([z - half_len, z + half_len, None])

    return xs, ys, zs


# ---------- Camera persistence via plotly_events ----------

DEFAULT_CAMERA = dict(
    eye=dict(x=1.6, y=1.6, z=1.2),
    up=dict(x=0, y=0, z=1),
)


def update_camera_from_events(events):
    if not events:
        return
    for e in events:
        if not isinstance(e, dict):
            continue

        cam = None
        if "scene.camera" in e and isinstance(e["scene.camera"], dict):
            cam = e["scene.camera"]
        else:
            cam_update = {}
            for k, v in e.items():
                if isinstance(k, str) and k.startswith("scene.camera."):
                    cam_update[k.replace("scene.camera.", "")] = v

            if cam_update:
                base = st.session_state.get("plotly_camera", DEFAULT_CAMERA)
                cam = {
                    "eye": dict(base.get("eye", DEFAULT_CAMERA["eye"])),
                    "up": dict(base.get("up", DEFAULT_CAMERA["up"])),
                }
                for subk, subv in cam_update.items():
                    if "." in subk:
                        top, leaf = subk.split(".", 1)
                        cam.setdefault(top, {})
                        try:
                            cam[top][leaf] = float(subv)
                        except Exception:
                            cam[top][leaf] = subv

        if cam is not None:
            if "eye" not in cam or not isinstance(cam["eye"], dict):
                cam = DEFAULT_CAMERA.copy()
            else:
                for ax in ("x", "y", "z"):
                    if ax not in cam["eye"]:
                        cam = DEFAULT_CAMERA.copy()
                        break
            st.session_state.plotly_camera = cam


# ---------- Plotly 3D figure (mesh points template) ----------

def make_3d_figure(points,
                   points_T,
                   cube_corners,
                   cube_corners_T,
                   cube_edge_pairs,
                   show_arrows,
                   A,
                   x_range,
                   y_range,
                   z_range,
                   half,
                   camera,
                   uirevision_key,
                   t,
                   axis_V,
                   axis_U):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", showlegend=True)

    # Color scheme
    C_ORIG_PTS = "#0b3d91"    # deep blue
    C_TRANS_PTS = "#ff7f0e"   # orange
    C_CUBE_ORIG = "#444444"   # slightly gray
    C_CUBE_TRANS = "#ff2b2b"  # red
    C_DISP = "#6f93a6"        # darker blue-gray

    EIG_COLORS = ["#74d37a", "#ff7f0e", "#f2c14e"]

    # ---- Rotation-axis guides (NOT in legend) ----
    def add_rotation_axis(axis_vec, length, opacity, color):
        if opacity <= 0:
            return
        a = np.asarray(axis_vec, dtype=float)
        n = np.linalg.norm(a)
        if n < 1e-12:
            return
        a = a / n
        p0 = -length * a
        p1 = length * a
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines",
            line=dict(width=2, color=color, dash="dash"),
            opacity=float(opacity),
            showlegend=True,
            hoverinfo="skip",
            name="__rot_axis__",
        ))

    # ---- Original points: Mesh3d spheres ----
    base_sphere = make_unit_sphere_mesh(n_theta=10, n_phi=18)
    scene_span = (x_range[1] - x_range[0])
    sphere_radius = 0.008 * scene_span
    sx, sy, sz, si, sj, sk = pack_spheres(points, sphere_radius, base_sphere)

    fig.add_trace(go.Mesh3d(
        x=sx, y=sy, z=sz,
        i=si, j=sj, k=sk,
        color=C_ORIG_PTS,
        opacity=1.0,
        name="Original points",
        showlegend=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.2, roughness=0.8),
    ))

    # Legend-only marker (blue circle)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=9, symbol="circle", color=C_ORIG_PTS, opacity=1.0),
        name="Original points",
        visible="legendonly",
        showlegend=True,
        legendrank=10
    ))

    # ---- Transformed points: 3D crosses using line segments ----
    cross_half_len = 0.012 * scene_span
    cx, cy, cz = build_3d_cross_segments(points_T, cross_half_len)

    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="lines",
        line=dict(width=4, color=C_TRANS_PTS),
        name="Transformed points",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Legend-only marker (orange +)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=10, symbol="cross", color=C_TRANS_PTS, line=dict(width=0.2, color=C_TRANS_PTS)),
        name="Transformed points",
        visible="legendonly",
        showlegend=True,
        legendrank=20
    ))

    # Original cube
    cube_edges = build_edge_lines(cube_corners, cube_edge_pairs)
    fig.add_trace(go.Scatter3d(
        x=cube_edges[0], y=cube_edges[1], z=cube_edges[2],
        mode="lines",
        line=dict(width=2, color=C_CUBE_ORIG),
        name="Original cube",
        showlegend=True,
        legendrank=30
    ))

    # Transformed cube
    cube_edges_T = build_edge_lines(cube_corners_T, cube_edge_pairs)
    fig.add_trace(go.Scatter3d(
        x=cube_edges_T[0], y=cube_edges_T[1], z=cube_edges_T[2],
        mode="lines",
        line=dict(width=3, color=C_CUBE_TRANS),
        name="Transformed cube",
        showlegend=True,
        legendrank=40
    ))

    # Displacement lines (darker)
    if show_arrows:
        xs, ys, zs = [], [], []
        for p, q in zip(points, points_T):
            xs.extend([p[0], q[0], None])
            ys.extend([p[1], q[1], None])
            zs.extend([p[2], q[2], None])

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=2, color=C_DISP),
            name="Point displacements",
            opacity=0.45,
            showlegend=True,
            legendrank=50
        ))

    # Invisible hover targets
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.0),
        showlegend=False,
        hovertemplate="Original<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="__hover_orig__",
    ))
    fig.add_trace(go.Scatter3d(
        x=points_T[:, 0], y=points_T[:, 1], z=points_T[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.0),
        showlegend=False,
        hovertemplate="Transformed<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="__hover_trans__",
    ))

    # Eigenvectors (real only)
    evals, evecs = np.linalg.eig(A)
    real_mask = np.isclose(evals.imag, 0.0, atol=1e-8)
    evals_real = evals[real_mask].real
    evecs_real = evecs[:, real_mask].real

    if evals_real.size > 0:
        max_abs_lambda = np.max(np.abs(evals_real)) or 1.0
        base_scale = 0.7 * half / max_abs_lambda
        for i, lam in enumerate(evals_real):
            v = evecs_real[:, i]
            v = v / np.linalg.norm(v)
            length = base_scale * abs(lam)
            end = length * v
            col = EIG_COLORS[i % len(EIG_COLORS)]
            fig.add_trace(go.Scatter3d(
                x=[0, end[0]],
                y=[0, end[1]],
                z=[0, end[2]],
                mode="lines+markers",
                marker=dict(size=5, color=col),
                line=dict(width=6, color=col),
                name=f"eigenvector (λ={lam:.2f})",
            ))

    # Add SVD rotation-axis guides (u_V and u_U), with stage-dependent opacity
    base_op = 0.85
    if t <= 1.0:
        op_V = base_op
        op_U = 0.0
    elif t <= 2.0:
        op_V = base_op * (2.0 - t)
        op_U = 0.0
    else:
        op_V = 0.0
        op_U = base_op

    axis_len = 0.95 * half
    add_rotation_axis(axis_V, axis_len, op_V, color="#8A2BE2")  # purple
    add_rotation_axis(axis_U, axis_len, op_U, color="#00B3B3")  # teal

    fig.update_layout(
        uirevision=uirevision_key,
        scene=dict(
            xaxis=dict(range=x_range, title="x", showbackground=False, showgrid=True, zeroline=False),
            yaxis=dict(range=y_range, title="y", showbackground=False, showgrid=True, zeroline=False),
            zaxis=dict(range=z_range, title="z", showbackground=False, showgrid=True, zeroline=False),
            aspectmode="cube",
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98),
        title="3D Linear Transformation: points, cube, eigenvectors",
        height=1000,
    )

    return fig


# ---------- Matplotlib 3D animation (GIF) ----------

def create_animation_gif_3d(filename,
                            points,
                            cube_corners,
                            cube_edge_pairs,
                            A,
                            U_rot,
                            Sigma_signed,
                            V_rot,
                            n_frames=150,
                            fps=30,
                            show_arrows=True):
    M_final = V_rot @ Sigma_signed @ U_rot.T
    points_final = points @ M_final
    cube_final = cube_corners @ M_final

    all_pts = np.vstack([points, points_final, cube_corners, cube_final])
    x_range, y_range, z_range, center, half = compute_bounds_3d([all_pts])

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    evals, evecs = np.linalg.eig(A)
    real_mask = np.isclose(evals.imag, 0.0, atol=1e-8)
    evals_real = evals[real_mask].real
    evecs_real = evecs[:, real_mask].real

    writer = PillowWriter(fps=fps)

    with writer.saving(fig, filename, dpi=200):
        for i in range(n_frames):
            t = 3.0 * i / (n_frames - 1)
            M = svd_path_transform_3d(t, V_rot, Sigma_signed, U_rot)
            pts_M = points @ M
            cube_M = cube_corners @ M

            ax.cla()

            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       s=8, alpha=0.9, label="Original points")

            ax.scatter(pts_M[:, 0], pts_M[:, 1], pts_M[:, 2],
                       s=12, marker="x", alpha=0.9, label="Transformed points")

            for (i1, i2) in cube_edge_pairs:
                ax.plot([cube_corners[i1, 0], cube_corners[i2, 0]],
                        [cube_corners[i1, 1], cube_corners[i2, 1]],
                        [cube_corners[i1, 2], cube_corners[i2, 2]],
                        "k-", linewidth=1.5)

            for (i1, i2) in cube_edge_pairs:
                ax.plot([cube_M[i1, 0], cube_M[i2, 0]],
                        [cube_M[i1, 1], cube_M[i2, 1]],
                        [cube_M[i1, 2], cube_M[i2, 2]],
                        "r-", linewidth=1.5)

            if show_arrows:
                for p, q in zip(points, pts_M):
                    ax.plot([p[0], q[0]],
                            [p[1], q[1]],
                            [p[2], q[2]],
                            color="gray", linewidth=0.8, alpha=0.4)

            if evals_real.size > 0 and i == n_frames - 1:
                max_abs_lambda = np.max(np.abs(evals_real)) or 1.0
                base_scale = 0.7 * half / max_abs_lambda
                for lam, v in zip(evals_real, evecs_real.T):
                    v = v / np.linalg.norm(v)
                    length = base_scale * abs(lam)
                    end = length * v
                    ax.plot([0, end[0]],
                            [0, end[1]],
                            [0, end[2]],
                            linewidth=2.5,
                            color="green")
                    ax.scatter(end[0], end[1], end[2], s=15, color="green")

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_box_aspect((1, 1, 1))

            ax.view_init(elev=30, azim=45)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax.set_title(f"3D SVD path: t = {t:.2f}")
            ax.legend(loc="upper left")

            writer.grab_frame()

    plt.close(fig)


# ---------- Helper to format eigenvalues ----------

def format_eig_latex(ev, tol=1e-8):
    re = float(ev.real)
    im = float(ev.imag)
    if abs(im) < tol:
        return f"{re:.3f}"
    sign = "+" if im >= 0 else "-"
    return f"{re:.3f} {sign} {abs(im):.3f}i"


# ---------- Streamlit app ----------

def main():
    st.set_page_config(
        page_title="3D Linear Transformation (SVD Slider + GIF)",
        layout="wide"
    )

    # Camera persistence state
    if "plotly_camera" not in st.session_state:
        st.session_state.plotly_camera = DEFAULT_CAMERA.copy()
    if "uirevision_key" not in st.session_state:
        st.session_state.uirevision_key = "keep_camera_mesh_template_v1"

    st.title("3D Linear Transformation & Eigenvectors (SVD Slider + GIF)")

    st.write(
        """
        This app shows how a **3×3 matrix** transforms a random cluster of **30 points** in 3D,
        plus an outer reference cube, using a **rotation–stretch–rotation** SVD path.
        """
    )

    # Sidebar: random seed
    st.sidebar.header("Random points")
    seed = st.sidebar.slider("Random seed", 0, 100, 0, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("Transformation matrix A")

    mode = st.sidebar.radio(
        "Define A by:",
        ["Diagonal scaling", "XYZ rotation + scaling", "Symmetric 3×3", "Manual 3×3"],
        index=2,  # default to Symmetric 3×3
    )

    if mode == "Diagonal scaling":
        sx = st.sidebar.slider("Scale in x", 0.1, 3.0, 1.5, 0.1)
        sy = st.sidebar.slider("Scale in y", 0.1, 3.0, 1.0, 0.1)
        sz = st.sidebar.slider("Scale in z", 0.1, 3.0, 0.7, 0.1)
        A = np.diag([sx, sy, sz])

    elif mode == "XYZ rotation + scaling":
        st.sidebar.write("Rotation angles (degrees):")
        theta_x_deg = st.sidebar.slider("Rotate about x-axis", -180, 180, 20)
        theta_y_deg = st.sidebar.slider("Rotate about y-axis", -180, 180, 30)
        theta_z_deg = st.sidebar.slider("Rotate about z-axis", -180, 180, 40)

        st.sidebar.write("Scaling along axes:")
        sx = st.sidebar.slider("Scale in x", 0.1, 3.0, 1.5, 0.1)
        sy = st.sidebar.slider("Scale in y", 0.1, 3.0, 1.0, 0.1)
        sz = st.sidebar.slider("Scale in z", 0.1, 3.0, 0.7, 0.1)

        tx = np.deg2rad(theta_x_deg)
        ty = np.deg2rad(theta_y_deg)
        tz = np.deg2rad(theta_z_deg)

        cx, sx_sin = np.cos(tx), np.sin(tx)
        cy, sy_sin = np.cos(ty), np.sin(ty)
        cz, sz_sin = np.cos(tz), np.sin(tz)

        Rx = np.array([
            [1.0,    0.0,     0.0],
            [0.0,    cx,   -sx_sin],
            [0.0,  sx_sin,    cx],
        ])
        Ry = np.array([
            [cy,      0.0,     sy_sin],
            [0.0,     1.0,     0.0],
            [-sy_sin, 0.0,     cy],
        ])
        Rz = np.array([
            [cz, -sz_sin, 0.0],
            [sz_sin,  cz, 0.0],
            [0.0,    0.0, 1.0],
        ])

        R = Rz @ Ry @ Rx
        S = np.diag([sx, sy, sz])
        A = R @ S

    elif mode == "Symmetric 3×3":
        st.sidebar.write("Enter entries; off-diagonals are tied (a_ij = a_ji):")

        row1 = st.sidebar.columns(3)
        with row1[0]:
            a11 = st.number_input("a11", value=1.0, step=0.1, key="sym_a11")
        with row1[1]:
            a12 = st.number_input("a12", value=0.2, step=0.1, key="sym_a12")
        with row1[2]:
            a13 = st.number_input("a13", value=0.0, step=0.1, key="sym_a13")

        row2 = st.sidebar.columns(3)
        with row2[0]:
            st.markdown(f"a21 = {a12:.3f}")
        with row2[1]:
            a22 = st.number_input("a22", value=1.0, step=0.1, key="sym_a22")
        with row2[2]:
            a23 = st.number_input("a23", value=0.1, step=0.1, key="sym_a23")

        row3 = st.sidebar.columns(3)
        with row3[0]:
            st.markdown(f"a31 = {a13:.3f}")
        with row3[1]:
            st.markdown(f"a32 = {a23:.3f}")
        with row3[2]:
            a33 = st.number_input("a33", value=1.0, step=0.1, key="sym_a33")

        A = np.array([
            [a11, a12, a13],
            [a12, a22, a23],
            [a13, a23, a33],
        ])

    else:
        st.sidebar.write("Enter the entries of the 3×3 matrix A:")
        c1, c2, c3 = st.sidebar.columns(3)
        with c1:
            a11 = st.number_input("a11", value=1.0, step=0.1, key="a11")
            a21 = st.number_input("a21", value=0.0, step=0.1, key="a21")
            a31 = st.number_input("a31", value=0.0, step=0.1, key="a31")
        with c2:
            a12 = st.number_input("a12", value=0.0, step=0.1, key="a12")
            a22 = st.number_input("a22", value=1.0, step=0.1, key="a22")
            a32 = st.number_input("a32", value=0.0, step=0.1, key="a32")
        with c3:
            a13 = st.number_input("a13", value=0.0, step=0.1, key="a13")
            a23 = st.number_input("a23", value=0.0, step=0.1, key="a23")
            a33 = st.number_input("a33", value=1.0, step=0.1, key="a33")

        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33],
        ])

    show_arrows = st.sidebar.checkbox(
        "Show arrows from original to transformed points",
        value=True,
    )

    # Rotational SVD for CURRENT A
    U_rot, Sigma_signed, V_rot, s, Vt_raw = make_rotational_svd_3d(A)

    axis_V, theta_V = axis_angle_from_rot(V_rot)
    axis_U, theta_U = axis_angle_from_rot(U_rot)
    theta_V_deg = float(np.rad2deg(theta_V))
    theta_U_deg = float(np.rad2deg(theta_U))

    # Random 3D data + cube
    points = generate_random_cluster_3d(n_points=30, seed=seed)
    cube_corners, cube_edge_pairs = build_outer_cube(points)

    # ---- FIX CLIPPING: compute bounds across the whole SVD path ----
    Ms = [
        svd_path_transform_3d(0.0, V_rot, Sigma_signed, U_rot),
        svd_path_transform_3d(1.0, V_rot, Sigma_signed, U_rot),
        svd_path_transform_3d(2.0, V_rot, Sigma_signed, U_rot),
        svd_path_transform_3d(3.0, V_rot, Sigma_signed, U_rot),
    ]
    clouds = [points, cube_corners]
    for Mtmp in Ms:
        clouds.append(points @ Mtmp)
        clouds.append(cube_corners @ Mtmp)

    x_range, y_range, z_range, center, half = compute_bounds_3d(clouds, margin_factor=1.30)

    # Slider
    st.sidebar.markdown("---")
    t = st.sidebar.slider(
        "SVD path (0 = I, 1 = V, 2 = V+Σ̃, 3 = full A via Uᵀ)",
        min_value=0.0,
        max_value=3.0,
        value=3.0,
        step=0.01,
    )

    M = svd_path_transform_3d(t, V_rot, Sigma_signed, U_rot)
    points_t = points @ M
    cube_t = cube_corners @ M

    # Show A
    st.subheader("Transformation matrix A")
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        %.3f & %.3f & %.3f \\
        %.3f & %.3f & %.3f \\
        %.3f & %.3f & %.3f
        \end{bmatrix}
        """
        % (
            A[0, 0], A[0, 1], A[0, 2],
            A[1, 0], A[1, 1], A[1, 2],
            A[2, 0], A[2, 1], A[2, 2],
        )
    )

    # Eigenvalues
    evals, evecs = np.linalg.eig(A)
    lam1 = format_eig_latex(evals[0])
    lam2 = format_eig_latex(evals[1])
    lam3 = format_eig_latex(evals[2])

    st.subheader("Eigenvalues of A")
    st.latex(
        rf"""
        \lambda_1 = {lam1},\quad
        \lambda_2 = {lam2},\quad
        \lambda_3 = {lam3}
        """
    )

    real_mask = np.isclose(evals.imag, 0.0, atol=1e-8)
    n_real = real_mask.sum()
    if n_real > 0:
        st.write(
            f"We draw eigenvectors in 3D for the {n_real} eigenvalue(s) "
            "whose imaginary part is essentially zero. "
            "Eigenvectors corresponding to complex eigenvalues are not drawn."
        )
    else:
        st.write(
            "All three eigenvalues have nonzero imaginary parts. "
            "Their eigenvectors live in complex 3D space, so we do not draw any eigenvectors in the 3D diagram."
        )

    # Layout
    col1, col2 = st.columns([4, 1])

    with col1:
        st.subheader("3D Visualization (drag to rotate, view persists)")

        fig_3d = make_3d_figure(
            points=points,
            points_T=points_t,
            cube_corners=cube_corners,
            cube_corners_T=cube_t,
            cube_edge_pairs=cube_edge_pairs,
            show_arrows=show_arrows,
            A=A,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            half=half,
            camera=st.session_state.plotly_camera,
            uirevision_key=st.session_state.uirevision_key,
            t=t,
            axis_V=axis_V,
            axis_U=axis_U,
        )

        events = plotly_events(
            fig_3d,
            click_event=False,
            select_event=False,
            hover_event=True,
            override_height=1000,
            key="plotly_events_main",
        )
        update_camera_from_events(events)

        # Removed the Reset camera button block
        st.caption("Rotate/zoom the plot, click it once somewhere in the figure, then move the SVD slider. The view should stay.")

    with col2:
        st.subheader("Rotation + stretching view (3D rotational SVD)")
        st.markdown(
            r"""
We build a rotation-friendly SVD:

$$
A = \tilde U \,\tilde\Sigma\, \tilde V^T,
$$

where:
- $\tilde U$ and $\tilde V$ are proper rotations (determinant $=1$),
- $\tilde\Sigma$ is diagonal and may carry signs (a signed scaling).

---

**Axis–angle explanation.**  
Any $3\times 3$ rotation matrix $R$ can be written as a rotation by an angle $\theta$
around some unit axis $\mathbf u$:

$$
R = I \cos\theta + (1-\cos\theta)\,\mathbf u \mathbf u^T + \sin\theta\,[\mathbf u]_\times,
$$

where $[\mathbf u]_\times$ is the skew-symmetric matrix built from $\mathbf u$.
We use this axis–angle form (Rodrigues’ formula) to interpolate smoothly
from the identity to $\tilde V$ and to $\tilde U^T$.
"""
        )

        Vt_disp = V_rot.T
        st.latex(
            r"""
            \tilde V^T \approx
            \begin{bmatrix}
            %.3f & %.3f & %.3f \\
            %.3f & %.3f & %.3f \\
            %.3f & %.3f & %.3f
            \end{bmatrix}
            """
            % (
                Vt_disp[0, 0], Vt_disp[0, 1], Vt_disp[0, 2],
                Vt_disp[1, 0], Vt_disp[1, 1], Vt_disp[1, 2],
                Vt_disp[2, 0], Vt_disp[2, 1], Vt_disp[2, 2],
            )
        )
        st.latex(
            r"""
            \mathbf u_V \approx
            \begin{bmatrix}
            %.3f \\ %.3f \\ %.3f
            \end{bmatrix}
            """
            % (axis_V[0], axis_V[1], axis_V[2])
        )
        st.latex(r"\theta_V \approx %.1f^\circ" % theta_V_deg)

        st.latex(
            r"""
            \tilde\Sigma \approx
            \begin{bmatrix}
            %.3f & 0 & 0 \\
            0 & %.3f & 0 \\
            0 & 0 & %.3f
            \end{bmatrix}
            """
            % (Sigma_signed[0, 0], Sigma_signed[1, 1], Sigma_signed[2, 2])
        )

        st.latex(
            r"""
            \tilde U \approx
            \begin{bmatrix}
            %.3f & %.3f & %.3f \\
            %.3f & %.3f & %.3f \\
            %.3f & %.3f & %.3f
            \end{bmatrix}
            """
            % (
                U_rot[0, 0], U_rot[0, 1], U_rot[0, 2],
                U_rot[1, 0], U_rot[1, 1], U_rot[1, 2],
                U_rot[2, 0], U_rot[2, 1], U_rot[2, 2],
            )
        )
        st.latex(
            r"""
            \mathbf u_U \approx
            \begin{bmatrix}
            %.3f \\ %.3f \\ %.3f
            \end{bmatrix}
            """
            % (axis_U[0], axis_U[1], axis_U[2])
        )
        st.latex(r"\theta_U \approx %.1f^\circ" % theta_U_deg)

    st.markdown("---")
    st.caption(
        "Drag the slider slowly while you explain: "
        "I → V → V+Σ̃ → V Σ̃ Uᵀ (= Aᵀ), and connect it to the axis–angle rotations above."
    )

    # ---------- GIF generation ----------
    st.markdown("## GIF animation from the 3D SVD path")

    if st.button("Generate 3D GIF animation (svd3d_animation.gif)"):
        with st.spinner("Generating 3D GIF animation (this may take a bit)..."):
            try:
                create_animation_gif_3d(
                    filename="svd3d_animation.gif",
                    points=points,
                    cube_corners=cube_corners,
                    cube_edge_pairs=cube_edge_pairs,
                    A=A,
                    U_rot=U_rot,
                    Sigma_signed=Sigma_signed,
                    V_rot=V_rot,
                    n_frames=150,
                    fps=30,
                    show_arrows=show_arrows,
                )
                st.success("Animation saved as svd3d_animation.gif")
            except Exception as e:
                st.error(f"Failed to create animation. Error: {e}")

    if os.path.exists("svd3d_animation.gif"):
        vc1, vc2, vc3 = st.columns([1, 2, 1])
        with vc2:
            st.image("svd3d_animation.gif")


if __name__ == "__main__":
    main()
