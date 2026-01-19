# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 11:09:04 2026

@author: Brayden Miao
"""

"""
3x2 linear transformation demo: R^2 -> R^3 via SVD with 3-stage cartoon
Camera persistence mirrors app2x3.py using streamlit-plotly-events.

Run:
    streamlit run app3x2.py
"""

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from streamlit_plotly_events import plotly_events


# ---------- Core helpers ----------

def generate_random_cluster_2d(n_points=30, mean=(0.0, 0.0), cov=None, seed=0):
    if cov is None:
        cov = np.array([
            [1.0, 0.4],
            [0.4, 1.2],
        ])
    rng = np.random.default_rng(seed)
    pts = rng.multivariate_normal(mean, cov, size=n_points)
    return pts  # (n, 2)


def build_outer_square_2d(points_2d):
    xmin, ymin = points_2d.min(axis=0)
    xmax, ymax = points_2d.max(axis=0)

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span_x, span_y = xmax - xmin, ymax - ymin
    half_side = 0.5 * max(span_x, span_y) * 1.3 + 1e-6

    corners = np.array([
        [cx - half_side, cy - half_side],
        [cx + half_side, cy - half_side],
        [cx + half_side, cy + half_side],
        [cx - half_side, cy + half_side],
    ])
    return corners  # (4,2)


def compute_bounds_3d(points_list, margin_factor=1.05):
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


def angle_from_rot2d(R):
    return np.arctan2(R[1, 0], R[0, 0])


def rotation_matrix_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


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


# ---------- SVD cartoon for 3x2 map (R^2 -> R^3) ----------

def svd_positions_3x2(points_2d, t, U, s, V):
    s1, s2 = float(s[0]), float(s[1])

    theta_V = angle_from_rot2d(V)
    R2d_final = V

    if t <= 1.0:
        alpha = t
        R2d_t = rotation_matrix_2d(alpha * theta_V)
        pts2_t = points_2d @ R2d_t
        pts3_t = np.hstack([pts2_t, np.zeros((points_2d.shape[0], 1))])
        return pts3_t

    pts2_rot = points_2d @ R2d_final

    if t <= 2.0:
        beta = t - 1.0
        s1_t = 1.0 + beta * (s1 - 1.0)
        s2_t = 1.0 + beta * (s2 - 1.0)
        D_t = np.diag([s1_t, s2_t])
        pts2_scaled = pts2_rot @ D_t
        pts3_t = np.hstack([pts2_scaled, np.zeros((points_2d.shape[0], 1))])
        return pts3_t

    gamma = t - 2.0
    axis_U, theta_U = axis_angle_from_rot(U)
    R3_t = rotation_matrix_axis_angle(axis_U, gamma * theta_U)

    D_final = np.diag([s1, s2])
    pts2_final = pts2_rot @ D_final
    pts3_base = np.hstack([pts2_final, np.zeros((points_2d.shape[0], 1))])

    pts3_t = pts3_base @ R3_t.T
    return pts3_t


def plane_coords_3x2(points_2d, t, s, V):
    s1, s2 = float(s[0]), float(s[1])
    theta_V = angle_from_rot2d(V)

    if t <= 1.0:
        alpha = t
        R2d_t = rotation_matrix_2d(alpha * theta_V)
        return points_2d @ R2d_t

    pts2_rot = points_2d @ V

    if t <= 2.0:
        beta = t - 1.0
        s1_t = 1.0 + beta * (s1 - 1.0)
        s2_t = 1.0 + beta * (s2 - 1.0)
        D_t = np.diag([s1_t, s2_t])
        return pts2_rot @ D_t

    D_final = np.diag([s1, s2])
    return pts2_rot @ D_final


# ---------- Mesh primitives (robust markers like app2x3) ----------

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
        xs.extend([x - half_len, x + half_len, None]); ys.extend([y, y, None]); zs.extend([z, z, None])
        xs.extend([x, x, None]); ys.extend([y - half_len, y + half_len, None]); zs.extend([z, z, None])
        xs.extend([x, x, None]); ys.extend([y, y, None]); zs.extend([z - half_len, z + half_len, None])
    return xs, ys, zs


# ---------- Camera persistence (mirrors app2x3.py logic) ----------

DEFAULT_CAMERA_3X2 = dict(
    eye=dict(x=1.6, y=1.6, z=1.2),
    up=dict(x=0, y=0, z=1),
)

def update_camera_from_events_3x2(events, state_key="plotly_camera_3x2"):
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
                base = st.session_state.get(state_key, DEFAULT_CAMERA_3X2)
                cam = {
                    "eye": dict(base.get("eye", DEFAULT_CAMERA_3X2["eye"])),
                    "up": dict(base.get("up", DEFAULT_CAMERA_3X2["up"])),
                    "center": dict(base.get("center", {})) if isinstance(base.get("center", {}), dict) else {},
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
                cam = DEFAULT_CAMERA_3X2.copy()
            else:
                for ax in ("x", "y", "z"):
                    if ax not in cam["eye"]:
                        cam = DEFAULT_CAMERA_3X2.copy()
                        break
            st.session_state[state_key] = cam


# ---------- Plotly 3D figure ----------

def make_3d_figure(points_2d, points_3d_t, square_2d, square_3d_t, t, U, V,
                   show_arrows=True, camera=None, uirevision_key=None):
    pts3_orig = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])
    square3_orig = np.hstack([square_2d, np.zeros((square_2d.shape[0], 1))])

    pts_for_bounds = np.vstack([pts3_orig, points_3d_t, square3_orig, square_3d_t])
    x_range, y_range, z_range, center, half = compute_bounds_3d([pts_for_bounds], margin_factor=1.35)
    scene_span = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])

    sphere_radius = 0.010 * scene_span
    cross_half_len = 0.018 * scene_span

    C_ORIG_PTS = "#0b3d91"
    C_TRANS_PTS = "#ff7f0e"
    C_SQUARE_EDGE = "#ff2b2b"
    C_ARROW = "#6f93a6"
    C_EDGE_ORIG = "#444444"

    fig = go.Figure()
    fig.update_layout(template="plotly_white", showlegend=True)

    # Original points as spheres
    base_sphere = make_unit_sphere_mesh(n_theta=10, n_phi=18)
    sx, sy, sz, si, sj, sk = pack_spheres(pts3_orig, sphere_radius, base_sphere)
    fig.add_trace(go.Mesh3d(
        x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
        color=C_ORIG_PTS, opacity=1.0,
        name="Original points",
        showlegend=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.15, roughness=0.85),
    ))
    # Legend-only marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=9, symbol="circle", color=C_ORIG_PTS, opacity=1.0),
        name="Original points",
        visible="legendonly",
        showlegend=True,
        legendrank=10,
    ))

    # Transformed points as 3D crosses
    cx, cy, cz = build_3d_cross_segments(points_3d_t, cross_half_len)
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="lines",
        line=dict(width=4, color=C_TRANS_PTS),
        name="Transformed points",
        showlegend=False,
        hoverinfo="skip",
    ))
    # Legend-only marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=10, symbol="cross", color=C_TRANS_PTS),
        name="Transformed points",
        visible="legendonly",
        showlegend=True,
        legendrank=20,
    ))

    # Invisible hover targets (helps hover_event still work even with Mesh3d)
    fig.add_trace(go.Scatter3d(
        x=pts3_orig[:, 0], y=pts3_orig[:, 1], z=pts3_orig[:, 2],
        mode="markers",
        marker=dict(size=4, opacity=0.0),
        showlegend=False,
        hovertemplate="Original<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="__hover_orig__",
    ))
    fig.add_trace(go.Scatter3d(
        x=points_3d_t[:, 0], y=points_3d_t[:, 1], z=points_3d_t[:, 2],
        mode="markers",
        marker=dict(size=4, opacity=0.0),
        showlegend=False,
        hovertemplate="Transformed<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="__hover_trans__",
    ))

    # Original square surface + edges
    p00, p01, p11, p10 = square3_orig[0], square3_orig[1], square3_orig[2], square3_orig[3]
    fig.add_trace(go.Mesh3d(
        x=[p00[0], p01[0], p11[0], p10[0]],
        y=[p00[1], p01[1], p11[1], p10[1]],
        z=[p00[2], p01[2], p11[2], p10[2]],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="rgba(255, 43, 43, 0.35)",
        opacity=0.12,
        name="Original square",
        showlegend=True,
        legendrank=30,
        hoverinfo="skip",
    ))
    sq_x = [square3_orig[i, 0] for i in [0, 1, 2, 3, 0]]
    sq_y = [square3_orig[i, 1] for i in [0, 1, 2, 3, 0]]
    sq_z = [square3_orig[i, 2] for i in [0, 1, 2, 3, 0]]
    fig.add_trace(go.Scatter3d(
        x=sq_x, y=sq_y, z=sq_z,
        mode="lines",
        line=dict(width=2, color=C_EDGE_ORIG),
        name="Original square edges",
        showlegend=True,
        legendrank=40,
        hoverinfo="skip",
    ))

    # Transformed square surface + edges
    q00, q01, q11, q10 = square_3d_t[0], square_3d_t[1], square_3d_t[2], square_3d_t[3]
    fig.add_trace(go.Mesh3d(
        x=[q00[0], q01[0], q11[0], q10[0]],
        y=[q00[1], q01[1], q11[1], q10[1],],
        z=[q00[2], q01[2], q11[2], q10[2],],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="rgba(80, 140, 255, 0.45)",
        opacity=0.12,
        name="Transformed square",
        showlegend=True,
        legendrank=50,
        hoverinfo="skip",
    ))
    sqt_x = [square_3d_t[i, 0] for i in [0, 1, 2, 3, 0]]
    sqt_y = [square_3d_t[i, 1] for i in [0, 1, 2, 3, 0]]
    sqt_z = [square_3d_t[i, 2] for i in [0, 1, 2, 3, 0]]
    fig.add_trace(go.Scatter3d(
        x=sqt_x, y=sqt_y, z=sqt_z,
        mode="lines",
        line=dict(width=3, color=C_SQUARE_EDGE),
        name="Transformed square edges",
        showlegend=True,
        legendrank=60,
        hoverinfo="skip",
    ))

    # Arrows
    if show_arrows:
        xs, ys, zs = [], [], []
        for p0, p1 in zip(pts3_orig, points_3d_t):
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=2, color=C_ARROW),
            name="Arrows",
            opacity=0.45,
            showlegend=True,
            legendrank=70,
            hoverinfo="skip",
        ))

    # ---------- Rotation axis guides (same idea as app2x3.py) ----------
    # Stage 1–2: 2D rotation is about z-axis. Stage 3: 3D rotation about axis of U.
    axis_V3 = np.array([0.0, 0.0, 1.0])
    axis_U, _theta_U = axis_angle_from_rot(U)

    def add_axis_line(axis_vec, length, opacity, color, name, legendrank):
        if opacity <= 0:
            return
        a = np.asarray(axis_vec, dtype=float)
        an = np.linalg.norm(a)
        if an < 1e-12:
            return
        a = a / an
        p0 = -length * a
        p1 = length * a
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines",
            line=dict(width=3, color=color, dash="dash"),
            opacity=float(opacity),
            showlegend=True,
            hoverinfo="skip",
            name=name,
            legendrank=legendrank,
        ))

    axis_len = 0.95 * half
    base_op = 0.85

    # Fade u_V out from t=1.5 to t=2.0
    # Before 1.5: fully visible
    # At 2.0: reduced to 0.20
    if t <= 1.5:
        op_V = base_op
    elif t <= 2.0:
        w = (t - 1.5) / 0.5  # 0 -> 1
        op_V = base_op * (1.0 - w) + 0.20 * w
    else:
        op_V = 0.20
    
    # Keep u_U hidden until Stage 3 starts, then show fully
    op_U = base_op if t > 2.0 else 0.0

    # base_op = 0.85
    # if t <= 2.0:
    #     op_V = base_op
    #     op_U = 0.0
    # else:
    #     op_V = 0.20
    #     op_U = base_op

    add_axis_line(axis_V3, axis_len, op_V, color="#8A2BE2", name="Rotation axis u_V (about z)", legendrank=75)
    add_axis_line(axis_U,  axis_len, op_U, color="#00B3B3", name="Rotation axis u_U", legendrank=80)

    fig.update_layout(
        uirevision=uirevision_key,
        scene_camera=camera,
        scene=dict(
            xaxis=dict(range=x_range, title="x", showbackground=False, showgrid=True, zeroline=False),
            yaxis=dict(range=y_range, title="y", showbackground=False, showgrid=True, zeroline=False),
            zaxis=dict(range=z_range, title="z", showbackground=False, showgrid=True, zeroline=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98),
        title="3D view: 3×2 SVD path (app2x3 style + rotation axes)",
        height=800,
    )

    return fig


# ---------- Matplotlib 2D figure ----------

def make_2d_figure(points_orig, points_plane_t, square_2d, square_plane_t):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points_orig[:, 0], points_orig[:, 1], s=30, label="Original 2D points")
    ax.scatter(points_plane_t[:, 0], points_plane_t[:, 1], s=40, marker="x",
               label="In-plane coords along SVD path")

    sq = np.vstack([square_2d, square_2d[0]])
    ax.fill(sq[:, 0], sq[:, 1], facecolor=(1, 0, 0, 0.06),
            edgecolor="black", linewidth=1.0, label="Original square")

    sq_t = np.vstack([square_plane_t, square_plane_t[0]])
    ax.fill(sq_t[:, 0], sq_t[:, 1], facecolor=(1, 0, 0, 0.0),
            edgecolor="red", linewidth=2, linestyle="--",
            label="Transformed square in plane coords")

    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)

    all_pts = np.vstack([points_orig, points_plane_t, square_2d, square_plane_t])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.5 * span * 1.1 + 1e-9

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D view: input plane R² (square tracks rotate + stretch)",
                 pad=90
                 )

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.35),  # place above axes, left-aligned
        ncol=1,                      # vertical list
        fontsize=9,
        frameon=False,               # remove boundary box
        labelspacing=0.35,
        handlelength=1.4,
        handletextpad=0.5,
        borderaxespad=0.0,
    )


    
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


# ---------- GIF generation ----------

def create_animation_gif_3x2(filename,
                             points_2d,
                             square_2d,
                             U,
                             s,
                             V,
                             n_frames=120,
                             fps=15,
                             pause_seconds=1.0,
                             show_arrows=True):
    pts3_orig = np.hstack([points_2d, np.zeros((points_2d.shape[0], 1))])
    square3_orig = np.hstack([square_2d, np.zeros((square_2d.shape[0], 1))])

    pts3_final = svd_positions_3x2(points_2d, 3.0, U, s, V)
    square3_final = svd_positions_3x2(square_2d, 3.0, U, s, V)

    all_pts = np.vstack([pts3_orig, pts3_final, square3_orig, square3_final])
    x_range, y_range, z_range, center, half = compute_bounds_3d([all_pts], margin_factor=1.1)

    pause_frames = int(fps * pause_seconds)
    total_frames = n_frames + pause_frames

    frames = []

    for frame_idx in range(total_frames):
        if frame_idx < n_frames:
            t = 3.0 * frame_idx / (n_frames - 1)
        else:
            t = 3.0

        pts3_t = svd_positions_3x2(points_2d, t, U, s, V)
        square3_t = svd_positions_3x2(square_2d, t, U, s, V)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(pts3_orig[:, 0], pts3_orig[:, 1], pts3_orig[:, 2],
                   s=8, alpha=0.9, label="Original 2D (z=0)")
        ax.scatter(pts3_t[:, 0], pts3_t[:, 1], pts3_t[:, 2],
                   s=12, marker="x", alpha=0.9, label="3D image")

        faces_orig = [[square3_orig[i] for i in [0, 1, 2, 3]]]
        poly_orig = Poly3DCollection(
            faces_orig,
            facecolors=(1, 0, 0, 0.06),
            edgecolors="black",
            linewidths=1
        )
        ax.add_collection3d(poly_orig)

        faces_t = [[square3_t[i] for i in [0, 1, 2, 3]]]
        poly_t = Poly3DCollection(
            faces_t,
            facecolors=(0.31, 0.55, 1.0, 0.06),
            edgecolors="red",
            linewidths=1
        )
        ax.add_collection3d(poly_t)

        if show_arrows:
            for p0, p1 in zip(pts3_orig, pts3_t):
                ax.plot([p0[0], p1[0]],
                        [p0[1], p1[1]],
                        [p0[2], p1[2]],
                        color="gray", linewidth=0.8, alpha=0.4)

        ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_zlim(z_range)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=30, azim=45)

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"3×2 SVD path (t = {t:.2f}) with square")
        ax.legend(loc="upper left")

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        if img.shape[2] == 4:
            img = img[:, :, :3]
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(filename, frames, fps=fps, loop=0)


# ---------- Streamlit app ----------

def main():
    st.set_page_config(page_title="3×2 Matrix: R² → R³ via SVD (cartoon + GIF)", layout="wide")
    st.title("3×2 Linear Transformation: Lifting R² into R³ via SVD (Cartoon + GIF)")

    # Camera persistence state
    if "plotly_camera_3x2" not in st.session_state:
        st.session_state.plotly_camera_3x2 = DEFAULT_CAMERA_3X2.copy()
    if "uirevision_key_3x2" not in st.session_state:
        st.session_state.uirevision_key_3x2 = "keep_camera_3x2_v1"

    st.sidebar.header("Random 2D points")
    seed = st.sidebar.slider("Random seed", 0, 100, 0, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("3×2 transformation matrix A")

    mode = st.sidebar.radio("Define A by:", ["Preset: simple embedding", "Manual 3×2 entries"])

    if mode == "Preset: simple embedding":
        A = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, 0.0]])
    else:
        c1, c2 = st.sidebar.columns(2)
        with c1:
            a11 = st.number_input("a11", value=1.0, step=0.1, key="a11_3x2")
            a21 = st.number_input("a21", value=0.0, step=0.1, key="a21_3x2")
            a31 = st.number_input("a31", value=-1.0, step=0.1, key="a31_3x2")
        with c2:
            a12 = st.number_input("a12", value=2.0, step=0.1, key="a12_3x2")
            a22 = st.number_input("a22", value=1.0, step=0.1, key="a22_3x2")
            a32 = st.number_input("a32", value=0.0, step=0.1, key="a32_3x2")
        A = np.array([[a11, a12],
                      [a21, a22],
                      [a31, a32]])

    show_arrows = st.sidebar.checkbox("Show arrows from 2D to 3D image", value=True)

    st.sidebar.markdown("---")
    t = st.sidebar.slider(
        "SVD path slider (0 → 3)",
        min_value=0.0, max_value=3.0, value=3.0, step=0.01,
        help="Stage 1 (0–1): rotate by V. Stage 2 (1–2): stretch by Σ. Stage 3 (2–3): rotate in 3D by U."
    )

    points_2d = generate_random_cluster_2d(n_points=30, seed=seed)
    square_2d = build_outer_square_2d(points_2d)

    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T

    points_3d_t = svd_positions_3x2(points_2d, t, U, s, V)
    square_3d_t = svd_positions_3x2(square_2d, t, U, s, V)
    plane_coords_t = plane_coords_3x2(points_2d, t, s, V)
    square_plane_t = plane_coords_3x2(square_2d, t, s, V)

    st.subheader("Current 3×2 matrix A")
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        %.3f & %.3f \\
        %.3f & %.3f \\
        %.3f & %.3f
        \end{bmatrix}
        """
        % (A[0, 0], A[0, 1],
           A[1, 0], A[1, 1],
           A[2, 0], A[2, 1])
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("3D view (drag to rotate, view persists)")
        fig3d = make_3d_figure(
            points_2d=points_2d,
            points_3d_t=points_3d_t,
            square_2d=square_2d,
            square_3d_t=square_3d_t,
            t=t,
            U=U,
            V=V,
            show_arrows=show_arrows,
            camera=st.session_state.plotly_camera_3x2,
            uirevision_key=st.session_state.uirevision_key_3x2,
        )

        events = plotly_events(
            fig3d,
            click_event=True,
            select_event=False,
            hover_event=True,
            override_height=800,
            key="plotly_events_3x2_main",
        )
        update_camera_from_events_3x2(events, state_key="plotly_camera_3x2")

    with col2:
        # push the right panel down so it visually aligns with the 3D plot area
        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)
        
        st.subheader("2D view: input plane R²")
        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)
        
        fig2d = make_2d_figure(points_2d, plane_coords_t, square_2d, square_plane_t)
        st.pyplot(fig2d)

    st.markdown("---")
    st.subheader("GIF animation from the 3×2 SVD path (with square)")

    if st.button("Generate GIF animation (svd3x2_animation.gif)"):
        with st.spinner("Generating 3×2 SVD GIF animation..."):
            try:
                create_animation_gif_3x2(
                    filename="svd3x2_animation.gif",
                    points_2d=points_2d,
                    square_2d=square_2d,
                    U=U,
                    s=s,
                    V=V,
                    n_frames=120,
                    fps=15,
                    pause_seconds=1.0,
                    show_arrows=show_arrows,
                )
                st.success("Animation saved as svd3x2_animation.gif")
            except Exception as e:
                st.error(f"Failed to create animation. Error: {e}")

    if os.path.exists("svd3x2_animation.gif"):
        st.image("svd3x2_animation.gif")


if __name__ == "__main__":
    main()
