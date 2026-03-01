"""
2x3 linear transformation demo: R^3 -> R^2 via SVD with 3-stage cartoon

How to run locally:

    cd D:\\Visualizer2025
    streamlit run app2x3.py

requirements.txt needs at least:
    streamlit
    numpy
    matplotlib
    plotly
    pillow
    imageio
    streamlit-plotly-events
"""

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # for animated GIFs

from streamlit_plotly_events import plotly_events


# ---------- Core helpers ----------

def generate_random_cluster_3d(n_points=30, mean=(0.0, 0.0, 0.0), cov=None, seed=0):
    """
    Generate a random 3D cluster of points.
    """
    if cov is None:
        cov = np.array([
            [1.0, 0.4, 0.2],
            [0.4, 1.2, 0.3],
            [0.2, 0.3, 0.8],
        ])
    rng = np.random.default_rng(seed)
    pts = rng.multivariate_normal(mean, cov, size=n_points)
    return pts  # (n, 3)


def build_outer_cube(points):
    """
    Build a cube that surrounds the given 3D points.
    """
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)

    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span_x, span_y, span_z = xmax - xmin, ymax - ymin, zmax - zmin
    half_side = 0.5 * max(span_x, span_y, span_z) * 1.3 + 1e-6

    corners = np.array([
        [cx - half_side, cy - half_side, cz - half_side],  # 0
        [cx + half_side, cy - half_side, cz - half_side],  # 1
        [cx + half_side, cy + half_side, cz - half_side],  # 2
        [cx - half_side, cy + half_side, cz - half_side],  # 3
        [cx - half_side, cy - half_side, cz + half_side],  # 4
        [cx + half_side, cy - half_side, cz + half_side],  # 5
        [cx + half_side, cy + half_side, cz + half_side],  # 6
        [cx - half_side, cy + half_side, cz + half_side],  # 7
    ])

    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return corners, edge_pairs


def build_edge_lines(corners, edge_pairs):
    """
    Build x, y, z arrays for line segments for cube edges.
    """
    xs, ys, zs = [], [], []
    for i, j in edge_pairs:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])
    return xs, ys, zs





def compute_bounds_3d(points_list, margin_factor=1.05):
    """
    Given a list of 3D point arrays, compute global bounds and a cubic range.
    """
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
    """
    Extract rotation angle (radians) from a 2x2 rotation matrix R.
    """
    return np.arctan2(R[1, 0], R[0, 0])


def rotation_matrix_2d(theta):
    """
    2D rotation matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def rotation_matrix_axis_angle(axis, angle):
    """
    Rodrigues' formula for 3D rotation matrix from unit axis and angle (radians).
    """
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
    """
    Extract (axis, angle) from a 3x3 rotation-like matrix R.
    Assumes det(R) ~ 1 (a proper rotation).
    """
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


# ---------- Mesh primitives for robust point rendering (same as app3d style) ----------

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


# ---------- Camera persistence via plotly_events (same pattern as app3d) ----------

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
                base = st.session_state.get("plotly_camera_2x3", DEFAULT_CAMERA)
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
            st.session_state.plotly_camera_2x3 = cam


# ---------- SVD cartoon positions (pointwise, as discussed) ----------

def svd_positions(points_3d, t, U, s, V):
    """
    Robust 2×3 SVD cartoon that keeps the image plane span(v1, v2) FIXED in world coords.

    Stage 1 (0–1): rigidly rotate the cloud from I to V (axis–angle interpolation).
    Stage 2 (1–2): decompose into the orthonormal basis (v1, v2, v3), scale along v1/v2
                  by sigma_1, sigma_2 (signed allowed), and squash the v3 component to 0.
    Stage 3 (2–3): rotate ONLY the (v1, v2) coefficients (in-plane) according to U.

    This avoids the visual artifact where the whole plane appears to tilt in Step 3.
    """
    s1, s2 = float(s[0]), float(s[1])

    # Orthonormal plane basis
    v1 = V[:, 0].astype(float)
    v2 = V[:, 1].astype(float)
    v3 = V[:, 2].astype(float)

    v1 /= (np.linalg.norm(v1) + 1e-12)
    v2 /= (np.linalg.norm(v2) + 1e-12)
    v3 /= (np.linalg.norm(v3) + 1e-12)

    # ---- Stage 1: rotate rigidly from I to V ----
    if t <= 1.0:
        axis_V, theta_V = axis_angle_from_rot(V)
        Rv_t = rotation_matrix_axis_angle(axis_V, t * theta_V)
        return points_3d @ Rv_t

    # Stage 1 endpoint (the starting state for later stages)
    p1 = points_3d @ V  # world coordinates after full Step 1 rotation

    # Express p1 in the (v1, v2, v3) basis via dot products
    a = p1 @ v1  # (n,)
    b = p1 @ v2
    c = p1 @ v3

    # ---- Stage 2: scale within plane, squash normal component ----
    if t <= 2.0:
        beta = t - 1.0  # in [0, 1]
        s1_t = 1.0 + beta * (s1 - 1.0)
        s2_t = 1.0 + beta * (s2 - 1.0)
        z_t = 1.0 - beta  # c -> 0 smoothly

        a2 = s1_t * a
        b2 = s2_t * b
        c2 = z_t * c

        return (a2[:, None] * v1[None, :]
                + b2[:, None] * v2[None, :]
                + c2[:, None] * v3[None, :])

    # ---- Stage 3: rotate within the plane (v1, v2), keep c = 0 ----
    gamma = t - 2.0  # in [0, 1]

    theta_U = angle_from_rot2d(U)
    R2d_t = rotation_matrix_2d(gamma * theta_U)

    # Start from the end of Stage 2: (a, b) scaled by (s1, s2), c = 0
    ab = np.column_stack([s1 * a, s2 * b])  # (n, 2)
    ab_rot = ab @ R2d_t

    a3 = ab_rot[:, 0]
    b3 = ab_rot[:, 1]
    c3 = np.zeros_like(a3)

    return (a3[:, None] * v1[None, :]
            + b3[:, None] * v2[None, :]
            + c3[:, None] * v3[None, :])





# ---------- Plotly 3D figure (upgraded to app3d style + camera persistence) ----------

def make_3d_figure(points_3d,
                   points_3d_t,
                   cube_corners,
                   cube_corners_t,
                   cube_edge_pairs,
                   show_arrows,
                   V,
                   t,
                   x_range,
                   y_range,
                   z_range,
                   half,
                   camera,
                   uirevision_key):
    """
    Build an interactive 3D Plotly figure showing:
    - Original 3D points and cube
    - Their images along the SVD path
    - A translucent plane span(V₁, V₂) drawn explicitly
    - Rotation axis guides shown in legend:
        u_V (axis of V rotation)
        u_plane (v3, normal axis to the image plane)
    """
    fig = go.Figure()
    fig.update_layout(template="plotly_white", showlegend=True)

    # Color scheme (same style as app3d)
    C_ORIG_PTS = "#0b3d91"    # deep blue
    C_TRANS_PTS = "#ff7f0e"   # orange
    C_CUBE_ORIG = "#444444"   # gray
    C_CUBE_TRANS = "#ff2b2b"  # red
    C_DISP = "#6f93a6"        # darker blue-gray

    # ---- Original points: Mesh3d spheres ----
    base_sphere = make_unit_sphere_mesh(n_theta=10, n_phi=18)
    scene_span = (x_range[1] - x_range[0])
    sphere_radius = 0.004 * scene_span
    sx, sy, sz, si, sj, sk = pack_spheres(points_3d, sphere_radius, base_sphere)

    fig.add_trace(go.Mesh3d(
        x=sx, y=sy, z=sz,
        i=si, j=sj, k=sk,
        color=C_ORIG_PTS,
        opacity=1.0,
        name="Original 3D points",
        showlegend=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.2, roughness=0.8),
    ))

    # Legend-only marker (blue circle)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=9, symbol="circle", color=C_ORIG_PTS, opacity=1.0),
        name="Original 3D points",
        visible="legendonly",
        showlegend=True,
        legendrank=10
    ))

    # ---- Transformed points: 3D crosses using line segments ----
    cross_half_len = 0.009 * scene_span
    cx, cy, cz = build_3d_cross_segments(points_3d_t, cross_half_len)

    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="lines",
        line=dict(width=4, color=C_TRANS_PTS),
        name="SVD-path image",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Legend-only marker (orange +)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=10, symbol="cross", color=C_TRANS_PTS, line=dict(width=0.2, color=C_TRANS_PTS)),
        name="SVD-path image",
        visible="legendonly",
        showlegend=True,
        legendrank=20
    ))

    
    # ---- Original cube edges ----
    ex, ey, ez = build_edge_lines(cube_corners, cube_edge_pairs)
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(width=2, color=C_CUBE_ORIG),
        name="Original cube",
        showlegend=True,
        legendrank=30,
        hoverinfo="skip",
    ))

    # ---- Transformed cube edges ----
    ex2, ey2, ez2 = build_edge_lines(cube_corners_t, cube_edge_pairs)
    fig.add_trace(go.Scatter3d(
        x=ex2, y=ey2, z=ez2,
        mode="lines",
        line=dict(width=3, color=C_CUBE_TRANS),
        name="Cube along SVD path",
        showlegend=True,
        legendrank=40,
        hoverinfo="skip",
    ))
# ---- Optional arrows from 3D points to their images ----
    if show_arrows:
        xs, ys, zs = [], [], []
        for p, q in zip(points_3d, points_3d_t):
            xs.extend([p[0], q[0], None])
            ys.extend([p[1], q[1], None])
            zs.extend([p[2], q[2], None])
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(width=2, color=C_DISP),
            name="Arrows to SVD image",
            opacity=0.45,
            showlegend=True,
            legendrank=50
        ))

    # ---- Semi-transparent image plane span(V1, V2) (kept from original) ----
    if V is not None:
        # ---- Image plane span(V1, V2) ----
        # For visibility, draw the plane through the current scene center (not necessarily the origin),
        # so it stays inside the fixed bounds even when the cloud is not centered at (0,0,0).
        v1 = V[:, 0].astype(float)
        v2 = V[:, 1].astype(float)
        v1 /= (np.linalg.norm(v1) + 1e-12)
        v2 /= (np.linalg.norm(v2) + 1e-12)

        center = np.array([
            0.5 * (x_range[0] + x_range[1]),
            0.5 * (y_range[0] + y_range[1]),
            0.5 * (z_range[0] + z_range[1]),
        ], dtype=float)

        n_grid = 14
        u_vals = np.linspace(-half, half, n_grid)
        v_vals = np.linspace(-half, half, n_grid)
        U_grid, V_grid = np.meshgrid(u_vals, v_vals)

        X_plane = np.zeros_like(U_grid)
        Y_plane = np.zeros_like(U_grid)
        Z_plane = np.zeros_like(U_grid)

        for i in range(n_grid):
            for j in range(n_grid):
                p = center + U_grid[i, j] * v1 + V_grid[i, j] * v2
                X_plane[i, j] = p[0]
                Y_plane[i, j] = p[1]
                Z_plane[i, j] = p[2]

        # A semi-transparent plane (make it clearly visible; Plotly surfaces can look too faint
        # depending on theme and depth shading).
        # --- Image plane span(V1, V2) : Mesh3d version (reliable visibility) ---
        plane_scale = 1.1 * half
        
        p00 = center + (-plane_scale)*v1 + (-plane_scale)*v2
        p01 = center + (-plane_scale)*v1 + ( plane_scale)*v2
        p10 = center + ( plane_scale)*v1 + (-plane_scale)*v2
        p11 = center + ( plane_scale)*v1 + ( plane_scale)*v2
        
        X = [p00[0], p01[0], p11[0], p10[0]]
        Y = [p00[1], p01[1], p11[1], p10[1]]
        Z = [p00[2], p01[2], p11[2], p10[2]]
        
        fig.add_trace(go.Mesh3d(
            x=X,
            y=Y,
            z=Z,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="rgba(80, 140, 255, 0.45)",
            opacity=0.10,
            name="Image plane span(V₁, V₂)",
            showlegend=True,
            hoverinfo="skip",
        ))


        # Thin border so the plane is still visible even when nearly edge-on
        c1 = center + (-half) * v1 + (-half) * v2
        c2 = center + ( half) * v1 + (-half) * v2
        c3 = center + ( half) * v1 + ( half) * v2
        c4 = center + (-half) * v1 + ( half) * v2
        bx = [c1[0], c2[0], c3[0], c4[0], c1[0], None]
        by = [c1[1], c2[1], c3[1], c4[1], c1[1], None]
        bz = [c1[2], c2[2], c3[2], c4[2], c1[2], None]
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="lines",
            line=dict(width=2, color="#8aa4d6"),
            name="Image plane boundary",
            showlegend=False,
            hoverinfo="skip",
        ))

    # ---- Rotation axis guides (shown in legend) ----
    # u_V: axis-angle axis of V
    axis_V, theta_V = axis_angle_from_rot(V)
    # u_plane: normal axis to image plane = v3 in old coords
    axis_plane = V[:, 2].copy()
    n = np.linalg.norm(axis_plane)
    if n > 1e-12:
        axis_plane = axis_plane / n

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

    base_op = 0.85
    if t <= 1.0:
        op_V = base_op
        op_plane = 0.0
    elif t <= 2.0:
        op_V = base_op * (2.0 - t)   # fade out during Stage 2
        op_plane = 0.0
    else:
        op_V = 0.0
        op_plane = base_op

    axis_len = 0.95 * half
    add_axis_line(axis_V, axis_len, op_V, color="#8A2BE2", name="Rotation axis u_V", legendrank=60)
    add_axis_line(axis_plane, axis_len, op_plane, color="#00B3B3", name="Plane axis u_plane (v₃)", legendrank=70)

    # Invisible hover targets (so hovering is stable even with mesh/cross)
    fig.add_trace(go.Scatter3d(
        x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.0),
        showlegend=False,
        hovertemplate="Original<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="__hover_orig__",
    ))
    fig.add_trace(go.Scatter3d(
        x=points_3d_t[:, 0], y=points_3d_t[:, 1], z=points_3d_t[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.0),
        showlegend=False,
        hovertemplate="SVD image<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="__hover_trans__",
    ))

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
        title="3D view: SVD path in tilted image plane",
        height=800,
    )
    return fig


# ---------- Matplotlib 2D figure ----------

def make_2d_figure(points_2d):
    """
    Simple 2D scatter of the plane coordinates (x', y').
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=40)

    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)

    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.5 * span * 1.1 + 1e-9

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x'")
    ax.set_ylabel("y'")
    ax.set_title("2D view in plane coordinates (x', y')")

    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------- GIF generation (using imageio, bigger cartoon) ----------

def create_animation_gif_2x3(filename,
                             points_3d,
                             cube_corners,
                             cube_edge_pairs,
                             U,
                             s,
                             V,
                             n_frames=120,
                             fps=15,
                             pause_seconds=1.0,
                             show_arrows=True):
    """
    Generate an animated GIF following the 3-stage SVD path t in [0, 3].
    Uses imageio for robust GIF animation.
    """
    points_final = svd_positions(points_3d, 3.0, U, s, V)

    all_pts = np.vstack([points_3d, points_final])
    x_range, y_range, z_range, center, half = compute_bounds_3d(
        [all_pts],
        margin_factor=1.1
    )

    pause_frames = int(fps * pause_seconds)
    total_frames = n_frames + pause_frames

    frames = []

    for frame_idx in range(total_frames):
        if frame_idx < n_frames:
            t = 3.0 * frame_idx / (n_frames - 1)
        else:
            t = 3.0

        pts_M = svd_positions(points_3d, t, U, s, V)
        cube_M = svd_positions(cube_corners, t, U, s, V)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                   s=8, alpha=0.9, label="Original points")

        ax.scatter(pts_M[:, 0], pts_M[:, 1], pts_M[:, 2],
                   s=12, marker="x", alpha=0.9, label="SVD path image")

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
            for p, q in zip(points_3d, pts_M):
                ax.plot([p[0], q[0]],
                        [p[1], q[1]],
                        [p[2], q[2]],
                        color="gray", linewidth=0.8, alpha=0.4)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_box_aspect((1, 1, 1))

        ax.view_init(elev=30, azim=45)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_title(f"2×3 SVD path (t = {t:.2f})")
        ax.legend(loc="upper left")

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        if img.shape[2] == 4:
            img = img[:, :, :3]
        frames.append(img)

        plt.close(fig)

    imageio.mimsave(
        filename,
        frames,
        fps=fps,
        loop=0
    )


# ---------- Streamlit app ----------

def main():
    st.set_page_config(
        page_title="2×3 Matrix: Project R³ → R² via SVD (cartoon + GIF)",
        layout="wide"
    )

    # Camera persistence state (separate key from app3d)
    if "plotly_camera_2x3" not in st.session_state:
        st.session_state.plotly_camera_2x3 = DEFAULT_CAMERA.copy()
    if "uirevision_key_2x3" not in st.session_state:
        st.session_state.uirevision_key_2x3 = "keep_camera_2x3_v1"

    st.title("2×3 Linear Transformation: Projecting 3D to 2D via SVD (Cartoon + GIF)")

    st.write(
        r"""
This app shows how a **2×3 matrix** maps points in **$\mathbb R^3$** into **$\mathbb R^2$**.

We factor
$$
A = U \Sigma V^T
$$
and think in row-vector form:
$$
y = x A^T = x V \Sigma^T U^T.
$$

Geometric idea in a local 3D coordinate system $(x',y',z')$ given by $V$:

1. Use $V$ to define a *tilted image plane* spanned by its first two columns (the local $x'y'$-plane).
2. In $(x',y',z')$, $\Sigma^T$ stretches along $x'$ and $y'$ by $\sigma_1,\sigma_2$
   and squashes the $z'$ direction to 0.
3. Still in $(x',y',z')$, $U^T$ rotates coordinates inside that plane.

For drawing, we always convert back from new coords $(x',y',z')$ to the original
$(x,y,z)$ via $V^T$.
"""
    )

    # Sidebar: random seed
    st.sidebar.header("Random 3D points")
    seed = st.sidebar.slider("Random seed", 0, 100, 0, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("2×3 transformation matrix A")

    mode = st.sidebar.radio(
        "Define A by:",
        ["Preset: projection to xy-plane", "Manual 2×3 entries"],
        index=1
    )

    if mode == "Preset: projection to xy-plane":
        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        st.sidebar.write("Using preset projection:")
        st.sidebar.latex(
            r"""
            A =
            \begin{bmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0
            \end{bmatrix}
            """
        )
    else:
        st.sidebar.write("Enter entries for the 2×3 matrix A:")
        c1, c2, c3 = st.sidebar.columns(3)
        with c1:
            # Default matrix requested: [[1, 2, 0], [0, 1, -1]]
            a11 = st.number_input("a11", value=1.0, step=0.1, key="a11")
            a21 = st.number_input("a21", value=0.0, step=0.1, key="a21")
        with c2:
            a12 = st.number_input("a12", value=2.0, step=0.1, key="a12")
            a22 = st.number_input("a22", value=1.0, step=0.1, key="a22")
        with c3:
            a13 = st.number_input("a13", value=0.0, step=0.1, key="a13")
            a23 = st.number_input("a23", value=-1.0, step=0.1, key="a23")

        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
        ])

    show_arrows = st.sidebar.checkbox(
        "Show arrows from original 3D points to SVD images",
        value=True,
    )

    st.sidebar.markdown("---")
    t = st.sidebar.slider(
        "SVD path slider (0 → 3)",
        min_value=0.0,
        max_value=3.0,
        value=3.0,
        step=0.01,
        help=(
            "Stage 1 (0–1): rotate 3D by V; "
            "Stage 2 (1–2): stretch/compress in (x′,y′,z′); "
            "Stage 3 (2–3): rotate inside the x′y′ plane by U."
        )
    )

    # Random data and cube
    points_3d = generate_random_cluster_3d(n_points=30, seed=seed)
    cube_corners, cube_edges = build_outer_cube(points_3d)

    # ---------- SVD of a 2×3 matrix ----------
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T

    # Make U a proper rotation for the Step 3 cartoon
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1          # flip one column of U
        s[1] *= -1             # carry sign into sigma2 (signed scaling)
    # Axis–angle for V (3×3)
    axis_V, theta_V = axis_angle_from_rot(V)
    theta_V_deg = float(np.rad2deg(theta_V))

    # Rotation angle for U (2×2)
    theta_U = angle_from_rot2d(U)
    theta_U_deg = float(np.rad2deg(theta_U))

    # Positions at current t
    points_3d_t = svd_positions(points_3d, t, U, s, V)
    cube_3d_t = svd_positions(cube_corners, t, U, s, V)

    # For the 2D view: plane coordinates (x′, y′) in the fixed basis (v1, v2)
    v1 = V[:, 0] / (np.linalg.norm(V[:, 0]) + 1e-12)
    v2 = V[:, 1] / (np.linalg.norm(V[:, 1]) + 1e-12)
    coords_2d_t = np.column_stack([points_3d_t @ v1, points_3d_t @ v2])

    # ---------- Stable bounds across the whole SVD path (0,1,2,3) ----------
    pts_t0 = svd_positions(points_3d, 0.0, U, s, V)
    pts_t1 = svd_positions(points_3d, 1.0, U, s, V)
    pts_t2 = svd_positions(points_3d, 2.0, U, s, V)
    pts_t3 = svd_positions(points_3d, 3.0, U, s, V)

    cube_t0 = svd_positions(cube_corners, 0.0, U, s, V)
    cube_t1 = svd_positions(cube_corners, 1.0, U, s, V)
    cube_t2 = svd_positions(cube_corners, 2.0, U, s, V)
    cube_t3 = svd_positions(cube_corners, 3.0, U, s, V)

    # First pass bounds for a robust "half"
    x_range0, y_range0, z_range0, center0, half0 = compute_bounds_3d(
        [pts_t0, pts_t1, pts_t2, pts_t3, cube_t0, cube_t1, cube_t2, cube_t3],
        margin_factor=1.20
    )

    # Include plane extents too (span(V1,V2))
    v1 = V[:, 0]
    v2 = V[:, 1]
    plane_corners = np.array([
        (-half0 * v1) + (-half0 * v2),
        (-half0 * v1) + ( half0 * v2),
        ( half0 * v1) + ( half0 * v2),
        ( half0 * v1) + (-half0 * v2),
    ])

    x_range, y_range, z_range, center, half = compute_bounds_3d(
        [pts_t0, pts_t1, pts_t2, pts_t3, cube_t0, cube_t1, cube_t2, cube_t3, plane_corners],
        margin_factor=1.30
    )

    # ---------- Show A ----------
    st.subheader("Current 2×3 matrix A")
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        %.3f & %.3f & %.3f \\
        %.3f & %.3f & %.3f
        \end{bmatrix}
        """
        % (
            A[0, 0], A[0, 1], A[0, 2],
            A[1, 0], A[1, 1], A[1, 2],
        )
    )

    # Stage description (kept)
    if t <= 1.0:
        stage_text = (
            "Stage 1: 3D rotation from the original orientation to the new "
            "axes defined by V (rigid rotation of the whole cloud and cube)."
        )
    elif t <= 2.0:
        stage_text = (
            "Stage 2: In the new coordinates (x′,y′,z′), stretch along x′ and y′ "
            "and compress z′, then convert back via Vᵀ."
        )
    else:
        stage_text = (
            "Stage 3: In the new coordinates (x′,y′,z′), with z′ already near 0, "
            "rotate inside the x′y′-plane according to U, then convert back via Vᵀ."
        )

    st.markdown(f"**Current SVD stage (t = {t:.2f})** – {stage_text}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("3D view: original points and SVD path (drag to rotate)")

        fig3d = make_3d_figure(
            points_3d=points_3d,
            points_3d_t=points_3d_t,
            cube_corners=cube_corners,
            cube_corners_t=cube_3d_t,
            cube_edge_pairs=cube_edges,
            show_arrows=show_arrows,
            V=V,
            t=t,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            half=half,
            camera=st.session_state.plotly_camera_2x3,
            uirevision_key=st.session_state.uirevision_key_2x3,
        )

        events = plotly_events(
            fig3d,
            # Keep click enabled so the next interaction after a camera drag can trigger a rerun.
            click_event=True,
            select_event=False,
            hover_event=True,
            override_height=800,
            # Use the refresh counter in the key to force the component to remount on refresh.
            key="plotly_events_2x3",
        )
        update_camera_from_events(events)

        st.caption(
            "The translucent sheet is the image plane span(V₁, V₂): "
            "the local x′y′-plane determined by the 3D rotation V. "
            "In Stage 2 we scale in (x′,y′,z′), in Stage 3 we rotate in the x′y′ "
            "plane, always converting back to (x,y,z) for plotting."
        )

    with col2:
        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)
        st.subheader("2D view: plane coordinates (x′, y′)")
        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)
        fig2d = make_2d_figure(coords_2d_t)
        st.pyplot(fig2d)
        st.caption(
            "Each dot shows the transformed point expressed in plane coordinates "
            "(x′, y′) with respect to the basis (V₁, V₂). "
            "These are the same points as in the 3D view, but seen in the "
            "intrinsic image-plane coordinates."
        )

    st.markdown("---")
    st.subheader("SVD interpretation of a 2×3 map")

    st.write(
        r"""
For a **2×3** matrix $A$, a singular value decomposition is

$$
A = U \Sigma V^T,
$$

where:

- $U$ is $2\times 2$ orthogonal (rotation/reflection in the output plane),
- $V$ is $3\times 3$ orthogonal (rotation/reflection in 3D),
- $\Sigma$ is $2\times 3$ with nonnegative singular values $\sigma_1,\sigma_2$ on the diagonal:

$$
\Sigma =
\begin{bmatrix}
\sigma_1 & 0 & 0 \\
0 & \sigma_2 & 0
\end{bmatrix}.
$$

In the row-vector viewpoint:
$$
y = x A^T = x V \Sigma^T U^T.
$$

We interpret:

1. $V$ defines a **local 3D coordinate system** $(x',y',z')$; its first two columns span the
   image plane (our tilted output plane).
2. $\Sigma^T$ stretches along $x'$ and $y'$ by $\sigma_1,\sigma_2$ and squashes $z'$ to 0.
3. $U^T$ rotates the resulting 2D coordinates inside that plane.

For the cartoon, Stage 1 shows the rotation, Stage 2 the stretch/compression in $(x',y',z')$,
and Stage 3 the in-plane rotation in $(x',y',0)$, always mapped back by $V^T$.
"""
    )

    st.latex(
        r"""
        U \approx
        \begin{bmatrix}
        %.3f & %.3f \\
        %.3f & %.3f
        \end{bmatrix}
        """
        % (U[0, 0], U[0, 1], U[1, 0], U[1, 1])
    )

    st.latex(
        r"""
        \Sigma \approx
        \begin{bmatrix}
        %.3f & 0 & 0 \\
        0 & %.3f & 0
        \end{bmatrix}
        """
        % (s[0], s[1])
    )

    st.latex(
        r"""
        V^T \approx
        \begin{bmatrix}
        %.3f & %.3f & %.3f \\
        %.3f & %.3f & %.3f \\
        %.3f & %.3f & %.3f
        \end{bmatrix}
        """
        % (
            Vt[0, 0], Vt[0, 1], Vt[0, 2],
            Vt[1, 0], Vt[1, 1], Vt[1, 2],
            Vt[2, 0], Vt[2, 1], Vt[2, 2],
        )
    )

    st.markdown(
        r"""
### Rotation axis and angle of $V$

We can interpret $V$ (the 3D orthogonal factor) in axis–angle form:

$$
R_V = V \approx R(\mathbf u_V,\,\theta_V),
$$

where $\mathbf u_V$ is a unit axis in $\mathbb R^3$ and $\theta_V$ is a rotation angle.
"""
    )

    st.latex(
        r"""
        \mathbf u_V \approx
        \begin{bmatrix}
        %.3f \\ %.3f \\ %.3f
        \end{bmatrix},\quad
        \theta_V \approx %.1f^\circ
        """
        % (axis_V[0], axis_V[1], axis_V[2], theta_V_deg)
    )

    st.markdown(
        r"""
### Approximate rotation angle of $U$

If $U$ is close to a pure 2D rotation, its first column is approximately
$[\cos\theta_U,\ \sin\theta_U]^T$.
Here:

$$
\theta_U \approx %.1f^\circ,\quad
\cos\theta_U \approx U_{11} = %.3f,\quad
\sin\theta_U \approx U_{21} = %.3f.
$$
        """
        % (theta_U_deg, U[0, 0], U[1, 0])
    )

    st.caption(
        "Use t ∈ [0,1] to explain the 3D rotation by V, "
        "t ∈ [1,2] to show stretching/compression in (x′,y′,z′), "
        "and t ∈ [2,3] to show rotation inside the x′y′ plane."
    )

    # ---------- GIF animation section ----------
    st.markdown("## GIF animation from the 2×3 SVD path")

    if st.button("Generate GIF animation (svd2x3_animation.gif)"):
        with st.spinner("Generating 2×3 SVD GIF animation..."):
            try:
                create_animation_gif_2x3(
                    filename="svd2x3_animation.gif",
                    points_3d=points_3d,
                    cube_corners=cube_corners,
                    cube_edge_pairs=cube_edges,
                    U=U,
                    s=s,
                    V=V,
                    n_frames=120,
                    fps=15,
                    pause_seconds=1.0,
                    show_arrows=show_arrows,
                )
                st.success("Animation saved as svd2x3_animation.gif")
            except Exception as e:
                st.error(f"Failed to create animation. Error: {e}")

    if os.path.exists("svd2x3_animation.gif"):
        st.image("svd2x3_animation.gif")


if __name__ == "__main__":
    main()
