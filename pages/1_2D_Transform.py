"""
Created on Tue Jan  6 11:08:04 2026

@author: Brayden Miao
"""
# In the web UI:
# Adjust A (Angle+Scaling, Symmetric, Manual).
# Drag the slider to show each SVD stage.
# Scroll to the bottom and click Generate GIF animation.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import streamlit as st

# ---------- Core math functions ----------

def generate_random_cluster(n_points=30, mean=(0.0, 0.0), cov=None, seed=0):
    """
    Generate a random 2D cluster of points from a Gaussian distribution.
    """
    if cov is None:
        cov = np.array([[1.0, 0.6],
                        [0.6, 1.5]])

    rng = np.random.default_rng(seed)
    pts = rng.multivariate_normal(mean, cov, size=n_points)
    return pts  # shape (n_points, 2)


def apply_linear_transform(points, A):
    """
    Apply a 2x2 matrix A to all 2D points (rows) in 'points'.
    Uses row-vector convention: y = x A^T.
    """
    return points @ A.T


def rotation_matrix(theta):
    """
    2D rotation matrix for angle theta (radians).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def angle_from_rot(R):
    """
    Extract rotation angle from a 2×2 rotation matrix R.
    Assumes det(R) ~ 1.
    """
    return np.arctan2(R[1, 0], R[0, 0])


def make_rotational_svd(A):
    """
    Take SVD A = U Σ V^T and adjust U, V so both have det=+1 (pure rotations),
    possibly moving a reflection sign into Σ (signed scaling).

    Returns:
        U_rot, Sigma_signed, V_rot, s, Vt_raw
    where:
        - U_rot, V_rot are rotations (det ~ +1)
        - Sigma_signed is diagonal with possibly negative second entry
        - s are the usual non-negative singular values
        - Vt_raw is the original V^T from numpy.linalg.svd
    """
    U_raw, s, Vt_raw = np.linalg.svd(A)
    Sigma = np.diag(s)
    V_raw = Vt_raw.T

    U_rot = U_raw.copy()
    V_rot = V_raw.copy()
    Sigma_signed = Sigma.copy()

    detU = np.linalg.det(U_rot)
    detV = np.linalg.det(V_rot)
    R_ref = np.diag([1.0, -1.0])  # reflection across x-axis

    if detU < 0 and detV < 0:
        # Push a reflection into both U and V so both end up rotations.
        U_rot = U_rot @ R_ref
        V_rot = V_rot @ R_ref
        # Sigma stays the same, because R_ref Σ R_ref = Σ for diagonal Σ.
    elif detU < 0 and detV >= 0:
        # Move reflection into Σ via U.
        U_rot = U_rot @ R_ref
        Sigma_signed = R_ref @ Sigma_signed
    elif detU >= 0 and detV < 0:
        # Move reflection into Σ via V.
        V_rot = V_rot @ R_ref
        Sigma_signed = Sigma_signed @ R_ref

    return U_rot, Sigma_signed, V_rot, s, Vt_raw


def svd_path_transform(t, V_rot, Sigma_signed, U_rot, theta_V_rad, theta_U_rad):
    """
    Compute the current transform matrix M(t) along the SVD path:

    Stage 0: 0 <= t <= 1  : rotate from I to V_rot
    Stage 1: 1 <= t <= 2  : scale from I to Sigma_signed (after full V_rot)
    Stage 2: 2 <= t <= 3  : rotate from I to U_rot^T (after full V_rot and Sigma_signed)

    Returns:
        2x2 matrix M such that points -> points @ M.
    """
    s1_signed = Sigma_signed[0, 0]
    s2_signed = Sigma_signed[1, 1]

    if t <= 1.0:
        alpha = t
        phi = alpha * theta_V_rad
        Rv_t = rotation_matrix(phi)
        M = Rv_t
    elif t <= 2.0:
        alpha = t - 1.0
        Rv = rotation_matrix(theta_V_rad)
        s1_t = 1.0 + alpha * (s1_signed - 1.0)
        s2_t = 1.0 + alpha * (s2_signed - 1.0)
        S_t = np.array([[s1_t, 0.0],
                        [0.0,  s2_t]])
        M = Rv @ S_t
    else:
        alpha = t - 2.0
        Rv = rotation_matrix(theta_V_rad)
        S_final = Sigma_signed
        psi = alpha * theta_U_rad
        Ru_t = rotation_matrix(-psi)  # gradually approach U_rot^T
        M = Rv @ S_final @ Ru_t

    return M


# ---------- Plotting helpers ----------

def plot_overlay(points,
                 points_transformed,
                 A,
                 corners,
                 corners_transformed,
                 show_point_arrows=True,
                 show_original=True,
                 show_transformed=True,
                 draw_eigs=True,
                 title_suffix="",
                 xlim=None,
                 ylim=None):
    """
    Plot original and transformed points plus outer square and (optionally) eigenvectors.
    If xlim/ylim are provided, use them to keep the view stable.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_frame(ax, points, points_transformed, A, corners, corners_transformed,
               show_point_arrows=show_point_arrows,
               show_original=show_original,
               show_transformed=show_transformed,
               draw_eigs=draw_eigs,
               title_suffix=title_suffix,
               xlim=xlim, ylim=ylim)
    plt.tight_layout()
    return fig


def draw_frame(ax,
               points,
               points_transformed,
               A,
               corners,
               corners_transformed,
               show_point_arrows=True,
               show_original=True,
               show_transformed=True,
               draw_eigs=True,
               title_suffix="",
               xlim=None,
               ylim=None):
    """
    Draw a single frame on an existing Matplotlib Axes.
    Used both for Streamlit static plots and for animation.
    """
    ax.clear()

    # Original points (random + corners)
    if show_original:
        ax.scatter(points[:, 0], points[:, 1],
                   s=40, alpha=0.9, label="Original points")

        # Original outer square (black dashed)
        sq_x = np.append(corners[:, 0], corners[0, 0])
        sq_y = np.append(corners[:, 1], corners[0, 1])
        ax.plot(sq_x, sq_y, "k--", linewidth=1.5, label="Original square")

    # Transformed points
    if show_transformed:
        ax.scatter(points_transformed[:, 0], points_transformed[:, 1],
                   s=70, marker="x", alpha=0.9, label="Transformed points")

        # Transformed outer square (red dashed)
        sq_tx = np.append(corners_transformed[:, 0], corners_transformed[0, 0])
        sq_ty = np.append(corners_transformed[:, 1], corners_transformed[0, 1])
        ax.plot(sq_tx, sq_ty, "r--", linewidth=1.5, label="Transformed square")

    # Arrows from original to transformed
    if show_point_arrows and show_original and show_transformed:
        for (x, y), (xp, yp) in zip(points, points_transformed):
            ax.arrow(x, y, xp - x, yp - y,
                     head_width=0.08,
                     length_includes_head=True,
                     linewidth=1.0,
                     alpha=0.4)

    # Axes through origin
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)

    # Eigenvectors of A (only meaningful for final A)
    if draw_eigs:
        evals, evecs = np.linalg.eig(A)
        if np.all(np.isreal(evals)):
            all_x = np.concatenate([
                points[:, 0], points_transformed[:, 0],
                corners[:, 0], corners_transformed[:, 0], [0.0]
            ])
            all_y = np.concatenate([
                points[:, 1], points_transformed[:, 1],
                corners[:, 1], corners_transformed[:, 1], [0.0]
            ])
            max_rad = np.max(np.sqrt(all_x**2 + all_y**2)) or 1.0

            max_abs_lambda = np.max(np.abs(evals)) or 1.0
            base_scale = 0.7 * max_rad / max_abs_lambda

            for i in range(2):
                lam = float(np.real(evals[i]))
                v = np.real(evecs[:, i])
                v = v / np.linalg.norm(v)

                length = base_scale * abs(lam)

                ax.arrow(0, 0,
                         length * v[0], length * v[1],
                         head_width=0.12,
                         length_includes_head=True,
                         linewidth=2.0,
                         alpha=0.9,
                         label=f"eigenvector {i+1} (λ={lam:.2f})")

    # Square plotting region (equal x and y scale)
    if xlim is None or ylim is None:
        all_x = np.concatenate([
            points[:, 0], points_transformed[:, 0],
            corners[:, 0], corners_transformed[:, 0]
        ])
        all_y = np.concatenate([
            points[:, 1], points_transformed[:, 1],
            corners[:, 1], corners_transformed[:, 1]
        ])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)

        span = max(x_max - x_min, y_max - y_min)
        half = 0.5 * span * 1.1  # margin

        xlim = (x_mid - half, x_mid + half)
        ylim = (y_mid - half, y_mid + half)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    base_title = "2D Linear Transformation: points, square, eigenvectors"
    if title_suffix:
        ax.set_title(f"{base_title} {title_suffix}")
    else:
        ax.set_title(base_title)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize="small")


# ---------- Animation generator (GIF) ----------

def create_animation_gif(filename,
                         points,
                         corners,
                         A,
                         U_rot,
                         Sigma_signed,
                         V_rot,
                         theta_V_rad,
                         theta_U_rad,
                         xlim,
                         ylim,
                         show_arrows=True,
                         n_frames=120,
                         fps=30):
    """
    Generate a GIF animation following the SVD path t in [0, 3].
    Saves to 'filename', using PillowWriter (no ffmpeg needed).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    writer = PillowWriter(fps=fps)

    with writer.saving(fig, filename, dpi=150):
        for i in range(n_frames):
            t = 3.0 * i / (n_frames - 1)
            M = svd_path_transform(t, V_rot, Sigma_signed, U_rot,
                                   theta_V_rad, theta_U_rad)
            pts_M = points @ M
            crn_M = corners @ M

            draw_frame(
                ax,
                points,
                pts_M,
                A,
                corners,
                crn_M,
                show_point_arrows=show_arrows,
                show_original=True,
                show_transformed=True,
                draw_eigs=(i == n_frames - 1),  # eigenvectors on final frame
                title_suffix=f" (t={t:.2f})",
                xlim=xlim,
                ylim=ylim,
            )
            writer.grab_frame()

    plt.close(fig)


# ---------- Streamlit app (slider + GIF) ----------

def main():
    st.set_page_config(page_title="2D Linear Transformation (Rotation + GIF)",
                       layout="wide")

    st.title("2D Linear Transformation & Eigenvectors (Rotation Slider + GIF)")

    st.write(
        """
        This app shows how a **2×2 matrix** transforms a random cluster of **30 points** in 2D,
        plus an outer reference square, using a **true rotation–stretch–rotation path**:

        1. Rotate by $\\tilde V$  
        2. Stretch by a signed diagonal $\\tilde\\Sigma$  
        3. Rotate by $\\tilde U^T$  

        At the end, the total effect matches your matrix $A$.
        """
    )

    # Sidebar: random seed
    st.sidebar.header("Random points")
    seed = st.sidebar.slider("Random seed", 0, 100, 0, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("Transformation matrix A")

    # 3 modes: Angle+Scaling, Symmetric, Manual
    mode = st.sidebar.radio("Define A by:",
                            ["Angle + Scaling", "Symmetric", "Manual matrix entries"])

    # --- Define A from sidebar ---
    if mode == "Angle + Scaling":
        theta_deg = st.sidebar.slider("Rotation angle (degrees)", -180, 180, 30)
        scale_x = st.sidebar.slider("Scale in x direction", 0.1, 3.0, 2.0, 0.1)
        scale_y = st.sidebar.slider("Scale in y direction", 0.1, 3.0, 0.5, 0.1)

        theta = np.deg2rad(theta_deg)
        R = rotation_matrix(theta)
        S = np.array([[scale_x, 0.0],
                      [0.0,     scale_y]])
        A = R @ S

    elif mode == "Symmetric":
        st.sidebar.write("Enter entries for a symmetric matrix A:")
        # 2×2 grid layout, but only a11, a12, a22 are editable
        c1, c2 = st.sidebar.columns(2)
        with c1:
            a11 = st.number_input("a11", value=2.0, step=0.1, key="sym_a11")
            a12 = st.number_input("a12", value=0.5, step=0.1, key="sym_a12")
        with c2:
            st.markdown("a21 (symmetric to a12)")
            st.text(f"{a12:.3f}")
            a22 = st.number_input("a22", value=2.0, step=0.1, key="sym_a22")

        A = np.array([[a11, a12],
                      [a12, a22]])

    else:  # Manual matrix entries
        st.sidebar.write("Enter the entries of the 2×2 matrix A:")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            a11 = st.number_input("a11", value=2.0, step=0.1, key="man_a11")
            a21 = st.number_input("a21", value=0.5, step=0.1, key="man_a21")
        with c2:
            a12 = st.number_input("a12", value=1.0, step=0.1, key="man_a12")
            a22 = st.number_input("a22", value=2.0, step=0.1, key="man_a22")

        A = np.array([[a11, a12],
                      [a21, a22]])

    show_arrows = st.sidebar.checkbox(
        "Show arrows from original to transformed points",
        value=True,
    )

    # --- Rotational SVD of A: A = U_rot Σ_signed V_rot^T ---
    U_rot, Sigma_signed, V_rot, s, Vt_raw = make_rotational_svd(A)

    # Rotation angles for U_rot and V_rot
    theta_U_rad = angle_from_rot(U_rot)
    theta_V_rad = angle_from_rot(V_rot)
    theta_U_deg = np.degrees(theta_U_rad)
    theta_V_deg = np.degrees(theta_V_rad)

    # Random points + outer square
    random_points = generate_random_cluster(n_points=30, seed=seed)
    min_x, max_x = random_points[:, 0].min(), random_points[:, 0].max()
    min_y, max_y = random_points[:, 1].min(), random_points[:, 1].max()

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_span = 0.5 * max(max_x - min_x, max_y - min_y)
    half_side = half_span * 1.3 + 1e-6

    corners = np.array([
        [cx - half_side, cy - half_side],
        [cx + half_side, cy - half_side],
        [cx + half_side, cy + half_side],
        [cx - half_side, cy + half_side],
    ])

    points = np.vstack([random_points, corners])

    # Final transform using A^T = V_rot Σ_signed U_rot.T
    B3 = V_rot @ Sigma_signed @ U_rot.T
    points_A = points @ B3
    corners_A = corners @ B3

    # Global bounds for all slider positions
    all_x = np.concatenate([points[:, 0], points_A[:, 0],
                            corners[:, 0], corners_A[:, 0]])
    all_y = np.concatenate([points[:, 1], points_A[:, 1],
                            corners[:, 1], corners_A[:, 1]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.5 * span * 1.1
    xlim = (x_mid - half, x_mid + half)
    ylim = (y_mid - half, y_mid + half)

    # --- Slider controlling rotation + scaling path ---
    st.sidebar.markdown("---")
    t = st.sidebar.slider(
        "SVD path (0 = I, 1 = V rotation, 2 = V + Σ̃, 3 = full A via Uᵀ)",
        min_value=0.0,
        max_value=3.0,
        value=3.0,
        step=0.01,
    )

    # Current transform M(t)
    M = svd_path_transform(t, V_rot, Sigma_signed, U_rot,
                           theta_V_rad, theta_U_rad)
    points_M = points @ M
    corners_M = corners @ M

    # Eigenvalues of A
    evals, _ = np.linalg.eig(A)
    st.subheader("Transformation matrix A")
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        %.3f & %.3f \\
        %.3f & %.3f
        \end{bmatrix}
        """ % (A[0, 0], A[0, 1], A[1, 0], A[1, 1])
    )

    st.subheader("Eigenvalues of A")
    if np.all(np.isreal(evals)):
        st.latex(
            r"""
            \lambda_1 = %.3f,\quad
            \lambda_2 = %.3f
            """ % (np.real(evals[0]), np.real(evals[1]))
        )
    else:
        st.write("Eigenvalues are complex; real eigenvectors do not lie in the 2D plane.")

    # Layout
    col1, col2 = st.columns([3, 2])

    if t <= 1.0:
        suffix = "(Step 1: partial V rotation)"
    elif t <= 2.0:
        suffix = "(Step 2: stretching by Σ̃)"
    else:
        suffix = "(Step 3: partial Uᵀ rotation / towards full A)"

    with col1:
        st.subheader("Visualization (slider controlled)")
        fig = plot_overlay(points,
                           points_M,
                           A,
                           corners,
                           corners_M,
                           show_point_arrows=show_arrows,
                           show_original=True,
                           show_transformed=True,
                           draw_eigs=(t >= 2.99),
                           title_suffix=f" {suffix}",
                           xlim=xlim,
                           ylim=ylim)
        st.pyplot(fig, width="stretch")

    with col2:
        st.subheader("Rotation + stretching view (via rotational SVD)")
        st.markdown(
            r"""
We build a rotation-friendly version of SVD:

$$
A = \tilde U \,\tilde\Sigma\, \tilde V^T,
$$

where:
- $\tilde U$ and $\tilde V$ are **pure rotations** (determinant $=1$),
- $\tilde\Sigma$ is diagonal and may carry a sign to encode reflections
  (so we call it a **signed scaling** matrix).
"""
        )

        st.latex(
            r"""
            \Sigma =
            \begin{bmatrix}
            %.3f & 0 \\
            0 & %.3f
            \end{bmatrix}
            """
            % (s[0], s[1])
        )

        st.latex(
            r"""
            \tilde U \approx
            \begin{bmatrix}
            %.3f & %.3f \\
            %.3f & %.3f
            \end{bmatrix}
            """
            % (U_rot[0, 0], U_rot[0, 1], U_rot[1, 0], U_rot[1, 1])
        )

        Vt_disp = V_rot.T
        st.latex(
            r"""
            \tilde V^T \approx
            \begin{bmatrix}
            %.3f & %.3f \\
            %.3f & %.3f
            \end{bmatrix}
            """
            % (Vt_disp[0, 0], Vt_disp[0, 1], Vt_disp[1, 0], Vt_disp[1, 1])
        )

        st.markdown(
            r"""
For a pure 2D rotation matrix
$$
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix},
$$
the **first column** is $\bigl[\cos\theta,\ \sin\theta\bigr]^T$.

So we can read approximate angles from the first columns of $\tilde U$ and $\tilde V$.
"""
        )

        st.latex(
            r"""
            \theta_{\tilde U} \approx %.1f^\circ,\quad
            \cos(\theta_{\tilde U}) \approx \tilde U_{11} = %.3f,\quad
            \sin(\theta_{\tilde U}) \approx \tilde U_{21} = %.3f
            """
            % (theta_U_deg, U_rot[0, 0], U_rot[1, 0])
        )

        st.latex(
            r"""
            \theta_{\tilde V} \approx %.1f^\circ,\quad
            \cos(\theta_{\tilde V}) \approx \tilde V_{11} = %.3f,\quad
            \sin(\theta_{\tilde V}) \approx \tilde V_{21} = %.3f
            """
            % (theta_V_deg, V_rot[0, 0], V_rot[1, 0])
        )

    st.markdown("---")
    st.caption(
        "Drag the slider slowly while you explain: "
        "I → partial V rotation → full V → stretching by Σ̃ → partial Uᵀ → full A."
    )

    # ---------- GIF generation section ----------
    st.markdown("## GIF animation from the SVD path")

    if st.button("Generate GIF animation (svd2d_animation.gif)"):
        with st.spinner("Generating GIF animation (this may take a bit)..."):
            try:
                create_animation_gif(
                    filename="svd2d_animation.gif",
                    points=points,
                    corners=corners,
                    A=A,
                    U_rot=U_rot,
                    Sigma_signed=Sigma_signed,
                    V_rot=V_rot,
                    theta_V_rad=theta_V_rad,
                    theta_U_rad=theta_U_rad,
                    xlim=xlim,
                    ylim=ylim,
                    show_arrows=show_arrows,
                    n_frames=120,
                    fps=30,
                )
                st.success("Animation saved as svd2d_animation.gif")
            except Exception as e:
                st.error(f"Failed to create animation. Error: {e}")

    if os.path.exists("svd2d_animation.gif"):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("svd2d_animation.gif")


if __name__ == "__main__":
    main()
