"""
Interactive textured 3D shape with fixed lighting.

Install:
  pip install pyvista numpy

Run (interactive viewer):
  python texture_gen.py

Optional static panel render:
  python texture_gen.py --panel

Optional free-view panel (legacy behavior; panel 2/3 dots need not align):
  python texture_gen.py --panel --free-view
"""

import argparse
from pathlib import Path
import numpy as np
import pyvista as pv

BW_CMAP = ["white", "black"]
BASE_SURFACE_COLOR = "#ededed"
TEXTURE_CMAP = [BASE_SURFACE_COLOR, "black"]
INTERACTIVE_WINDOW_SIZE = (1600, 1200)
PANEL_WINDOW_SIZE = (2400, 1040)
SHAPE_THETA_RES = 560
SHAPE_PHI_RES = 560
# One-switch panel interaction mode:
# True  -> allow rotate/orbit in panel view
# False -> lock rotation (pan/zoom only)
PANEL_ROTATION_ENABLED = True
# Controls how different shapes A and B are in panel mode while preserving silhouette.
# 0.0 -> very similar depth; 1.0 -> clear difference; >1.0 -> stronger separation.
PANEL_SHAPE_DIFF_STRENGTH = 1.35


# -----------------------------
# 1) Shape: smooth perturbed sphere
# -----------------------------
def make_perturbed_sphere(
    radius=1.0,
    theta_res=SHAPE_THETA_RES,
    phi_res=SHAPE_PHI_RES,
    n_waves=30,
    amp=0.34,
    k_range=(3.0, 12.0),
    seed=2,
):
    """
    Create a smooth, randomly perturbed sphere surface.
    Uses a sum of sinusoidal radial bumps along random directions.
    """
    rng = np.random.default_rng(seed)

    # Start from a closed sphere mesh so there is no seam/slit.
    surf = pv.Sphere(radius=radius, theta_resolution=theta_res, phi_resolution=phi_res).triangulate()
    n = surf.points.copy()
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

    # Random sinusoid mixture on unit directions
    r = np.full(n.shape[0], radius, dtype=float)
    for _ in range(n_waves):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u) + 1e-12
        k = rng.uniform(*k_range)
        phase = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.35, 1.0) * amp / np.sqrt(n_waves)

        # Dot with direction field -> smooth bump bands
        dot = n @ u
        r += a * np.sin(k * dot + phase)

    # Apply perturbed radius on the closed mesh.
    surf.points = n * r[:, None]
    surf = surf.clean(tolerance=1e-12)

    # Keep a robust smoothing path (some VTK builds can crash on subdivide).
    surf = surf.smooth(n_iter=12, relaxation_factor=0.04, feature_smoothing=False)

    # Normalize size for consistent texture scale
    surf.points /= np.max(np.linalg.norm(surf.points, axis=1))
    return surf


# -----------------------------
# 2) Volumetric texture fields
# -----------------------------
def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blob_texture(points, period=0.22, sphere_radius=0.075, anisotropy=(1.0, 1.0, 1.0), edge_softness=0.012):
    """
    "Blob" texture by tiling spheres in a 3D lattice, then thresholding.
    This is a simplified lattice (cubic), but the perceptual idea matches:
    a 3D packed field intersected by the surface.

    anisotropy stretches the lattice axes (e.g., (2,1,1) -> elongated blobs).
    edge_softness controls how soft the dot boundary is.
    Returns values in [0,1] where 1=black and 0=white.
    """
    a = np.array(anisotropy, dtype=float)
    p = points / a  # stretching the *texture space* causes anisotropic surface pattern

    # Map to cell coordinates
    q = p / period
    q0 = np.floor(q)
    # nearest lattice point (within this simple model, it's the cell corner)
    # Compute distance to nearest of 8 corners for a more "packed" look:
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)

    # q within cell:
    f = q - q0
    # Dist to each corner
    d2 = np.min(np.sum((f[:, None, :] - corners[None, :, :]) ** 2, axis=2), axis=1)
    d = np.sqrt(d2) * period

    # Soft threshold for anti-aliased dot edges (optional hard edge if softness <= 0).
    if edge_softness <= 0:
        return (d < sphere_radius).astype(np.float32)
    ramp = (sphere_radius + edge_softness - d) / (2.0 * edge_softness)
    return smoothstep01(ramp).astype(np.float32)


def _orthonormal_basis_from_axis(axis):
    z = np.asarray(axis, dtype=np.float32)
    z /= np.linalg.norm(z) + 1e-12
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(z @ ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x = np.cross(ref, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_nonoverlap_centers(points, n_centers, min_dist, seed=0, max_trials=500000):
    """
    Sample approximately isotropic surface centers with a hard minimum distance.
    Uses rejection sampling over mesh vertices (Poisson-disk-like on the surface).
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float32)
    n_pts = pts.shape[0]
    order = rng.integers(0, n_pts, size=max_trials, endpoint=False)
    min_dist2 = float(min_dist * min_dist)

    centers = []
    for idx in order:
        c = pts[idx]
        if not centers:
            centers.append(c)
        else:
            existing = np.asarray(centers, dtype=np.float32)
            d2 = np.sum((existing - c[None, :]) ** 2, axis=1)
            if np.all(d2 >= min_dist2):
                centers.append(c)
        if len(centers) >= n_centers:
            break

    return np.asarray(centers, dtype=np.float32)


def sample_farthest_centers(points, n_centers, seed=0):
    """
    Quasi-uniform center placement by iterative farthest-point sampling on vertices.
    Produces a regular isotropic-looking arrangement (no directional streaking).
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float32)
    n_pts = pts.shape[0]
    n_centers = int(np.clip(n_centers, 1, n_pts))

    first = int(rng.integers(0, n_pts))
    chosen = [first]
    min_d2 = np.sum((pts - pts[first][None, :]) ** 2, axis=1)

    for _ in range(1, n_centers):
        nxt = int(np.argmax(min_d2))
        chosen.append(nxt)
        d2 = np.sum((pts - pts[nxt][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    return pts[np.asarray(chosen, dtype=np.int64)]


def paper_dot_texture(
    points,
    n_dots=450,
    radii=(0.055, 0.055, 0.055),
    axis=(0.0, 0.0, 1.0),
    edge_softness=0.08,
    seed=7,
    min_center_dist=None,
    chunk_size=12000,
    aa_samples=5,
    aa_radius=0.0035,
    placement="farthest",
):
    """
    Todd & Thaler-style volumetric dots:
    place 3D spheres/ellipsoids with centers on the surface and mark
    black where the surface intersects those volumes.

    radii are ellipsoid semi-axes in world units (object is normalized to ~unit radius).
    """
    pts = np.asarray(points, dtype=np.float32)
    n_pts = pts.shape[0]
    n_centers = int(np.clip(n_dots, 1, n_pts))
    r = np.asarray(radii, dtype=np.float32)
    if placement == "farthest":
        centers = sample_farthest_centers(pts, n_centers=n_centers, seed=seed)
    else:
        if min_center_dist is None:
            # Non-overlap guarantee for isotropic dots; conservative for ellipsoids.
            min_center_dist = 2.05 * float(np.max(r))
        centers = sample_nonoverlap_centers(
            pts,
            n_centers=n_centers,
            min_dist=min_center_dist,
            seed=seed,
            max_trials=800000,
        )
    if centers.shape[0] == 0:
        return np.zeros(n_pts, dtype=np.float32)

    basis = _orthonormal_basis_from_axis(axis)
    inv_radii2 = 1.0 / (r ** 2 + 1e-12)
    values = np.zeros(n_pts, dtype=np.float32)
    center_chunk = 96

    # Centered plus symmetric offsets for supersampled edge coverage.
    offsets = [np.zeros(3, dtype=np.float32)]
    if aa_samples >= 2 and aa_radius > 0:
        offsets.extend(
            [
                np.array([aa_radius, 0.0, 0.0], dtype=np.float32),
                np.array([-aa_radius, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, aa_radius, 0.0], dtype=np.float32),
                np.array([0.0, -aa_radius, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, aa_radius], dtype=np.float32),
                np.array([0.0, 0.0, -aa_radius], dtype=np.float32),
            ]
        )
        offsets = offsets[: max(1, int(aa_samples))]

    for i0 in range(0, n_pts, chunk_size):
        i1 = min(i0 + chunk_size, n_pts)
        p = pts[i0:i1]  # (B, 3)
        cov_accum = np.zeros(p.shape[0], dtype=np.float32)

        for off in offsets:
            ps = p + off[None, :]
            min_e2 = np.full(ps.shape[0], np.inf, dtype=np.float32)

            # Exact nearest-ellipsoid query (chunked over centers to control memory).
            for c0 in range(0, centers.shape[0], center_chunk):
                c1 = min(c0 + center_chunk, centers.shape[0])
                d = ps[:, None, :] - centers[None, c0:c1, :]  # (B, Cc, 3)
                local = d @ basis  # (B, Cc, 3)
                e2 = np.sum((local * local) * inv_radii2[None, None, :], axis=2)  # (B, Cc)
                min_e2 = np.minimum(min_e2, np.min(e2, axis=1))

            ellip = np.sqrt(min_e2)
            if edge_softness <= 0:
                cov = (ellip <= 1.0).astype(np.float32)
            else:
                ramp = (1.0 + edge_softness - ellip) / (2.0 * edge_softness)
                cov = smoothstep01(ramp).astype(np.float32)
            cov_accum += cov

        values[i0:i1] = cov_accum / float(len(offsets))

    return values


def contour_texture(points, period=0.12, direction=(0.0, 0.0, 1.0)):
    """
    "Contour" texture via alternating slabs (plane waves).
    direction defines slab normal in world coordinates.
    Returns values in {0,1}.
    """
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d) + 1e-12

    # Coordinate along direction
    t = points @ d
    # Alternating slabs using a square wave from sine
    s = np.sin(2 * np.pi * t / period)
    return (s > 0).astype(np.float32)


def isotropic_variable_dot_texture(
    points,
    period=0.23,
    radius_mean=0.066,
    size_randomness=0.0,
    edge_softness=0.045,
    seed=11,
):
    """
    Fast isotropic volumetric dots for real-time interaction.
    Dots are centered on a 3D cubic lattice (isotropic in volume), and each cell gets
    a deterministic random dot radius for controlled size variability.
    """
    p = np.asarray(points, dtype=np.float32)
    q = p / float(period)
    base_cell = np.floor(q).astype(np.int32)
    s = np.uint32(seed)

    # Evaluate best neighboring sphere using normalized distance (d/r), not raw distance.
    # This avoids false "holes" when nearby tiny dots beat larger nearby dots.
    min_norm = np.full(p.shape[0], np.inf, dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = base_cell + np.array([dx, dy, dz], dtype=np.int32)
                center = cell.astype(np.float32) + 0.5
                d = np.linalg.norm(q - center, axis=1) * float(period)

                # Hash cell coordinates -> deterministic uniform random u in [0, 1).
                x = cell[:, 0].astype(np.uint32)
                y = cell[:, 1].astype(np.uint32)
                z = cell[:, 2].astype(np.uint32)
                h = x * np.uint32(73856093) ^ y * np.uint32(19349663) ^ z * np.uint32(83492791) ^ s
                u = (h.astype(np.float64) / np.float64(2**32 - 1)).astype(np.float32)

                # Uniform radius spread around mean.
                r = float(np.clip(size_randomness, 0.0, 0.95))
                min_scale = max(0.65, 1.0 - r)
                max_scale = 1.0 + r
                radius = radius_mean * (min_scale + (max_scale - min_scale) * u)

                # Keep circles non-overlapping in texture volume.
                max_allowed = 0.485 * float(period)
                radius = np.minimum(radius, max_allowed)

                norm = d / (radius + 1e-12)
                min_norm = np.minimum(min_norm, norm)

    if edge_softness <= 0:
        return (min_norm <= 1.0).astype(np.float32)
    soft_norm = max(1e-6, edge_softness / (radius_mean + 1e-12))
    ramp = (1.0 + soft_norm - min_norm) / (2.0 * soft_norm)
    return smoothstep01(ramp).astype(np.float32)


# -----------------------------
# 3) Rendering helpers
# -----------------------------
def render_panel(mesh, scalar, title, plotter, row, col):
    """
    Add a subplot with consistent lighting/camera.
    """
    plotter.subplot(row, col)

    m = mesh.copy(deep=True)
    m["tex"] = scalar

    plotter.add_text(title, font_size=12)
    plotter.add_mesh(
        m,
        scalars="tex",
        cmap=BW_CMAP,
        clim=[0, 1],
        smooth_shading=True,
        specular=0.0,
        specular_power=20,
        show_scalar_bar=False,
    )
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.25)
    plotter.enable_lightkit()


def configure_fixed_light(plotter):
    """
    Approximate indirect illumination with a small world-fixed light rig:
    key + cool sky fill + warm ground bounce.
    """
    if hasattr(plotter.renderer, "remove_all_lights"):
        plotter.renderer.remove_all_lights()

    key = pv.Light(
        position=(2.5, 1.7, 3.0),
        focal_point=(0.0, 0.0, 0.0),
        color="white",
        intensity=0.85,
        positional=False,
    )
    sky_fill = pv.Light(
        position=(-2.2, 2.8, 1.2),
        focal_point=(0.0, 0.0, 0.0),
        color=(0.82, 0.88, 1.0),
        intensity=0.22,
        positional=False,
    )
    bounce_fill = pv.Light(
        position=(0.0, -2.8, -1.8),
        focal_point=(0.0, 0.0, 0.0),
        color=(1.0, 0.93, 0.82),
        intensity=0.18,
        positional=False,
    )
    plotter.add_light(key)
    plotter.add_light(sky_fill)
    plotter.add_light(bounce_fill)
    return {"key": key, "sky": sky_fill, "bounce": bounce_fill}


def enable_gi_approx(plotter):
    """
    Enable the best available GI approximation in this VTK build.
    """
    if hasattr(plotter, "enable_ssao"):
        plotter.enable_ssao(radius=0.22, bias=0.008, kernel_size=256, blur=True)


def _camera_basis(cam_pos, focal, view_up):
    c = np.asarray(cam_pos, dtype=np.float32)
    f = np.asarray(focal, dtype=np.float32)
    up = np.asarray(view_up, dtype=np.float32)
    forward = f - c
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up) + 1e-12
    return c, f, right, true_up, forward


def _points_to_camera_coords(points, cam_pos, focal, view_up):
    _, f, right, true_up, forward = _camera_basis(cam_pos, focal, view_up)
    q = np.asarray(points, dtype=np.float32) - f[None, :]
    u = q @ right
    v = q @ true_up
    w = q @ forward
    return np.stack([u, v, w], axis=1)


def _camera_coords_to_points(uvw, cam_pos, focal, view_up):
    _, f, right, true_up, forward = _camera_basis(cam_pos, focal, view_up)
    uvw = np.asarray(uvw, dtype=np.float32)
    return (
        f[None, :]
        + uvw[:, 0:1] * right[None, :]
        + uvw[:, 1:2] * true_up[None, :]
        + uvw[:, 2:3] * forward[None, :]
    )


def enforce_same_front_silhouette(reference_mesh, target_mesh, camera):
    """
    Force `target_mesh` to have exactly the same front-view silhouette as
    `reference_mesh` from `camera`, while keeping target depth variation.
    """
    cam_pos, focal, view_up = camera
    ref_uvw = _points_to_camera_coords(reference_mesh.points, cam_pos, focal, view_up)
    tgt_uvw = _points_to_camera_coords(target_mesh.points, cam_pos, focal, view_up)

    out = tgt_uvw.copy()
    # Copy only image-plane coords (u,v). Keep target depth (w) to preserve internal bumps.
    out[:, 0] = ref_uvw[:, 0]
    out[:, 1] = ref_uvw[:, 1]

    # Keep absolute scale unchanged so the rendered silhouette stays matched.
    target_mesh.points = _camera_coords_to_points(out, cam_pos, focal, view_up)
    return target_mesh


def add_depth_bumps_in_view(
    mesh,
    camera,
    seed=0,
    n_bumps=7,
    amp=0.12,
    sigma_range=(0.22, 0.45),
    center_bulge=0.0,
    lateral_strength=0.0,
    center_offset=(0.0, 0.0),
):
    """
    Add smooth bump structure in camera-space coordinates.
    - depth (w) always varies
    - in-plane (u,v) variation can be enabled via lateral_strength
    """
    rng = np.random.default_rng(seed)
    cam_pos, focal, view_up = camera
    uvw = _points_to_camera_coords(mesh.points, cam_pos, focal, view_up)
    u = uvw[:, 0]
    v = uvw[:, 1]

    umin, umax = float(np.min(u)), float(np.max(u))
    vmin, vmax = float(np.min(v)), float(np.max(v))
    us = (u - umin) / (umax - umin + 1e-12)
    vs = (v - vmin) / (vmax - vmin + 1e-12)
    uu = 2.0 * us - 1.0
    vv = 2.0 * vs - 1.0

    bump_count = max(1, int(n_bumps))
    du = np.zeros_like(uu, dtype=np.float32)
    dv = np.zeros_like(uu, dtype=np.float32)
    dw = np.zeros_like(uu, dtype=np.float32)
    signs = np.ones(bump_count, dtype=np.float32)
    signs[: bump_count // 2] = -1.0
    rng.shuffle(signs)
    for sign in signs:
        # Broad center-biased placement gives bumps in all directions while remaining interior.
        cu = float(np.clip(rng.normal(float(center_offset[0]), 0.44), -0.84, 0.84))
        cv = float(np.clip(rng.normal(float(center_offset[1]), 0.44), -0.84, 0.84))
        sigma = rng.uniform(*sigma_range)
        # Moderate falloff with bump_count keeps detail but avoids over-strong relief.
        a = sign * float(amp) * rng.uniform(0.62, 0.96) / (bump_count ** 0.45)
        g = np.exp(-((uu - cu) ** 2 + (vv - cv) ** 2) / (2.0 * sigma * sigma))
        dir_vec = rng.normal(size=3).astype(np.float32)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-12
        # Keep meaningful depth component so bumps read as in/out.
        dir_vec[2] = np.sign(dir_vec[2]) * max(abs(float(dir_vec[2])), 0.48)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-12
        lat = float(lateral_strength)
        du += (a * lat * dir_vec[0] * g).astype(np.float32)
        dv += (a * lat * dir_vec[1] * g).astype(np.float32)
        dw += (a * dir_vec[2] * g).astype(np.float32)

    if center_bulge != 0.0:
        g0 = np.exp(-(uu * uu + vv * vv) / (2.0 * (0.60**2)))
        dw += (float(center_bulge) * g0).astype(np.float32)

    # Fade relief near outline to keep bumps mostly interior.
    rr = np.sqrt(uu * uu + vv * vv)
    edge_taper = np.clip((1.03 - rr) / 0.22, 0.0, 1.0)
    edge_taper = smoothstep01(edge_taper).astype(np.float32)
    taper_w = (0.24 + 0.76 * edge_taper).astype(np.float32)
    taper_uv = (0.24 + 0.76 * edge_taper).astype(np.float32)
    center_boost = np.exp(-(rr * rr) / (2.0 * (0.40**2))).astype(np.float32)
    du *= taper_uv
    dv *= taper_uv
    dw *= (taper_w * (1.0 + 0.62 * center_boost)).astype(np.float32)

    # Keep centroid stable while changing internal relief.
    du -= np.float32(np.mean(du))
    dv -= np.float32(np.mean(dv))
    dw -= np.float32(np.mean(dw))
    uvw[:, 0] += du
    uvw[:, 1] += dv
    uvw[:, 2] += dw
    mesh.points = _camera_coords_to_points(uvw, cam_pos, focal, view_up)
    return mesh


def add_radial_bumps_on_sphere(mesh, seed=0, n_bumps=20, amp=0.20, sigma_range=(0.10, 0.24)):
    """
    Apply smooth random in/out radial bumps to a sphere-like mesh.
    Bumps are centered on random unit directions and use Gaussian angular falloff.
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(mesh.points, dtype=np.float32)
    radii = np.linalg.norm(pts, axis=1)
    n = pts / (radii[:, None] + 1e-12)

    bump_count = max(1, int(n_bumps))
    delta = np.zeros(n.shape[0], dtype=np.float32)
    signs = np.ones(bump_count, dtype=np.float32)
    signs[: bump_count // 2] = -1.0
    rng.shuffle(signs)

    for sign in signs:
        cdir = rng.normal(size=3).astype(np.float32)
        cdir /= np.linalg.norm(cdir) + 1e-12
        dot = np.clip(n @ cdir, -1.0, 1.0)
        sigma = float(rng.uniform(*sigma_range))
        g = np.exp(-((1.0 - dot) ** 2) / (2.0 * sigma * sigma))
        a = sign * float(amp) * rng.uniform(0.65, 1.0) / (bump_count ** 0.45)
        delta += (a * g).astype(np.float32)

    # Keep object centered and avoid global inflation/deflation drift.
    delta -= np.float32(np.mean(delta))
    out_r = radii * (1.0 + delta)
    out_r = np.clip(out_r, 0.22, None)
    mesh.points = (n * out_r[:, None]).astype(np.float32)
    return mesh


def make_conflicting_surface_pair(theta_res=300, phi_res=300, fixed_camera=None, seed=None, diff_strength=1.0):
    """
    Build two same-topology surfaces with intentionally different geometry:
    - surface A drives shading
    - surface B defines where texture dots live
    Texture can then be transferred exactly by vertex index.
    """
    d = float(np.clip(diff_strength, 0.0, 2.5))

    # Randomized seeds so bump locations vary between runs unless a seed is provided.
    rng = np.random.default_rng(seed)
    seed_shape_a = int(rng.integers(0, 2**31 - 1))
    seed_shape_b = int(rng.integers(0, 2**31 - 1))
    seed_shape_b_extra = int(rng.integers(0, 2**31 - 1))
    seed_depth_a = int(rng.integers(0, 2**31 - 1))
    seed_depth_b = int(rng.integers(0, 2**31 - 1))
    # Start from a perfect sphere for both A and B.
    base = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res).triangulate().clean(tolerance=1e-12)
    surf_a = base.copy(deep=True)
    surf_b = base.copy(deep=True)

    # Independent in/out radial bump fields so A and B are definitely different.
    surf_a = add_radial_bumps_on_sphere(
        surf_a,
        seed=seed_shape_a,
        n_bumps=int(np.clip(round(2.0 * (18 + 10 * d)), 20, 88)),
        amp=0.44 + 0.20 * d,
        sigma_range=(0.10, 0.22),
    )
    surf_b = add_radial_bumps_on_sphere(
        surf_b,
        seed=seed_shape_b,
        n_bumps=int(np.clip(round(2.0 * (22 + 11 * d)), 24, 104)),
        amp=0.48 + 0.22 * d,
        sigma_range=(0.10, 0.23),
    )

    # If random draws end up too similar, force extra independent bumps on B.
    mean_delta = float(np.mean(np.linalg.norm(surf_a.points - surf_b.points, axis=1)))
    if mean_delta < 0.09:
        surf_b = add_radial_bumps_on_sphere(
            surf_b,
            seed=seed_shape_b_extra,
            n_bumps=int(np.clip(round(2.0 * (10 + 6 * d)), 12, 48)),
            amp=0.24 + 0.12 * d,
            sigma_range=(0.08, 0.18),
        )

    surf_a = surf_a.smooth(
        n_iter=int(np.clip(round(12 + 5 * d), 8, 24)),
        relaxation_factor=float(np.clip(0.045 + 0.008 * d, 0.03, 0.08)),
        feature_smoothing=False,
    )
    surf_b = surf_b.smooth(
        n_iter=int(np.clip(round(12 + 6 * d), 8, 26)),
        relaxation_factor=float(np.clip(0.046 + 0.009 * d, 0.03, 0.08)),
        feature_smoothing=False,
    )

    # Keep both normalized to the same global scale.
    surf_a.points /= np.max(np.linalg.norm(surf_a.points, axis=1)) + 1e-12
    surf_b.points /= np.max(np.linalg.norm(surf_b.points, axis=1)) + 1e-12

    if fixed_camera is not None:
        surf_b = enforce_same_front_silhouette(surf_a, surf_b, fixed_camera)
        # Add independent in/out depth structure while preserving silhouette.
        sigma_a_lo = float(np.clip(0.10 + 0.02 * d, 0.09, 0.18))
        sigma_a_hi = float(np.clip(0.24 + 0.03 * d, sigma_a_lo + 0.05, 0.36))
        sigma_b_lo = float(np.clip(0.10 + 0.02 * d, 0.09, 0.18))
        sigma_b_hi = float(np.clip(0.25 + 0.03 * d, sigma_b_lo + 0.05, 0.38))
        offset_a = rng.uniform(-0.16, 0.16, size=2).astype(np.float32)
        offset_b = rng.uniform(-0.16, 0.16, size=2).astype(np.float32)
        if np.linalg.norm(offset_a - offset_b) < 0.12:
            offset_b = np.clip(-offset_a + rng.uniform(-0.05, 0.05, size=2), -0.22, 0.22).astype(np.float32)
        center_a = float(rng.uniform(-0.06, 0.06) * (0.35 + 0.60 * d))
        center_b = float(rng.uniform(-0.06, 0.06) * (0.35 + 0.60 * d))
        surf_a = add_depth_bumps_in_view(
            surf_a,
            fixed_camera,
            seed=seed_depth_a,
            n_bumps=int(np.clip(round(2.0 * (20 + 10 * d)), 24, 108)),
            amp=0.36 + 0.24 * d,
            sigma_range=(sigma_a_lo, sigma_a_hi),
            center_bulge=center_a,
            lateral_strength=0.22 + 0.10 * d,
            center_offset=(float(offset_a[0]), float(offset_a[1])),
        )
        surf_b = add_depth_bumps_in_view(
            surf_b,
            fixed_camera,
            seed=seed_depth_b,
            n_bumps=int(np.clip(round(2.0 * (22 + 10 * d)), 24, 112)),
            amp=0.36 + 0.26 * d,
            sigma_range=(sigma_b_lo, sigma_b_hi),
            center_bulge=center_b,
            lateral_strength=0.22 + 0.11 * d,
            center_offset=(float(offset_b[0]), float(offset_b[1])),
        )
        surf_a = surf_a.smooth(
            n_iter=int(np.clip(round(8 + 4 * d), 5, 18)),
            relaxation_factor=float(np.clip(0.038 + 0.008 * d, 0.025, 0.06)),
            feature_smoothing=False,
        )
        surf_b = surf_b.smooth(
            n_iter=int(np.clip(round(9 + 4 * d), 5, 19)),
            relaxation_factor=float(np.clip(0.039 + 0.008 * d, 0.025, 0.06)),
            feature_smoothing=False,
        )
        # Re-enforce silhouette after depth edits (numerical safety).
        surf_b = enforce_same_front_silhouette(surf_a, surf_b, fixed_camera)

    if surf_a.n_points != surf_b.n_points:
        raise RuntimeError("Conflicting surfaces must have matching point counts for exact transfer.")
    # Enforce identical triangulation/connectivity to avoid panel 2 vs 3 raster mismatch.
    surf_b = pv.PolyData(surf_b.points.copy(), surf_a.faces.copy())
    return surf_a, surf_b


def project_points_to_view(points, cam_pos, focal, view_up):
    """
    Orthographic-style 2D coordinates of points in a fixed camera plane.
    Returns Nx2 (u,v) coordinates in world units.
    """
    p = np.asarray(points, dtype=np.float32)
    c = np.asarray(cam_pos, dtype=np.float32)
    f = np.asarray(focal, dtype=np.float32)
    up = np.asarray(view_up, dtype=np.float32)

    forward = f - c
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up) + 1e-12

    q = p - f[None, :]
    u = q @ right
    v = q @ true_up
    return np.stack([u, v], axis=1)


def run_panel_render(fixed_view=False, shape_diff_strength=PANEL_SHAPE_DIFF_STRENGTH):
    fixed_camera = [(0.0, 0.0, 3.1), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    panel_capture_y_shift_px = 150
    # Cue comparison design:
    # - Panel 1 is shape A shading.
    # - Panel 2 is texture on shape B using B's native 3D coordinates (surface-conforming dots).
    # - Panels 3/4 share one transferred dot mask.
    # - Panels 1 and 4 use the same true renderer lighting/material pipeline.
    surf_shading, surf_texture = make_conflicting_surface_pair(
        theta_res=300,
        phi_res=300,
        fixed_camera=fixed_camera if fixed_view else None,
        diff_strength=shape_diff_strength,
    )

    if fixed_view:
        # Center both shapes in fixed camera image plane so saved panel crops are centered.
        uv_a0 = project_points_to_view(np.asarray(surf_shading.points, dtype=np.float32), *fixed_camera).astype(np.float32)
        uv_b0 = project_points_to_view(np.asarray(surf_texture.points, dtype=np.float32), *fixed_camera).astype(np.float32)
        uv_all0 = np.vstack([uv_a0, uv_b0])
        u_center = 0.5 * (float(np.min(uv_all0[:, 0])) + float(np.max(uv_all0[:, 0])))
        v_center = 0.5 * (float(np.min(uv_all0[:, 1])) + float(np.max(uv_all0[:, 1])))
        _, _, right, true_up, _ = _camera_basis(*fixed_camera)
        shift_world = (-u_center) * right + (-v_center) * true_up
        surf_shading.points = (np.asarray(surf_shading.points, dtype=np.float32) + shift_world[None, :]).astype(np.float32)
        surf_texture.points = (np.asarray(surf_texture.points, dtype=np.float32) + shift_world[None, :]).astype(np.float32)

    points_texture_snapshot = np.asarray(surf_texture.points, dtype=np.float32)
    points_shading_snapshot = np.asarray(surf_shading.points, dtype=np.float32)
    fixed_parallel_scale = None
    if fixed_view:
        uv_texture_snapshot = project_points_to_view(points_texture_snapshot, *fixed_camera).astype(np.float32)
        uv_shading_snapshot = project_points_to_view(points_shading_snapshot, *fixed_camera).astype(np.float32)
        uv_all = np.vstack([uv_texture_snapshot, uv_shading_snapshot])
        u_max = float(np.max(np.abs(uv_all[:, 0])))
        v_max = float(np.max(np.abs(uv_all[:, 1])))
        panel_aspect = max((PANEL_WINDOW_SIZE[0] / 4.0) / PANEL_WINDOW_SIZE[1], 1e-6)
        # In parallel projection, visible half-height is parallel_scale.
        fixed_parallel_scale = 1.10 * max(v_max, u_max / panel_aspect)
    else:
        uv_texture_snapshot = None
        uv_shading_snapshot = None

    # Random non-overlapping center sampling each update (non-grid, no fixed anchors).
    n_pts = points_texture_snapshot.shape[0]
    rng = np.random.default_rng(1234)
    dot_radius = 0.078
    edge_softness = 0.010
    # Strict non-overlap (including soft edges).
    min_center_dist = 2.08 * (dot_radius + edge_softness)

    def sample_random_nonoverlap_center_indices(k, min_center_dist):
        min_dist2 = float(min_center_dist**2)
        selected_idx = []

        # A few passes keep it fast while still filling target count reliably.
        for _ in range(3):
            need = max(0, k - len(selected_idx))
            if need == 0:
                break
            candidate_count = min(n_pts, max(4000, 28 * need))
            cand_idx = rng.choice(n_pts, size=candidate_count, replace=False)
            candidates = points_texture_snapshot[cand_idx]
            for idx, c in zip(cand_idx, candidates):
                if not selected_idx:
                    selected_idx.append(int(idx))
                else:
                    s = points_texture_snapshot[np.asarray(selected_idx, dtype=np.int64)]
                    d2 = np.sum((s - c[None, :]) ** 2, axis=1)
                    if np.all(d2 >= min_dist2):
                        selected_idx.append(int(idx))
                if len(selected_idx) >= k:
                    break

        if not selected_idx:
            return rng.choice(n_pts, size=1, replace=False).astype(np.int64)

        # Safety cleanup: strictly enforce non-overlap.
        kept = []
        for idx in selected_idx:
            c = points_texture_snapshot[int(idx)]
            if not kept:
                kept.append(int(idx))
            else:
                karr = points_texture_snapshot[np.asarray(kept, dtype=np.int64)]
                d2 = np.sum((karr - c[None, :]) ** 2, axis=1)
                if np.all(d2 >= min_dist2):
                    kept.append(int(idx))
        return np.asarray(kept, dtype=np.int64)

    def compute_dot_textures(dot_count):
        k = max(1, int(dot_count))
        center_idx = sample_random_nonoverlap_center_indices(k, min_center_dist)
        centers_tex = points_texture_snapshot[center_idx]

        min_d2_tex = np.full(n_pts, np.inf, dtype=np.float32)
        center_chunk = 64
        for c0 in range(0, centers_tex.shape[0], center_chunk):
            c1 = min(c0 + center_chunk, centers_tex.shape[0])
            d = points_texture_snapshot[:, None, :] - centers_tex[None, c0:c1, :]
            d2 = np.sum(d * d, axis=2)
            min_d2_tex = np.minimum(min_d2_tex, np.min(d2, axis=1))
        d_tex = np.sqrt(min_d2_tex)
        ramp_tex = (dot_radius + edge_softness - d_tex) / (2.0 * edge_softness)
        tex_b = smoothstep01(ramp_tex).astype(np.float32)

        if fixed_view:
            centers_view = uv_texture_snapshot[center_idx]
            min_d2_shading = np.full(uv_shading_snapshot.shape[0], np.inf, dtype=np.float32)
            for c0 in range(0, centers_view.shape[0], center_chunk):
                c1 = min(c0 + center_chunk, centers_view.shape[0])
                d = uv_shading_snapshot[:, None, :] - centers_view[None, c0:c1, :]
                d2 = np.sum(d * d, axis=2)
                min_d2_shading = np.minimum(min_d2_shading, np.min(d2, axis=1))
            d_shading = np.sqrt(min_d2_shading)
            ramp_shading = (dot_radius + edge_softness - d_shading) / (2.0 * edge_softness)
            tex_a = smoothstep01(ramp_shading).astype(np.float32)
        else:
            tex_a = tex_b.copy()
        return tex_b, tex_a

    base_rgb = np.array(
        [int(BASE_SURFACE_COLOR[i : i + 2], 16) for i in (1, 3, 5)],
        dtype=np.float32,
    ) / 255.0

    def dots_to_lit_rgb(dot_scalar):
        dot = np.clip(np.asarray(dot_scalar, dtype=np.float32), 0.0, 1.0)
        rgb = (1.0 - dot[:, None]) * base_rgb[None, :]
        return np.round(255.0 * rgb).astype(np.uint8)

    tex_b, tex_a = compute_dot_textures(dot_count=160)
    textured_panel2 = surf_texture.copy(deep=True)
    textured_panel2["tex"] = tex_b.copy()
    textured_panel3 = surf_shading.copy(deep=True)
    textured_panel3["tex"] = tex_a.copy()
    combined_mesh = surf_shading.copy(deep=True)
    combined_mesh["dot_rgb"] = dots_to_lit_rgb(tex_a)
    panel2_title = "Texture Only (Shape B)"

    pv.set_plot_theme("document")
    plotter = pv.Plotter(shape=(1, 4), window_size=PANEL_WINDOW_SIZE, border=False)
    # SSAO is intentionally disabled for panel mode so panel 4 (which has an
    # extra overlay actor) keeps the same shading brightness as panel 1.

    def set_panel_camera(col, camera_position):
        plotter.subplot(0, col)
        plotter.camera_position = camera_position
        if fixed_view:
            if hasattr(plotter, "enable_parallel_projection"):
                plotter.enable_parallel_projection()
            cam = getattr(plotter, "camera", None)
            if cam is not None:
                if hasattr(cam, "parallel_projection"):
                    cam.parallel_projection = True
                elif hasattr(cam, "SetParallelProjection"):
                    cam.SetParallelProjection(True)
                if fixed_parallel_scale is not None:
                    if hasattr(cam, "parallel_scale"):
                        cam.parallel_scale = float(fixed_parallel_scale)
                    elif hasattr(cam, "SetParallelScale"):
                        cam.SetParallelScale(float(fixed_parallel_scale))

    # 1) Shading only
    plotter.subplot(0, 0)
    plotter.add_text("Shading Only (Shape A)", font_size=15)
    actor_shading = plotter.add_mesh(
        surf_shading,
        color=BASE_SURFACE_COLOR,
        smooth_shading=True,
        ambient=0.12,
        diffuse=0.82,
        specular=0.0,
        specular_power=24,
        show_scalar_bar=False,
    )
    lights_shading = configure_fixed_light(plotter)
    if fixed_view:
        target_camera = fixed_camera
        set_panel_camera(0, target_camera)
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)
        target_camera = tuple(plotter.camera_position)

    # 2) Texture-only on Shape B (native geometry).
    plotter.subplot(0, 1)
    plotter.add_text(panel2_title, font_size=15)
    plotter.add_mesh(
        textured_panel2,
        scalars="tex",
        cmap=TEXTURE_CMAP,
        clim=[0, 1],
        lighting=False,
        show_scalar_bar=False,
    )
    set_panel_camera(1, target_camera)

    # 3) Texture cue from B transferred to A
    plotter.subplot(0, 2)
    plotter.add_text("Texture Cue (from B) on Shape A", font_size=15)
    plotter.add_mesh(
        textured_panel3,
        scalars="tex",
        cmap=TEXTURE_CMAP,
        clim=[0, 1],
        lighting=False,
        show_scalar_bar=False,
    )
    set_panel_camera(2, target_camera)

    # 4) Combined cue on A
    plotter.subplot(0, 3)
    plotter.add_text("Combined Cues on Shape A", font_size=15)
    actor_both_shading = plotter.add_mesh(
        combined_mesh,
        scalars="dot_rgb",
        rgb=True,
        smooth_shading=True,
        ambient=0.12,
        diffuse=0.82,
        specular=0.0,
        specular_power=24,
        show_scalar_bar=False,
    )
    lights_both = configure_fixed_light(plotter)
    set_panel_camera(3, target_camera)

    shading_levels = [("Low", 0.00), ("Med", 0.35), ("High", 0.70)]
    # Low stays sparse; increase Med->High gap for clearer perceptual difference.
    texture_levels = [("Low", 20), ("Med", 70), ("High", 120)]

    shade_state = {"level_idx": 1, "strength": shading_levels[1][1]}
    tex_state = {"level_idx": 1, "dot_count": texture_levels[1][1]}
    control_state = {"syncing_buttons": False}
    shade_buttons = []
    tex_buttons = []

    def refresh_textures():
        tex_b_new, tex_a_new = compute_dot_textures(tex_state["dot_count"])
        textured_panel2["tex"] = tex_b_new
        textured_panel3["tex"] = tex_a_new
        combined_mesh["dot_rgb"] = dots_to_lit_rgb(tex_a_new)
        plotter.render()

    def set_button_group(buttons, active_idx):
        control_state["syncing_buttons"] = True
        for i, btn in enumerate(buttons):
            rep = btn.GetRepresentation()
            if rep is not None and hasattr(rep, "SetState"):
                rep.SetState(1 if i == active_idx else 0)
        control_state["syncing_buttons"] = False

    def set_shading_level_idx(idx):
        idx = int(np.clip(idx, 0, 2))
        shade_state["level_idx"] = idx
        shade_state["strength"] = shading_levels[idx][1]
        t = shade_state["strength"]
        # Higher directional contrast for clearer shape-from-shading.
        actor_shading.prop.ambient = 0.10 - 0.08 * t
        actor_shading.prop.diffuse = 0.62 + 0.24 * t
        actor_shading.prop.specular = 0.0
        actor_shading.prop.specular_power = 10.0 + 28.0 * t
        actor_both_shading.prop.ambient = actor_shading.prop.ambient
        actor_both_shading.prop.diffuse = actor_shading.prop.diffuse
        actor_both_shading.prop.specular = actor_shading.prop.specular
        actor_both_shading.prop.specular_power = actor_shading.prop.specular_power
        for lights in (lights_shading, lights_both):
            lights["key"].intensity = 0.42 + 0.40 * t
            lights["sky"].intensity = 0.06 + 0.08 * t
            lights["bounce"].intensity = 0.04 + 0.06 * t
        set_button_group(shade_buttons, idx)
        refresh_textures()

    def set_texture_level_idx(idx):
        idx = int(np.clip(idx, 0, 2))
        tex_state["level_idx"] = idx
        tex_state["dot_count"] = int(texture_levels[idx][1])
        set_button_group(tex_buttons, idx)
        refresh_textures()

    def make_shading_button_callback(idx):
        def _callback(_flag):
            if control_state["syncing_buttons"]:
                return
            set_shading_level_idx(idx)

        return _callback

    def make_texture_button_callback(idx):
        def _callback(_flag):
            if control_state["syncing_buttons"]:
                return
            set_texture_level_idx(idx)

        return _callback

    # Widgets are viewport-local; anchor coordinates within the right-most panel.
    plotter.subplot(0, 3)
    shade_origin = (28, 78)
    tex_origin = (28, 24)
    level_dx = 86
    button_size = 22

    plotter.add_text("Shading Level", position=(shade_origin[0], shade_origin[1] + 34), font_size=11)
    for i, (name, _) in enumerate(shading_levels):
        x = shade_origin[0] + i * level_dx
        y = shade_origin[1]
        btn = plotter.add_checkbox_button_widget(
            callback=make_shading_button_callback(i),
            value=(i == shade_state["level_idx"]),
            position=(x, y),
            size=button_size,
        )
        shade_buttons.append(btn)
        plotter.add_text(name, position=(x + 28, y + 2), font_size=10)

    plotter.add_text("Texture Level", position=(tex_origin[0], tex_origin[1] + 34), font_size=11)
    for i, (name, _) in enumerate(texture_levels):
        x = tex_origin[0] + i * level_dx
        y = tex_origin[1]
        btn = plotter.add_checkbox_button_widget(
            callback=make_texture_button_callback(i),
            value=(i == tex_state["level_idx"]),
            position=(x, y),
            size=button_size,
        )
        tex_buttons.append(btn)
        plotter.add_text(name, position=(x + 28, y + 2), font_size=10)

    set_shading_level_idx(shade_state["level_idx"])
    set_texture_level_idx(tex_state["level_idx"])

    # View controls: synced navigation by default, with optional independent edits.
    view_state = {"sync": True}

    def reset_views():
        for col in range(4):
            set_panel_camera(col, target_camera)
        if view_state["sync"]:
            plotter.link_views()
        plotter.render()

    def set_sync_views(flag):
        view_state["sync"] = bool(flag)
        if view_state["sync"]:
            plotter.link_views()
        else:
            plotter.unlink_views()
        plotter.render()

    def on_reset_click(_flag):
        reset_views()

    plotter.subplot(0, 3)
    plotter.add_checkbox_button_widget(
        callback=set_sync_views,
        value=True,
        position=(30, 182),
        size=26,
    )
    plotter.add_text("Sync Views", position=(65, 184), font_size=11)

    plotter.add_checkbox_button_widget(
        callback=on_reset_click,
        value=False,
        position=(30, 138),
        size=26,
    )
    plotter.add_text("Reset Views", position=(65, 140), font_size=11)
    plotter.add_key_event("r", reset_views)

    def _foreground_center_and_square(panel_rgb, ignore_bottom_frac=1.0, box_margin=1.12):
        ph, pw = panel_rgb.shape[:2]
        search_h = max(1, int(float(ignore_bottom_frac) * ph))
        search = panel_rgb[:search_h, :, :]
        border = np.concatenate(
            [search[0, :, :], search[-1, :, :], search[:, 0, :], search[:, -1, :]],
            axis=0,
        )
        bg = np.median(border, axis=0).astype(np.int16)
        diff = np.max(np.abs(search.astype(np.int16) - bg[None, None, :]), axis=2)
        mask = diff > 8
        if np.any(mask):
            ys, xs = np.where(mask)
            cx = 0.5 * (float(np.min(xs)) + float(np.max(xs)))
            cy = 0.5 * (float(np.min(ys)) + float(np.max(ys)))
            w_box = float(np.max(xs) - np.min(xs) + 1)
            h_box = float(np.max(ys) - np.min(ys) + 1)
            side = int(np.ceil(float(box_margin) * max(w_box, h_box)))
        else:
            cx = 0.5 * float(pw)
            cy = 0.5 * float(search_h)
            side = int(0.92 * min(pw, ph))
        side = max(64, min(side, pw, ph))
        cx = float(np.clip(cx, 0.5 * side, pw - 0.5 * side))
        cy = float(np.clip(cy, 0.5 * side, ph - 0.5 * side))
        return cx, cy, side

    def _crop_square(panel_rgb, cx, cy, side):
        ph, pw = panel_rgb.shape[:2]
        x_start = int(round(cx - 0.5 * side))
        y_start = int(round(cy - 0.5 * side))
        x_start = max(0, min(x_start, pw - side))
        y_start = max(0, min(y_start, ph - side))
        x_end = x_start + side
        y_end = y_start + side
        return np.ascontiguousarray(panel_rgb[y_start:y_end, x_start:x_end, :])

    def _to_gray_u8(img):
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            gray = arr
        else:
            gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        return np.clip(np.round(gray), 0, 255).astype(np.uint8)

    def save_panel1_panel2_product(auto=False):
        stimuli_dir = Path("stimuli")
        stimuli_dir.mkdir(parents=True, exist_ok=True)

        shade_name, shade_strength = shading_levels[shade_state["level_idx"]]
        tex_name, tex_count = texture_levels[tex_state["level_idx"]]
        stem = (
            f"panel12_shading-{shade_name.lower()}-{shade_strength:.2f}"
            f"_texture-{tex_name.lower()}-{int(tex_count)}"
        )
        p1_path = stimuli_dir / f"{stem}_panel1.png"
        p2_path = stimuli_dir / f"{stem}_panel2.png"
        mult_path = stimuli_dir / f"{stem}_mult.png"
        i = 2
        while p1_path.exists() or p2_path.exists() or mult_path.exists():
            p1_path = stimuli_dir / f"{stem}_{i}_panel1.png"
            p2_path = stimuli_dir / f"{stem}_{i}_panel2.png"
            mult_path = stimuli_dir / f"{stem}_{i}_mult.png"
            i += 1

        plotter.render()
        frame = plotter.screenshot(return_img=True)
        if frame is None:
            raise RuntimeError("Could not capture screenshot image from current plotter.")

        h, w = frame.shape[:2]
        panel_w = max(1, w // 4)
        panel1 = np.ascontiguousarray(frame[:, 0:panel_w, :])
        panel2 = np.ascontiguousarray(frame[:, panel_w : 2 * panel_w, :])

        c1x, c1y, s1 = _foreground_center_and_square(panel1, ignore_bottom_frac=1.0, box_margin=1.12)
        c2x, c2y, s2 = _foreground_center_and_square(panel2, ignore_bottom_frac=1.0, box_margin=1.12)
        c1y += float(panel_capture_y_shift_px)
        c2y += float(panel_capture_y_shift_px)
        side = max(s1, s2)
        side = max(64, min(side, panel1.shape[0], panel1.shape[1], panel2.shape[0], panel2.shape[1]))

        panel1_square = _crop_square(panel1, c1x, c1y, side)
        panel2_square = _crop_square(panel2, c2x, c2y, side)

        g1 = _to_gray_u8(panel1_square)
        g2 = _to_gray_u8(panel2_square)
        product = np.round((g1.astype(np.float32) / 255.0) * (g2.astype(np.float32) / 255.0) * 255.0).astype(np.uint8)

        try:
            from PIL import Image

            Image.fromarray(panel1_square).save(str(p1_path))
            Image.fromarray(panel2_square).save(str(p2_path))
            Image.fromarray(product, mode="L").save(str(mult_path))
            prefix = "Auto-saved" if auto else "Saved"
            print(
                f"{prefix} panel 1/panel 2 crops and multiplied grayscale image:\n"
                f"  {p1_path}\n  {p2_path}\n  {mult_path}"
            )
        except Exception as exc:
            np.save(str(p1_path.with_suffix(".npy")), panel1_square)
            np.save(str(p2_path.with_suffix(".npy")), panel2_square)
            np.save(str(mult_path.with_suffix(".npy")), product)
            print(f"Panel 1/2 product save as PNG failed ({exc}); saved NPY files in ./stimuli.")

    def save_rightmost_panel_stimulus():
        stimuli_dir = Path("stimuli")
        stimuli_dir.mkdir(parents=True, exist_ok=True)

        shade_name, shade_strength = shading_levels[shade_state["level_idx"]]
        tex_name, tex_count = texture_levels[tex_state["level_idx"]]
        stem = (
            f"panel4_shading-{shade_name.lower()}-{shade_strength:.2f}"
            f"_texture-{tex_name.lower()}-{int(tex_count)}"
        )
        out_path = stimuli_dir / f"{stem}.png"
        i = 2
        while out_path.exists():
            out_path = stimuli_dir / f"{stem}_{i}.png"
            i += 1

        # Safe path: capture current window once, crop right-most panel, and save.
        # Avoiding a second Plotter here prevents VTK re-entrancy crashes on some systems.
        plotter.render()
        frame = plotter.screenshot(return_img=True)
        if frame is None:
            raise RuntimeError("Could not capture screenshot image from current plotter.")

        h, w = frame.shape[:2]
        panel_w = max(1, w // 4)
        x0 = w - panel_w
        panel4 = np.ascontiguousarray(frame[:, x0:w, :])

        cx, cy, side = _foreground_center_and_square(panel4, ignore_bottom_frac=0.86, box_margin=1.10)
        panel4_square = _crop_square(panel4, cx, cy, side)

        try:
            from PIL import Image

            Image.fromarray(panel4_square).save(str(out_path))
            print(f"Saved panel 4 stimulus to: {out_path}")
        except Exception as exc:
            fallback = stimuli_dir / f"{stem}_full.png"
            plotter.screenshot(str(fallback))
            print(f"Panel-only save failed ({exc}); saved full screenshot to: {fallback}")

    set_sync_views(True)
    if PANEL_ROTATION_ENABLED:
        if hasattr(plotter, "enable_trackball_style"):
            plotter.enable_trackball_style()
    else:
        if hasattr(plotter, "enable_image_style"):
            plotter.enable_image_style()

    # Press "s" to save full 4-panel screenshot.
    # Press "p" to save panel 4 only into ./stimuli with level info in filename.
    out = "shape_shading_texture_comparison.png"
    plotter.add_key_event("s", lambda: plotter.screenshot(out))
    plotter.add_key_event("p", save_rightmost_panel_stimulus)
    plotter.add_key_event("m", lambda: save_panel1_panel2_product(auto=False))
    auto_capture_state = {"done": False}

    def _auto_capture_once(_step=None):
        if auto_capture_state["done"]:
            return
        auto_capture_state["done"] = True
        try:
            save_panel1_panel2_product(auto=True)
        except Exception as exc:
            # Keep the app usable even if timer/render timing varies by backend.
            auto_capture_state["done"] = False
            print(f'Auto-save after launch was not ready ({exc}). Press "m" once the window is visible.')

    if hasattr(plotter, "add_timer_event"):
        try:
            plotter.add_timer_event(max_steps=1, duration=250, callback=_auto_capture_once)
        except Exception as exc:
            print(f'Could not schedule auto panel capture ({exc}). Press "m" to save manually.')
    else:
        print('Auto panel capture timer unavailable in this backend. Press "m" to save manually.')
    print(f'Interactive panel ready. Press "s" for full screenshot: {out}')
    print('Press "p" to save panel 4 stimulus into ./stimuli.')
    print('Press "m" to save panel 1/panel 2 crops and their multiplied image into ./stimuli.')
    plotter.show()


def run_interactive_view():
    surf = make_perturbed_sphere(seed=5)
    pts = surf.points
    scalar = paper_dot_texture(
        pts,
        n_dots=340,
        radii=(0.066, 0.066, 0.066),  # isotropic volume dots (equal radii)
        axis=(0.0, 0.0, 1.0),
        edge_softness=0.045,
        seed=11,
        aa_samples=7,
        aa_radius=0.0035,
        placement="farthest",
    )

    m = surf.copy(deep=True)
    m["tex"] = scalar

    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=INTERACTIVE_WINDOW_SIZE)
    enable_gi_approx(plotter)
    if hasattr(plotter, "enable_anti_aliasing"):
        plotter.enable_anti_aliasing("ssaa")
    plotter.add_text("Drag to rotate | Scroll to zoom | Use slider for shading strength", font_size=12)
    plotter.add_axes()
    actor = plotter.add_mesh(
        m,
        scalars="tex",
        cmap=TEXTURE_CMAP,
        clim=[0, 1],
        smooth_shading=True,
        ambient=0.12,
        diffuse=0.82,
        specular=0.0,
        specular_power=24,
        show_scalar_bar=False,
    )
    lights = configure_fixed_light(plotter)

    def set_shading_strength(value):
        # 0.0 -> flatter lighting, 0.7 -> stronger shape-from-shading cues
        t = float(np.clip(value, 0.0, 0.7))
        actor.prop.ambient = 0.10 - 0.08 * t
        actor.prop.diffuse = 0.62 + 0.24 * t
        actor.prop.specular = 0.0
        actor.prop.specular_power = 10.0 + 28.0 * t
        lights["key"].intensity = 0.42 + 0.40 * t
        lights["sky"].intensity = 0.06 + 0.08 * t
        lights["bounce"].intensity = 0.04 + 0.06 * t
        plotter.render()

    plotter.add_slider_widget(
        callback=set_shading_strength,
        rng=[0.0, 0.7],
        value=0.35,
        title="Shading Strength",
        pointa=(0.03, 0.08),
        pointb=(0.35, 0.08),
    )
    set_shading_strength(0.35)
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.3)
    plotter.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", action="store_true", help="Render a 4-panel cue comparison.")
    parser.add_argument(
        "--fixed-view",
        dest="fixed_view",
        action="store_true",
        default=True,
        help="Lock camera/silhouette so panel 2 and panel 3 dot placement matches exactly (default in --panel mode).",
    )
    parser.add_argument(
        "--free-view",
        dest="fixed_view",
        action="store_false",
        help="Allow free camera and unconstrained geometry (panel 2/3 dots may diverge).",
    )
    parser.add_argument(
        "--shape-diff",
        type=float,
        default=PANEL_SHAPE_DIFF_STRENGTH,
        help=(
            "Strength of A/B shape difference in --panel mode while preserving silhouette in fixed view. "
            "0.0=similar, 1.0=clear difference, >1.0=stronger."
        ),
    )
    args = parser.parse_args()

    if args.panel:
        run_panel_render(fixed_view=args.fixed_view, shape_diff_strength=args.shape_diff)
    else:
        run_interactive_view()


if __name__ == "__main__":
    main()
