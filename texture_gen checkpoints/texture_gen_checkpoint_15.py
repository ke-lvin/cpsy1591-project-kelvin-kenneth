"""
Interactive textured 3D shape with fixed lighting.

Install:
  pip install pyvista numpy

Run (interactive viewer):
  python texture_gen.py

Optional static panel render:
  python texture_gen.py --panel

Optional free-view panel (silhouette lock disabled):
  python texture_gen.py --panel --free-view
"""

import argparse
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


def conformal_surface_dot_texture(
    points,
    n_dots=170,
    dot_radius=0.078,
    edge_softness=0.009,
    seed=1234,
    min_center_dist=None,
):
    """
    Surface-conforming dot mask:
    sample dot centers on the 3D surface, then threshold Euclidean distance
    in object space so the projected dots naturally deform with curvature.
    """
    pts = np.asarray(points, dtype=np.float32)
    n_pts = pts.shape[0]
    ctr = np.mean(pts, axis=0)
    radial = np.linalg.norm(pts - ctr[None, :], axis=1)
    mesh_scale = float(np.percentile(radial, 95))
    if (not np.isfinite(mesh_scale)) or mesh_scale <= 1e-6:
        mesh_scale = float(np.max(radial))
    mesh_scale = max(mesh_scale, 1e-6)

    # Keep radius stable in world units but avoid pathological overfill if mesh is tiny.
    effective_radius = min(float(dot_radius), 0.12 * mesh_scale)

    n_centers = int(np.clip(n_dots, 1, n_pts))
    if min_center_dist is None:
        # Keep discs separated, accounting for any optional soft edge.
        min_center_dist = 2.08 * (effective_radius + max(0.0, float(edge_softness)))
    centers = sample_nonoverlap_centers(
        pts,
        n_centers=n_centers,
        min_dist=float(min_center_dist),
        seed=seed,
        max_trials=800000,
    )
    if centers.shape[0] == 0:
        return np.zeros(n_pts, dtype=np.float32)

    min_d2 = np.full(n_pts, np.inf, dtype=np.float32)
    center_chunk = 64
    for c0 in range(0, centers.shape[0], center_chunk):
        c1 = min(c0 + center_chunk, centers.shape[0])
        d = pts[:, None, :] - centers[None, c0:c1, :]
        d2 = np.sum(d * d, axis=2)
        min_d2 = np.minimum(min_d2, np.min(d2, axis=1))

    d = np.sqrt(min_d2)
    radius = effective_radius
    values = None
    for _ in range(4):
        if edge_softness <= 0:
            values = (d <= radius).astype(np.float32)
        else:
            ramp = (radius + float(edge_softness) - d) / (2.0 * float(edge_softness))
            values = smoothstep01(ramp).astype(np.float32)
        fill = float(np.mean(values))
        if fill <= 0.32:
            break
        # If coverage is too high, shrink radius and recompute.
        radius *= max(0.10, np.sqrt(0.22 / (fill + 1e-12)))
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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
        specular=0.15,
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
):
    """
    Add smooth bump structure only along camera depth (w), preserving (u,v) silhouette.
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

    depth_delta = np.zeros_like(uu, dtype=np.float32)
    for _ in range(max(1, int(n_bumps))):
        cu = rng.uniform(-0.78, 0.78)
        cv = rng.uniform(-0.78, 0.78)
        sigma = rng.uniform(*sigma_range)
        sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32))
        a = sign * float(amp) * rng.uniform(0.55, 1.0) / np.sqrt(max(1, int(n_bumps)))
        g = np.exp(-((uu - cu) ** 2 + (vv - cv) ** 2) / (2.0 * sigma * sigma))
        depth_delta += (a * g).astype(np.float32)

    if center_bulge != 0.0:
        g0 = np.exp(-(uu * uu + vv * vv) / (2.0 * (0.60**2)))
        depth_delta += (float(center_bulge) * g0).astype(np.float32)

    # Keep centroid depth stable while changing internal relief.
    depth_delta -= np.mean(depth_delta).astype(np.float32)
    uvw[:, 2] += depth_delta
    mesh.points = _camera_coords_to_points(uvw, cam_pos, focal, view_up)
    return mesh


def reshape_depth_with_shared_silhouette(
    reference_mesh,
    camera,
    seed=0,
    bump_count=10,
    bump_amp=0.20,
    sphere_weight=0.92,
):
    """
    Build a new depth profile in camera-space while preserving the exact
    image-plane silhouette from `reference_mesh`.
    """
    rng = np.random.default_rng(seed)
    cam_pos, focal, view_up = camera
    uvw = _points_to_camera_coords(reference_mesh.points, cam_pos, focal, view_up)
    u = uvw[:, 0]
    v = uvw[:, 1]
    w_ref = uvw[:, 2]

    us = u / (np.max(np.abs(u)) + 1e-12)
    vs = v / (np.max(np.abs(v)) + 1e-12)
    rr = np.sqrt(us * us + vs * vs)

    # Hemisphere-like depth envelope (strong central bulge, taper near silhouette).
    cap = np.sqrt(np.clip(1.0 - np.minimum(rr, 1.0) ** 2, 0.0, 1.0)).astype(np.float32)

    bumps = np.zeros_like(cap, dtype=np.float32)
    for _ in range(max(1, int(bump_count))):
        cu = rng.uniform(-0.72, 0.72)
        cv = rng.uniform(-0.72, 0.72)
        sigma = rng.uniform(0.20, 0.44)
        a = rng.uniform(-1.0, 1.0) * float(bump_amp) / np.sqrt(max(1, int(bump_count)))
        g = np.exp(-((us - cu) ** 2 + (vs - cv) ** 2) / (2.0 * sigma * sigma))
        bumps += (a * g).astype(np.float32)

    profile = float(sphere_weight) * cap + bumps
    profile -= np.mean(profile).astype(np.float32)
    profile /= np.max(np.abs(profile)) + 1e-12

    ref_center = float(np.mean(w_ref))
    ref_span = float(np.percentile(w_ref, 97.5) - np.percentile(w_ref, 2.5))
    if ref_span <= 1e-6:
        ref_span = float(np.max(w_ref) - np.min(w_ref))
    ref_span = max(ref_span, 1e-6)
    w_new = ref_center + 0.55 * ref_span * profile

    out = uvw.copy()
    out[:, 2] = w_new.astype(np.float32)
    return _camera_coords_to_points(out, cam_pos, focal, view_up)


def make_conflicting_surface_pair(theta_res=300, phi_res=300, fixed_camera=None, seed=None):
    """
    Build two same-topology surfaces with intentionally different geometry:
    - surface A drives shading
    - surface B defines where texture dots live
    In fixed-view mode, A and B share the exact front silhouette.
    """
    rng = np.random.default_rng(seed)
    seed_shape = int(rng.integers(0, 2**31 - 1))
    seed_depth_a = int(rng.integers(0, 2**31 - 1))
    seed_depth_b = int(rng.integers(0, 2**31 - 1))

    # Panel 1: keep the existing perturbed/shaded shape pipeline.
    surf_a = make_perturbed_sphere(
        seed=seed_shape,
        theta_res=theta_res,
        phi_res=phi_res,
        amp=0.62,
        n_waves=24,
        k_range=(0.9, 5.0),
    )
    surf_a = surf_a.smooth(n_iter=16, relaxation_factor=0.056, feature_smoothing=False)
    surf_b = surf_a.copy(deep=True)

    if fixed_camera is not None:
        surf_a = add_depth_bumps_in_view(
            surf_a,
            fixed_camera,
            seed=seed_depth_a,
            n_bumps=24,
            amp=0.32,
            sigma_range=(0.20, 0.44),
            center_bulge=0.14,
        )
        # Panel 2: rebuild depth profile under a shared silhouette.
        surf_b.points = reshape_depth_with_shared_silhouette(
            surf_a,
            fixed_camera,
            seed=seed_depth_b,
            bump_count=12,
            bump_amp=0.18,
            sphere_weight=0.95,
        )
        surf_b = surf_b.smooth(n_iter=8, relaxation_factor=0.04, feature_smoothing=False)
        surf_b = enforce_same_front_silhouette(surf_a, surf_b, fixed_camera)
    else:
        # Free-view fallback: keep B geometrically different from A.
        rot = np.array(
            [
                [0.15, 0.25, 0.96],
                [0.98, -0.08, -0.18],
                [0.12, 0.96, -0.24],
            ],
            dtype=np.float32,
        )
        pts = surf_b.points @ rot.T
        pts[:, 0] *= -1.0
        pts[:, 1] *= -1.0
        pts[:, 2] += 0.18 * pts[:, 0] * pts[:, 1]
        pts /= np.max(np.linalg.norm(pts, axis=1))
        surf_b.points = pts
        surf_b = surf_b.smooth(n_iter=18, relaxation_factor=0.05, feature_smoothing=False)

    if surf_a.n_points != surf_b.n_points:
        raise RuntimeError("Conflicting surfaces must have matching point counts for exact transfer.")
    # Keep exact connectivity match.
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


def run_panel_render(fixed_view=False):
    fixed_camera = [(0.0, 0.0, 3.1), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    surf_shading, surf_texture = make_conflicting_surface_pair(
        theta_res=300,
        phi_res=300,
        fixed_camera=fixed_camera if fixed_view else None,
    )
    textured_panel2 = surf_texture.copy(deep=True)
    dots_panel2 = conformal_surface_dot_texture(
        textured_panel2.points,
        n_dots=140,
        dot_radius=0.078,
        edge_softness=0.009,
        seed=1234,
    )
    dot = np.clip(dots_panel2, 0.0, 1.0)
    base_rgb = np.array(
        [int(BASE_SURFACE_COLOR[i : i + 2], 16) for i in (1, 3, 5)],
        dtype=np.float32,
    ) / 255.0
    rgb = (1.0 - dot[:, None]) * base_rgb[None, :]
    # Use float RGB in [0,1] to avoid dtype/VTK uchar interpretation quirks.
    textured_panel2["dot_rgb"] = rgb.astype(np.float32)

    pv.set_plot_theme("document")
    plotter = pv.Plotter(shape=(1, 2), window_size=PANEL_WINDOW_SIZE, border=False)
    # Detect which subplot index is visually left/right (some builds invert columns).
    def _xmid_for_col(col):
        try:
            plotter.subplot(0, col)
            r = getattr(plotter, "renderer", None)
            if r is not None and hasattr(r, "GetViewport"):
                vp = r.GetViewport()
                return 0.5 * (float(vp[0]) + float(vp[2]))
        except Exception:
            pass
        return float(col)

    xmid0 = _xmid_for_col(0)
    xmid1 = _xmid_for_col(1)
    if xmid0 <= xmid1:
        left_col, right_col = 0, 1
    else:
        left_col, right_col = 1, 0

    if hasattr(plotter, "set_background"):
        for col in (left_col, right_col):
            plotter.subplot(0, col)
            try:
                plotter.set_background("white")
            except Exception:
                pass
    if hasattr(plotter, "enable_anti_aliasing"):
        try:
            plotter.enable_anti_aliasing("ssaa")
        except Exception:
            try:
                plotter.enable_anti_aliasing()
            except Exception:
                pass

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

    # 1) Shading-only cue on shape A (always placed on the visual left).
    plotter.subplot(0, left_col)
    plotter.add_text("Shading Only (Shape A)", font_size=15)
    actor_shading = plotter.add_mesh(
        surf_shading,
        color=BASE_SURFACE_COLOR,
        smooth_shading=True,
        ambient=0.12,
        diffuse=0.82,
        specular=0.10,
        specular_power=24,
        show_scalar_bar=False,
    )
    lights = configure_fixed_light(plotter)
    if fixed_view:
        target_camera = fixed_camera
        set_panel_camera(left_col, target_camera)
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)
        target_camera = tuple(plotter.camera_position)

    # 2) Unshaded dots on shape B (always placed on the visual right).
    plotter.subplot(0, right_col)
    plotter.add_text("Texture Only (Shape B, Conformal Dots)", font_size=15)
    plotter.add_mesh(
        textured_panel2,
        scalars="dot_rgb",
        rgb=True,
        lighting=False,
        show_scalar_bar=False,
    )
    set_panel_camera(right_col, target_camera)

    # Same base material tuning as interactive view.
    t = 0.65
    actor_shading.prop.ambient = 0.26 - 0.20 * t
    actor_shading.prop.diffuse = 0.54 + 0.34 * t
    actor_shading.prop.specular = 0.015 + 0.085 * t
    actor_shading.prop.specular_power = 10.0 + 28.0 * t
    lights["key"].intensity = 0.45 + 0.50 * t
    lights["sky"].intensity = 0.16 + 0.22 * t
    lights["bounce"].intensity = 0.12 + 0.18 * t

    # Keep both panels locked to one camera so drag/rotate/pan stays synchronized.
    linked = False
    if hasattr(plotter, "link_views"):
        try:
            plotter.link_views((0, 1))
            linked = True
        except Exception:
            try:
                plotter.link_views()
                linked = True
            except Exception:
                linked = False
    if not linked:
        try:
            r0 = plotter.renderers[0]
            shared_cam = r0.GetActiveCamera()
            for r in plotter.renderers[1:]:
                r.SetActiveCamera(shared_cam)
        except Exception:
            pass
    if PANEL_ROTATION_ENABLED:
        if hasattr(plotter, "enable_trackball_style"):
            plotter.enable_trackball_style()
    else:
        if hasattr(plotter, "enable_image_style"):
            plotter.enable_image_style()

    out = "shape_shading_texture_2panel.png"
    plotter.add_key_event("s", lambda: plotter.screenshot(out))
    print(f'Interactive panel ready. Press "s" for full screenshot: {out}')
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
        specular=0.10,
        specular_power=24,
        show_scalar_bar=False,
    )
    lights = configure_fixed_light(plotter)

    def set_shading_strength(value):
        # 0 -> flatter lighting, 1 -> stronger shape-from-shading cues
        t = float(np.clip(value, 0.0, 1.0))
        actor.prop.ambient = 0.26 - 0.20 * t
        actor.prop.diffuse = 0.54 + 0.34 * t
        actor.prop.specular = 0.015 + 0.085 * t
        actor.prop.specular_power = 10.0 + 28.0 * t
        lights["key"].intensity = 0.45 + 0.50 * t
        lights["sky"].intensity = 0.16 + 0.22 * t
        lights["bounce"].intensity = 0.12 + 0.18 * t
        plotter.render()

    plotter.add_slider_widget(
        callback=set_shading_strength,
        rng=[0.0, 1.0],
        value=0.65,
        title="Shading Strength",
        pointa=(0.03, 0.08),
        pointb=(0.35, 0.08),
    )
    set_shading_strength(0.65)
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.3)
    plotter.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", action="store_true", help="Render a 2-panel shading vs texture comparison.")
    parser.add_argument(
        "--fixed-view",
        dest="fixed_view",
        action="store_true",
        default=True,
        help="Lock camera/silhouette so panel 1 and panel 2 outlines match exactly (default in --panel mode).",
    )
    parser.add_argument(
        "--free-view",
        dest="fixed_view",
        action="store_false",
        help="Allow free camera and unconstrained geometry for panel mode.",
    )
    args = parser.parse_args()

    if args.panel:
        run_panel_render(fixed_view=args.fixed_view)
    else:
        run_interactive_view()


if __name__ == "__main__":
    main()
