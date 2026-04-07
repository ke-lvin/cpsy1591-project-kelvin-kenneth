"""
Interactive textured 3D shape with fixed lighting.

Install:
  pip install pyvista numpy

Run (interactive viewer):
  python texture_gen.py

Optional static panel render:
  python texture_gen.py --panel
"""

import argparse
import numpy as np
import pyvista as pv

BW_CMAP = ["white", "black"]
BASE_SURFACE_COLOR = "#ededed"
TEXTURE_CMAP = [BASE_SURFACE_COLOR, "black"]
INTERACTIVE_WINDOW_SIZE = (1600, 1200)
PANEL_WINDOW_SIZE = (1800, 1040)
SHAPE_THETA_RES = 560
SHAPE_PHI_RES = 560


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


def make_conflicting_surface_pair(theta_res=300, phi_res=300):
    """
    Build two same-topology surfaces with intentionally different geometry:
    - surface A drives shading
    - surface B defines where texture dots live
    Texture can then be transferred exactly by vertex index.
    """
    # Smooth-but-broad shapes: low frequency + higher amplitude.
    surf_a = make_perturbed_sphere(seed=5, theta_res=theta_res, phi_res=phi_res, amp=0.30, n_waves=11, k_range=(1.0, 3.6))
    surf_b = make_perturbed_sphere(seed=29, theta_res=theta_res, phi_res=phi_res, amp=0.42, n_waves=16, k_range=(1.2, 5.0))

    # Extra smoothing removes small ridges while keeping large undulations.
    surf_a = surf_a.smooth(n_iter=36, relaxation_factor=0.06, feature_smoothing=False)
    surf_b = surf_b.smooth(n_iter=44, relaxation_factor=0.055, feature_smoothing=False)

    # Rotate + mirror + shear texture source so texture gradients disagree strongly.
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

    # Add one broad outward bulge toward the viewer (+Z in our fixed-view setup).
    n = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = np.clip(n @ forward, -1.0, 1.0)
    # Gaussian on sphere: strongest at front center, smooth falloff.
    bulge = 0.26 * np.exp(-((1.0 - dot) ** 2) / (2.0 * (0.30**2)))
    pts = pts * (1.0 + bulge[:, None])
    pts /= np.max(np.linalg.norm(pts, axis=1))
    surf_b.points = pts

    if surf_a.n_points != surf_b.n_points:
        raise RuntimeError("Conflicting surfaces must have matching point counts for exact transfer.")
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
    # Conflicting cues:
    # - shading from surface A
    # - dots generated on surface B coordinates
    # - dot values transferred exactly to surface A by matching vertex indices
    surf_shading, surf_texture = make_conflicting_surface_pair(theta_res=300, phi_res=300)
    points_texture_snapshot = surf_texture.points

    def texture_strength_to_count(texture_strength):
        # 0 -> sparse (weaker cue), 1 -> dense (stronger cue)
        t = float(np.clip(texture_strength, 0.0, 1.0))
        return int(round(120 + 220 * t))

    # Random non-overlapping center sampling each update (non-grid, no fixed anchors).
    n_pts = points_texture_snapshot.shape[0]
    rng = np.random.default_rng(1234)
    dot_radius = 0.078
    edge_softness = 0.010
    # Strict non-overlap (including soft edges).
    min_center_dist = 2.08 * (dot_radius + edge_softness)
    texture_coords = points_texture_snapshot.copy()
    fixed_camera = [(0.0, 0.0, 3.1), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    if fixed_view:
        uv_b = project_points_to_view(points_texture_snapshot, fixed_camera[0], fixed_camera[1], fixed_camera[2])
        uv_a = project_points_to_view(surf_shading.points, fixed_camera[0], fixed_camera[1], fixed_camera[2])
        # Use only front-visible source points for snapshot transfer.
        b_tmp = surf_texture.copy(deep=True)
        b_tmp.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=True)
        b_normals = b_tmp.point_normals
        a_tmp = surf_shading.copy(deep=True)
        a_tmp.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=True)
        a_normals = a_tmp.point_normals
        cam = np.asarray(fixed_camera[0], dtype=np.float32)
        view_vec_b = cam[None, :] - points_texture_snapshot
        view_vec_a = cam[None, :] - surf_shading.points
        front_vis_b = np.sum(b_normals * view_vec_b, axis=1) > 0.0
        front_vis_a = np.sum(a_normals * view_vec_a, axis=1) > 0.0

    def sample_random_nonoverlap_centers(k, min_center_dist):
        min_dist2 = float(min_center_dist**2)
        selected = []

        # A few passes keep it fast while still filling target count reliably.
        for _ in range(3):
            need = max(0, k - len(selected))
            if need == 0:
                break
            candidate_count = min(n_pts, max(4000, 28 * need))
            cand_idx = rng.choice(n_pts, size=candidate_count, replace=False)
            candidates = points_texture_snapshot[cand_idx]
            for c in candidates:
                if not selected:
                    selected.append(c)
                else:
                    s = np.asarray(selected, dtype=np.float32)
                    d2 = np.sum((s - c[None, :]) ** 2, axis=1)
                    if np.all(d2 >= min_dist2):
                        selected.append(c)
                if len(selected) >= k:
                    break

        if not selected:
            return points_texture_snapshot[rng.choice(n_pts, size=1, replace=False)]
        centers = np.asarray(selected, dtype=np.float32)

        # Safety cleanup: strictly enforce non-overlap.
        kept = []
        for c in centers:
            if not kept:
                kept.append(c)
            else:
                karr = np.asarray(kept, dtype=np.float32)
                d2 = np.sum((karr - c[None, :]) ** 2, axis=1)
                if np.all(d2 >= min_dist2):
                    kept.append(c)
        return np.asarray(kept, dtype=np.float32)

    def transfer_snapshot_to_a(tex_b):
        """
        Transfer a frontal 2D snapshot of tex_b (defined on B vertices) onto A.
        Method: depth-aware raster mask (z-buffer style) on B, then bilinear sample on A.
        """
        tex_b = np.asarray(tex_b, dtype=np.float32)
        uv_src = uv_b[front_vis_b]
        tex_src = tex_b[front_vis_b]
        if uv_src.shape[0] == 0:
            return np.zeros(uv_a.shape[0], dtype=np.float32)
        tex_src = (tex_src > 0.5).astype(np.uint8)

        # Normalize snapshot to B's visible silhouette, then remap to A's visible silhouette.
        uv_b_vis = uv_b[front_vis_b]
        uv_a_vis = uv_a[front_vis_a]
        if uv_a_vis.shape[0] == 0:
            uv_a_vis = uv_a

        b_min = np.min(uv_b_vis, axis=0)
        b_max = np.max(uv_b_vis, axis=0)
        a_min = np.min(uv_a_vis, axis=0)
        a_max = np.max(uv_a_vis, axis=0)
        b_span = np.maximum(b_max - b_min, 1e-6)
        a_span = np.maximum(a_max - a_min, 1e-6)

        uv_src_n = (uv_src - b_min[None, :]) / b_span[None, :]
        uv_tgt_n = (uv_a - a_min[None, :]) / a_span[None, :]
        uv_tgt_n = np.clip(uv_tgt_n, 0.0, 1.0)

        grid_w = 1400
        grid_h = 1400
        xb = uv_src_n[:, 0] * (grid_w - 1)
        yb = uv_src_n[:, 1] * (grid_h - 1)
        xi = np.clip(np.rint(xb).astype(np.int32), 0, grid_w - 1)
        yi = np.clip(np.rint(yb).astype(np.int32), 0, grid_h - 1)

        mask_img = np.zeros((grid_h, grid_w), dtype=np.float32)

        # Splat dot pixels as small disks in snapshot space to avoid stipple artifacts.
        dot_idx = np.where(tex_src > 0)[0]
        splat_r = 2  # pixels
        offsets = []
        for oy in range(-splat_r, splat_r + 1):
            for ox in range(-splat_r, splat_r + 1):
                if ox * ox + oy * oy <= splat_r * splat_r:
                    offsets.append((ox, oy))
        for idx in dot_idx:
            x = xi[idx]
            y = yi[idx]
            for ox, oy in offsets:
                xx = x + ox
                yy = y + oy
                if 0 <= xx < grid_w and 0 <= yy < grid_h:
                    mask_img[yy, xx] = 1.0

        # Mild blur then re-threshold yields cleaner solid silhouettes.
        blur = (
            mask_img
            + np.roll(mask_img, 1, axis=0)
            + np.roll(mask_img, -1, axis=0)
            + np.roll(mask_img, 1, axis=1)
            + np.roll(mask_img, -1, axis=1)
        ) / 5.0
        mask_img = (blur > 0.34).astype(np.float32)

        xa = uv_tgt_n[:, 0] * (grid_w - 1)
        ya = uv_tgt_n[:, 1] * (grid_h - 1)
        ax0 = np.floor(xa).astype(np.int32)
        ay0 = np.floor(ya).astype(np.int32)
        ax1 = np.clip(ax0 + 1, 0, grid_w - 1)
        ay1 = np.clip(ay0 + 1, 0, grid_h - 1)
        ax0 = np.clip(ax0, 0, grid_w - 1)
        ay0 = np.clip(ay0, 0, grid_h - 1)

        awx = xa - ax0.astype(np.float32)
        awy = ya - ay0.astype(np.float32)
        a00 = (1.0 - awx) * (1.0 - awy)
        a10 = awx * (1.0 - awy)
        a01 = (1.0 - awx) * awy
        a11 = awx * awy

        cov = (
            a00 * mask_img[ay0, ax0]
            + a10 * mask_img[ay0, ax1]
            + a01 * mask_img[ay1, ax0]
            + a11 * mask_img[ay1, ax1]
        )
        tex_a = (cov > 0.40).astype(np.float32)
        return np.clip(tex_a, 0.0, 1.0).astype(np.float32)

    def compute_texture_pair(texture_strength):
        k = texture_strength_to_count(texture_strength)
        centers = sample_random_nonoverlap_centers(k, min_center_dist)

        min_d2 = np.full(n_pts, np.inf, dtype=np.float32)
        center_chunk = 64
        for c0 in range(0, centers.shape[0], center_chunk):
            c1 = min(c0 + center_chunk, centers.shape[0])
            d = points_texture_snapshot[:, None, :] - centers[None, c0:c1, :]
            d2 = np.sum(d * d, axis=2)
            min_d2 = np.minimum(min_d2, np.min(d2, axis=1))
        d = np.sqrt(min_d2)

        ramp = (dot_radius + edge_softness - d) / (2.0 * edge_softness)
        tex_b = smoothstep01(ramp).astype(np.float32)
        if fixed_view:
            tex_a = transfer_snapshot_to_a(tex_b)
            return tex_b, tex_a
        return tex_b, tex_b

    tex_b, tex_a = compute_texture_pair(texture_strength=0.62)
    textured_b = surf_texture.copy(deep=True)
    textured_b["tex"] = tex_b
    textured_a = surf_shading.copy(deep=True)
    textured_a["tex"] = tex_a

    pv.set_plot_theme("document")
    plotter = pv.Plotter(shape=(1, 3), window_size=PANEL_WINDOW_SIZE, border=False)
    enable_gi_approx(plotter)

    # 1) Shading only
    plotter.subplot(0, 0)
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
    lights_shading = configure_fixed_light(plotter)
    if fixed_view:
        plotter.camera_position = fixed_camera
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)
    target_camera = tuple(plotter.camera_position)

    # 2) Shape B with its corresponding texture
    plotter.subplot(0, 1)
    if fixed_view:
        plotter.add_text("Shape B + Native Texture", font_size=15)
    else:
        plotter.add_text("Shape B + Native Texture", font_size=15)
    actor_b_native = plotter.add_mesh(
        textured_b,
        scalars="tex",
        cmap=TEXTURE_CMAP,
        clim=[0, 1],
        lighting=False,
        show_scalar_bar=False,
    )
    plotter.camera_position = target_camera

    # 3) Texture from B transferred onto A
    plotter.subplot(0, 2)
    plotter.add_text("Shape A + Transferred B Texture", font_size=15)
    actor_both = plotter.add_mesh(
        textured_a,
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
    lights_both = configure_fixed_light(plotter)
    plotter.camera_position = target_camera

    def set_shading_strength(value):
        t = float(np.clip(value, 0.0, 1.0))
        for actor in (actor_shading, actor_b_native, actor_both):
            # Keep a matte/plastic range to avoid clipped white regions.
            actor.prop.ambient = 0.26 - 0.20 * t
            actor.prop.diffuse = 0.54 + 0.34 * t
            actor.prop.specular = 0.015 + 0.085 * t
            actor.prop.specular_power = 10.0 + 28.0 * t
        for lights in (lights_shading, lights_both):
            lights["key"].intensity = 0.45 + 0.50 * t
            lights["sky"].intensity = 0.16 + 0.22 * t
            lights["bounce"].intensity = 0.12 + 0.18 * t
        plotter.render()

    plotter.add_slider_widget(
        callback=set_shading_strength,
        rng=[0.0, 1.0],
        value=0.65,
        title="Shading",
        pointa=(0.03, 0.10),
        pointb=(0.32, 0.10),
    )
    set_shading_strength(0.65)

    tex_state = {"strength": 0.62}

    def update_texture():
        tex_b_new, tex_a_new = compute_texture_pair(tex_state["strength"])
        textured_b["tex"] = tex_b_new
        textured_a["tex"] = tex_a_new
        plotter.render()

    def set_texture_strength(value):
        tex_state["strength"] = float(value)
        update_texture()

    plotter.add_slider_widget(
        callback=set_texture_strength,
        rng=[0.0, 1.0],
        value=tex_state["strength"],
        title="Texture Strength",
        pointa=(0.42, 0.10),
        pointb=(0.88, 0.10),
        interaction_event="end",
    )
    update_texture()

    # View controls: synced navigation by default, with optional independent edits.
    view_state = {"sync": True}

    def reset_views():
        for col in range(3):
            plotter.subplot(0, col)
            plotter.camera_position = target_camera
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

    plotter.add_checkbox_button_widget(
        callback=set_sync_views,
        value=True,
        position=(30, 165),
        size=26,
    )
    plotter.add_text("Sync Views", position=(65, 167), font_size=11)

    plotter.add_checkbox_button_widget(
        callback=on_reset_click,
        value=False,
        position=(30, 120),
        size=26,
    )
    plotter.add_text("Reset Views", position=(65, 122), font_size=11)
    plotter.add_key_event("r", reset_views)
    set_sync_views(True)
    if fixed_view and hasattr(plotter, "enable_image_style"):
        plotter.enable_image_style()

    # Press "s" to save a screenshot after adjusting sliders/camera.
    out = "shape_shading_texture_comparison.png"
    plotter.add_key_event("s", lambda: plotter.screenshot(out))
    print(f'Interactive panel ready. Press "s" to save screenshot to: {out}')
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
    parser.add_argument("--panel", action="store_true", help="Render a 3-panel comparison: shading only, texture only, both.")
    parser.add_argument("--fixed-view", action="store_true", help="Use fixed camera with 2D snapshot texture transfer (no rotation).")
    args = parser.parse_args()

    if args.panel:
        run_panel_render(fixed_view=args.fixed_view)
    else:
        run_interactive_view()


if __name__ == "__main__":
    main()
