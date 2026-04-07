#!/usr/bin/env python3
"""
Compute Shape A/B midline depth profiles and run a participant matching task.

Usage:
  python midline_depth_task.py
  python midline_depth_task.py --no-task
  python midline_depth_task.py --stimuli-dir stimuli --participant-id p01
  python midline_depth_task.py --stimulus-glob "panel4_*.png"
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None

plt = None
Button = None
_MPL_IMPORT_ERROR = None
_MPL_IMPORT_ATTEMPTED = False

try:
    from texture_gen import _points_to_camera_coords, make_conflicting_surface_pair
except Exception:
    _points_to_camera_coords = None
    make_conflicting_surface_pair = None


FIXED_CAMERA = [(0.0, 0.0, 3.1), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def _require_matplotlib():
    global plt, Button, _MPL_IMPORT_ERROR, _MPL_IMPORT_ATTEMPTED
    if not _MPL_IMPORT_ATTEMPTED:
        _MPL_IMPORT_ATTEMPTED = True
        try:
            import matplotlib.pyplot as _plt
            from matplotlib.widgets import Button as _Button
            plt = _plt
            Button = _Button
            _MPL_IMPORT_ERROR = None
        except Exception as exc:
            _MPL_IMPORT_ERROR = exc
            plt = None
            Button = None

    if plt is not None and Button is not None:
        return
    msg = (
        "Matplotlib could not be imported. "
        "This task requires a working matplotlib + numpy runtime.\n"
        "Common fix: pip install --upgrade matplotlib or use numpy<2 with an older matplotlib build."
    )
    if _MPL_IMPORT_ERROR is not None:
        msg += f"\nOriginal import error: {_MPL_IMPORT_ERROR}"
    raise RuntimeError(msg)


def _read_image(path):
    """
    Read image as array. Prefer matplotlib when available; otherwise fallback to PIL.
    """
    if plt is not None:
        return plt.imread(path)
    if Image is not None:
        with Image.open(path) as im:
            return np.asarray(im)
    raise RuntimeError(
        "No image reader available (matplotlib and PIL are both unavailable)."
    )


def extract_continuous_midline_depth(
    mesh,
    camera,
    n_samples=401,
    band_fraction=0.08,
    smooth_fraction=0.03,
):
    """
    Split by top/bottom in view-space and estimate a continuous depth line at the
    horizontal midline (the split boundary).

    Returns:
      x_norm: normalized horizontal position in [0, 1]
      depth_raw: raw camera-depth values along the midline
      depth_norm_near: normalized depth in [0, 1], where 1 means nearer
      meta: dictionary with split/mask details
    """
    if _points_to_camera_coords is None:
        raise RuntimeError(
            "texture_gen/_points_to_camera_coords is unavailable. "
            "Use --task-mode cross-section, or install runtime deps for five-dot mode."
        )

    uvw = _points_to_camera_coords(mesh.points, *camera).astype(np.float32)
    u = uvw[:, 0]
    v = uvw[:, 1]
    w = uvw[:, 2]

    v_mid = 0.5 * (float(np.min(v)) + float(np.max(v)))
    top_mask = v >= v_mid
    bottom_mask = ~top_mask

    v_span = max(1e-6, float(np.max(v) - np.min(v)))
    band = float(band_fraction) * v_span
    mid_mask = np.abs(v - v_mid) <= band

    # Ensure enough support points for a stable continuous profile.
    if int(np.sum(mid_mask)) < 500:
        k = min(v.shape[0], max(4000, n_samples * 10))
        idx = np.argsort(np.abs(v - v_mid))[:k]
        mid_mask = np.zeros_like(v, dtype=bool)
        mid_mask[idx] = True

    u_mid = u[mid_mask]
    w_mid = w[mid_mask]
    order = np.argsort(u_mid)
    u_mid = u_mid[order]
    w_mid = w_mid[order]

    u_min = float(np.min(u_mid))
    u_max = float(np.max(u_mid))
    u_line = np.linspace(u_min, u_max, int(n_samples), dtype=np.float32)

    sigma = max(1e-6, float(smooth_fraction) * max(1e-6, u_max - u_min))
    depth_line = np.empty_like(u_line)

    chunk = 128
    for i0 in range(0, u_line.shape[0], chunk):
        i1 = min(i0 + chunk, u_line.shape[0])
        du = (u_mid[:, None] - u_line[None, i0:i1]) / sigma
        weights = np.exp(-0.5 * du * du).astype(np.float32)
        denom = np.sum(weights, axis=0) + 1e-12
        depth_line[i0:i1] = np.sum(weights * w_mid[:, None], axis=0) / denom

    d_min = float(np.min(depth_line))
    d_max = float(np.max(depth_line))
    span = max(1e-12, d_max - d_min)
    # Nearer points should be larger numbers for the participant-friendly output.
    depth_norm_near = (d_max - depth_line) / span

    x_norm = (u_line - u_line[0]) / max(1e-12, float(u_line[-1] - u_line[0]))
    meta = {
        "top_count": int(np.sum(top_mask)),
        "bottom_count": int(np.sum(bottom_mask)),
        "midline_support_count": int(np.sum(mid_mask)),
        "v_mid": v_mid,
    }
    return x_norm.astype(np.float32), depth_line.astype(np.float32), depth_norm_near.astype(np.float32), meta


def _extract_depth_profile_at_v_from_uvw(
    uvw,
    v_target,
    n_samples=601,
    band_fraction=0.08,
    smooth_fraction=0.03,
):
    """
    Estimate a continuous depth profile at a chosen horizontal camera-space row (v=v_target).

    Returns:
      u_line: sampled horizontal camera coordinate
      depth_near: absolute depth in near-positive convention (higher = closer)
      meta: support diagnostics
    """
    uvw = np.asarray(uvw, dtype=np.float32)
    u = uvw[:, 0]
    v = uvw[:, 1]
    w = uvw[:, 2]

    v_span = max(1e-6, float(np.max(v) - np.min(v)))
    band = float(band_fraction) * v_span
    mask = np.abs(v - float(v_target)) <= band

    # Keep profile stable even when the selected band is sparse.
    if int(np.sum(mask)) < 500:
        k = min(v.shape[0], max(4000, int(n_samples) * 10))
        idx = np.argsort(np.abs(v - float(v_target)))[:k]
        mask = np.zeros_like(v, dtype=bool)
        mask[idx] = True

    u_sel = u[mask]
    w_sel = w[mask]
    order = np.argsort(u_sel)
    u_sel = u_sel[order]
    w_sel = w_sel[order]

    u_min = float(np.min(u_sel))
    u_max = float(np.max(u_sel))
    u_line = np.linspace(u_min, u_max, int(n_samples), dtype=np.float32)

    sigma = max(1e-6, float(smooth_fraction) * max(1e-6, u_max - u_min))
    depth_w = np.empty_like(u_line)
    chunk = 128
    for i0 in range(0, u_line.shape[0], chunk):
        i1 = min(i0 + chunk, u_line.shape[0])
        du = (u_sel[:, None] - u_line[None, i0:i1]) / sigma
        weights = np.exp(-0.5 * du * du).astype(np.float32)
        denom = np.sum(weights, axis=0) + 1e-12
        depth_w[i0:i1] = np.sum(weights * w_sel[:, None], axis=0) / denom

    # Camera-space w: smaller is nearer for this camera setup.
    # Convert to near-positive so larger values mean closer to observer.
    depth_near = (-depth_w).astype(np.float32)
    meta = {
        "v_target": float(v_target),
        "support_count": int(np.sum(mask)),
        "u_min": u_min,
        "u_max": u_max,
    }
    return u_line, depth_near, meta


def _compute_cross_section_cue_targets(trial, uvw_shading, uvw_texture):
    """
    Compute shading/texture absolute depth targets for one cross-section trial.

    Output depths use near-positive convention (higher = closer).
    Left/right anchor agreement is expected to come from texture_gen geometry.
    """
    uvw_a = np.asarray(uvw_shading, dtype=np.float32)
    uvw_b = np.asarray(uvw_texture, dtype=np.float32)

    y_top = int(trial.get("fg_top_px", 0))
    y_bottom = int(trial.get("fg_bottom_px", max(1, int(trial.get("height_px", 1)) - 1)))
    y_probe = int(trial["probe_y_px"])
    if y_bottom <= y_top:
        y_frac = 0.5
    else:
        y_frac = float((y_probe - y_top) / max(1, (y_bottom - y_top)))
    y_frac = float(np.clip(y_frac, 0.0, 1.0))

    v_min = float(min(np.min(uvw_a[:, 1]), np.min(uvw_b[:, 1])))
    v_max = float(max(np.max(uvw_a[:, 1]), np.max(uvw_b[:, 1])))
    # Image y grows downward while camera v grows upward.
    v_target = v_max - y_frac * (v_max - v_min)

    u_a, depth_a_near, meta_a = _extract_depth_profile_at_v_from_uvw(uvw_a, v_target=v_target)
    u_b, depth_b_near, meta_b = _extract_depth_profile_at_v_from_uvw(uvw_b, v_target=v_target)

    x_left = float(trial["task_x_left_norm"])
    x_right = float(trial["task_x_right_norm"])
    if x_right <= x_left:
        x_right = x_left + 1e-3

    x_inner = np.asarray(trial["task_x_inner_norm"], dtype=np.float32)
    if x_inner.size == 0:
        x_inner = np.linspace(x_left, x_right, 7, dtype=np.float32)[1:-1]
    x_with_anchors = np.r_[np.array([x_left], dtype=np.float32), x_inner, np.array([x_right], dtype=np.float32)]

    t = (x_with_anchors - x_left) / max(1e-12, (x_right - x_left))
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    u_query_a = u_a[0] + t * (u_a[-1] - u_a[0])
    u_query_b = u_b[0] + t * (u_b[-1] - u_b[0])
    depth_a_query = np.interp(u_query_a, u_a, depth_a_near).astype(np.float32)
    depth_b_query = np.interp(u_query_b, u_b, depth_b_near).astype(np.float32)
    left_gap = float(depth_b_query[0] - depth_a_query[0])
    right_gap = float(depth_b_query[-1] - depth_a_query[-1])

    return {
        "x_with_anchors_norm": x_with_anchors.tolist(),
        "x_inner_norm": x_inner.tolist(),
        "probe_v": float(v_target),
        "probe_y_fraction": y_frac,
        "anchor_left_depth_near": float(depth_a_query[0]),
        "anchor_right_depth_near": float(depth_a_query[-1]),
        "anchor_left_depth_near_texture": float(depth_b_query[0]),
        "anchor_right_depth_near_texture": float(depth_b_query[-1]),
        "anchor_left_gap": left_gap,
        "anchor_right_gap": right_gap,
        "shading_depth_near_with_anchors": depth_a_query.tolist(),
        "texture_depth_near_with_anchors": depth_b_query.tolist(),
        "shading_depth_near_inner": depth_a_query[1:-1].tolist(),
        "texture_depth_near_inner": depth_b_query[1:-1].tolist(),
        "meta_shading": meta_a,
        "meta_texture": meta_b,
    }


def build_shape_ab_profiles(seed=1234, n_samples=401):
    if make_conflicting_surface_pair is None:
        raise RuntimeError(
            "texture_gen/make_conflicting_surface_pair is unavailable. "
            "Use --task-mode cross-section, or install runtime deps for five-dot mode."
        )

    surf_a, surf_b = make_conflicting_surface_pair(
        theta_res=300,
        phi_res=300,
        fixed_camera=FIXED_CAMERA,
        seed=seed,
    )

    xa, da_raw, da_norm, meta_a = extract_continuous_midline_depth(
        surf_a,
        FIXED_CAMERA,
        n_samples=n_samples,
    )
    xb, db_raw, db_norm, meta_b = extract_continuous_midline_depth(
        surf_b,
        FIXED_CAMERA,
        n_samples=n_samples,
    )

    return {
        "camera": FIXED_CAMERA,
        "seed": int(seed),
        "n_samples": int(n_samples),
        "shape_a": {
            "x_norm": xa.tolist(),
            "depth_raw": da_raw.tolist(),
            "depth_norm_near": da_norm.tolist(),
            "meta": meta_a,
        },
        "shape_b": {
            "x_norm": xb.tolist(),
            "depth_raw": db_raw.tolist(),
            "depth_norm_near": db_norm.tolist(),
            "meta": meta_b,
        },
    }


def save_profiles(profile_data, out_json, out_csv, out_plot):
    _require_matplotlib()
    out_json = Path(out_json)
    out_csv = Path(out_csv)
    out_plot = Path(out_plot)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(profile_data, indent=2))

    xa = np.asarray(profile_data["shape_a"]["x_norm"], dtype=np.float32)
    da_raw = np.asarray(profile_data["shape_a"]["depth_raw"], dtype=np.float32)
    da_norm = np.asarray(profile_data["shape_a"]["depth_norm_near"], dtype=np.float32)
    xb = np.asarray(profile_data["shape_b"]["x_norm"], dtype=np.float32)
    db_raw = np.asarray(profile_data["shape_b"]["depth_raw"], dtype=np.float32)
    db_norm = np.asarray(profile_data["shape_b"]["depth_norm_near"], dtype=np.float32)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "x_norm", "depth_raw", "depth_norm_near"])
        for i in range(xa.shape[0]):
            w.writerow(["A", float(xa[i]), float(da_raw[i]), float(da_norm[i])])
        for i in range(xb.shape[0]):
            w.writerow(["B", float(xb[i]), float(db_raw[i]), float(db_norm[i])])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xa, da_raw, label="Shape A", linewidth=2.0)
    ax.plot(xb, db_raw, label="Shape B", linewidth=2.0)
    gmin = float(min(np.min(da_raw), np.min(db_raw)))
    gmax = float(max(np.max(da_raw), np.max(db_raw)))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(gmin, gmax)
    ax.set_xlabel("Midline Position (Left to Right)")
    ax.set_ylabel("Absolute Depth (camera space)")
    ax.set_title("Continuous Midline Absolute Depth Profiles")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)

    print(f"Saved Shape A/B midline profiles JSON: {out_json}")
    print(f"Saved Shape A/B midline profiles CSV:  {out_csv}")
    print(f"Saved Shape A/B midline profile plot: {out_plot}")


def sample_profile_at_x(x_src, y_src, x_query):
    x_src = np.asarray(x_src, dtype=np.float32)
    y_src = np.asarray(y_src, dtype=np.float32)
    x_query = np.asarray(x_query, dtype=np.float32)
    return np.interp(x_query, x_src, y_src).astype(np.float32)


def compute_fit_metrics(response, target):
    """
    Return:
      rmse_abs: direct RMSE in same absolute units
      rmse_affine: RMSE after best affine fit response ~= a*target + b
      corr: Pearson correlation (NaN if degenerate)
    """
    y = np.asarray(response, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)

    rmse_abs = float(np.sqrt(np.mean((y - t) ** 2)))

    X = np.column_stack([t, np.ones_like(t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rmse_affine = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    if np.std(y) < 1e-12 or np.std(t) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y, t)[0, 1])

    return {
        "rmse_abs": rmse_abs,
        "rmse_affine": rmse_affine,
        "corr": corr,
        "affine_a": float(beta[0]),
        "affine_b": float(beta[1]),
    }


class MidlineMatchingTask:
    def __init__(self, stimuli_paths, participant_id, responses_dir, profile_data):
        _require_matplotlib()
        self.stimuli_paths = list(stimuli_paths)
        self.participant_id = participant_id
        self.responses_dir = Path(responses_dir)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.profile_data = profile_data

        if not self.stimuli_paths:
            raise ValueError("No stimuli images found.")

        self.n_trials = len(self.stimuli_paths)
        self.x_positions = np.linspace(0.10, 0.90, 5, dtype=np.float32)
        self.shape_a_x = np.asarray(self.profile_data["shape_a"]["x_norm"], dtype=np.float32)
        self.shape_a_depth_raw = np.asarray(self.profile_data["shape_a"]["depth_raw"], dtype=np.float32)
        self.shape_b_x = np.asarray(self.profile_data["shape_b"]["x_norm"], dtype=np.float32)
        self.shape_b_depth_raw = np.asarray(self.profile_data["shape_b"]["depth_raw"], dtype=np.float32)

        self.target_a_5 = sample_profile_at_x(self.shape_a_x, self.shape_a_depth_raw, self.x_positions)
        self.target_b_5 = sample_profile_at_x(self.shape_b_x, self.shape_b_depth_raw, self.x_positions)

        # One shared absolute scale across A and B, reused for all stimuli.
        gmin = float(min(np.min(self.shape_a_depth_raw), np.min(self.shape_b_depth_raw)))
        gmax = float(max(np.max(self.shape_a_depth_raw), np.max(self.shape_b_depth_raw)))
        self.y_min = gmin
        self.y_max = gmax
        self.raw_min = gmin
        self.raw_max = gmax

        # UI depth is mirrored so lower on screen means farther, without upside-down axes.
        self.target_a_5_ui = self._raw_to_ui(self.target_a_5)
        self.target_b_5_ui = self._raw_to_ui(self.target_b_5)

        y0 = 0.5 * (self.y_min + self.y_max)
        self.responses_ui = np.full((self.n_trials, 5), y0, dtype=np.float32)
        self.trial_view_seconds = np.zeros(self.n_trials, dtype=np.float64)

        self.current = 0
        self.drag_idx = None
        self.trial_start_time = time.time()
        self.saved_path = None

        self.fig = plt.figure(figsize=(13.5, 7.5))
        gs = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[1.35, 1.0],
            left=0.04,
            right=0.98,
            top=0.91,
            bottom=0.15,
            wspace=0.12,
        )
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_line = self.fig.add_subplot(gs[0, 1])

        (self.response_line,) = self.ax_line.plot(
            self.x_positions,
            self.responses_ui[self.current],
            "-o",
            color="black",
            linewidth=2.0,
            markersize=10,
            markerfacecolor="white",
        )
        self._style_line_axis()
        self._load_trial()

        ax_prev = self.fig.add_axes([0.55, 0.03, 0.10, 0.06])
        ax_next = self.fig.add_axes([0.67, 0.03, 0.10, 0.06])
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(lambda _evt: self.go_prev())
        self.btn_next.on_clicked(lambda _evt: self.go_next())

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        print(
            "Depth matching scale (shared absolute): "
            f"min={self.y_min:.4f}, max={self.y_max:.4f}"
        )

    def _raw_to_ui(self, raw_depth):
        raw = np.asarray(raw_depth, dtype=np.float32)
        return (self.raw_max + self.raw_min - raw).astype(np.float32)

    def _ui_to_raw(self, ui_depth):
        ui = np.asarray(ui_depth, dtype=np.float32)
        return (self.raw_max + self.raw_min - ui).astype(np.float32)

    def _style_line_axis(self):
        self.ax_line.clear()
        y_mid = 0.5 * (self.y_min + self.y_max)
        self.ax_line.axhline(y_mid, color="#7a7a7a", linewidth=1.2)
        for x in self.x_positions:
            self.ax_line.axvline(float(x), color="#ebebeb", linewidth=1.0, zorder=0)
        self.ax_line.set_xlim(0.0, 1.0)
        # Normal orientation (not upside down). Depth semantics are handled in UI mapping.
        self.ax_line.set_ylim(self.y_min, self.y_max)
        self.ax_line.set_xticks(self.x_positions.tolist())
        self.ax_line.set_xticklabels(["1", "2", "3", "4", "5"])
        self.ax_line.set_xlabel("Left  -->  Right Along the Midline")
        self.ax_line.set_ylabel("Absolute Depth (lower = farther)")
        self.ax_line.set_title("Adjust the 5 dots to match perceived midline depth")
        self.ax_line.grid(axis="y", alpha=0.22)
        (self.response_line,) = self.ax_line.plot(
            self.x_positions,
            self.responses_ui[self.current],
            "-o",
            color="black",
            linewidth=2.0,
            markersize=10,
            markerfacecolor="white",
        )

    def _load_trial(self):
        path = self.stimuli_paths[self.current]
        img = _read_image(path)
        self.ax_img.clear()
        self.ax_img.imshow(img)
        self.ax_img.axis("off")
        self.ax_img.set_title(path.name)

        self.response_line.set_data(self.x_positions, self.responses_ui[self.current])
        self.fig.suptitle(
            f"Participant: {self.participant_id} | Trial {self.current + 1}/{self.n_trials}\n"
            "Drag dots vertically only; dots are evenly spaced horizontally.",
            fontsize=12,
        )
        self.trial_start_time = time.time()
        self.fig.canvas.draw_idle()

    def _close_trial_timer(self):
        dt = max(0.0, time.time() - self.trial_start_time)
        self.trial_view_seconds[self.current] += dt

    def _set_current_dot_y(self, idx, y):
        y_clipped = float(np.clip(y, self.y_min, self.y_max))
        self.responses_ui[self.current, idx] = y_clipped
        self.response_line.set_data(self.x_positions, self.responses_ui[self.current])
        self.fig.canvas.draw_idle()

    def _pick_dot_index_from_event_x(self, event, max_px=20.0):
        if event.inaxes != self.ax_line or event.x is None:
            return None
        pts = np.column_stack([self.x_positions, self.responses_ui[self.current]])
        x_px = self.ax_line.transData.transform(pts)[:, 0]
        dx = np.abs(x_px - float(event.x))
        idx = int(np.argmin(dx))
        if float(dx[idx]) <= float(max_px):
            return idx
        return None

    def on_press(self, event):
        idx = self._pick_dot_index_from_event_x(event)
        if idx is None:
            return
        self.drag_idx = idx
        if event.ydata is not None:
            self._set_current_dot_y(idx, event.ydata)

    def on_motion(self, event):
        if self.drag_idx is None:
            return
        if event.inaxes != self.ax_line or event.ydata is None:
            return
        self._set_current_dot_y(self.drag_idx, event.ydata)

    def on_release(self, _event):
        self.drag_idx = None

    def on_key(self, event):
        if event.key in ("right", "n", "enter"):
            self.go_next()
        elif event.key in ("left", "p", "backspace"):
            self.go_prev()
        elif event.key in ("escape", "q"):
            plt.close(self.fig)

    def go_prev(self):
        self._close_trial_timer()
        if self.current > 0:
            self.current -= 1
        self._load_trial()

    def go_next(self):
        self._close_trial_timer()
        if self.current < self.n_trials - 1:
            self.current += 1
            self._load_trial()
        else:
            self.finish()

    def _build_output_rows(self):
        rows = []
        for i, path in enumerate(self.stimuli_paths):
            resp_ui = self.responses_ui[i, :].astype(np.float32)
            resp_raw = self._ui_to_raw(resp_ui)
            fit_a = compute_fit_metrics(resp_raw, self.target_a_5)
            fit_b = compute_fit_metrics(resp_raw, self.target_b_5)

            best_abs = "A" if fit_a["rmse_abs"] <= fit_b["rmse_abs"] else "B"
            best_affine = "A" if fit_a["rmse_affine"] <= fit_b["rmse_affine"] else "B"

            rows.append(
                {
                    "participant_id": self.participant_id,
                    "trial_index": i + 1,
                    "stimulus_file": path.name,
                    "dot1": float(resp_raw[0]),
                    "dot2": float(resp_raw[1]),
                    "dot3": float(resp_raw[2]),
                    "dot4": float(resp_raw[3]),
                    "dot5": float(resp_raw[4]),
                    "dot1_ui": float(resp_ui[0]),
                    "dot2_ui": float(resp_ui[1]),
                    "dot3_ui": float(resp_ui[2]),
                    "dot4_ui": float(resp_ui[3]),
                    "dot5_ui": float(resp_ui[4]),
                    "trial_view_seconds": float(self.trial_view_seconds[i]),
                    "a_rmse_abs": fit_a["rmse_abs"],
                    "b_rmse_abs": fit_b["rmse_abs"],
                    "a_rmse_affine": fit_a["rmse_affine"],
                    "b_rmse_affine": fit_b["rmse_affine"],
                    "a_corr": fit_a["corr"],
                    "b_corr": fit_b["corr"],
                    "best_match_abs": best_abs,
                    "best_match_affine": best_affine,
                }
            )
        return rows

    def finish(self):
        self._close_trial_timer()
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"midline_task_{self.participant_id}_{ts}"
        out_csv = self.responses_dir / f"{stem}.csv"
        out_json = self.responses_dir / f"{stem}.json"

        rows = self._build_output_rows()
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "participant_id",
                    "trial_index",
                    "stimulus_file",
                    "dot1",
                    "dot2",
                    "dot3",
                    "dot4",
                    "dot5",
                    "dot1_ui",
                    "dot2_ui",
                    "dot3_ui",
                    "dot4_ui",
                    "dot5_ui",
                    "trial_view_seconds",
                    "a_rmse_abs",
                    "b_rmse_abs",
                    "a_rmse_affine",
                    "b_rmse_affine",
                    "a_corr",
                    "b_corr",
                    "best_match_abs",
                    "best_match_affine",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)

        payload = {
            "participant_id": self.participant_id,
            "x_positions": self.x_positions.tolist(),
            "y_range": [self.y_min, self.y_max],
            "targets": {
                "shape_a_depth_raw_at_5dots": self.target_a_5.tolist(),
                "shape_b_depth_raw_at_5dots": self.target_b_5.tolist(),
                "shape_a_depth_ui_at_5dots": self.target_a_5_ui.tolist(),
                "shape_b_depth_ui_at_5dots": self.target_b_5_ui.tolist(),
            },
            "responses": rows,
        }
        out_json.write_text(json.dumps(payload, indent=2))

        self.saved_path = out_csv
        print(f"Saved participant responses CSV:  {out_csv}")
        print(f"Saved participant responses JSON: {out_json}")
        plt.close(self.fig)

    def run(self):
        plt.show()
        return self.saved_path


def list_stimuli_images(stimuli_dir, stimulus_glob="panel4_*.png"):
    stimuli_dir = Path(stimuli_dir)
    if not stimuli_dir.exists():
        return []
    files = [p for p in stimuli_dir.glob(stimulus_glob) if p.is_file()]
    return sorted(files)


def _to_rgb_u8(img):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32)
    if float(np.max(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def estimate_foreground_mask(img_rgb_u8, diff_threshold=8.0):
    img = np.asarray(img_rgb_u8, dtype=np.float32)
    border = np.concatenate(
        [img[0, :, :], img[-1, :, :], img[:, 0, :], img[:, -1, :]],
        axis=0,
    )
    bg = np.median(border, axis=0).astype(np.float32)
    diff = np.max(np.abs(img - bg[None, None, :]), axis=2)
    mask = diff > float(diff_threshold)

    # Light denoise to bridge tiny gaps from texture dots.
    row_hits = np.convolve(mask.sum(axis=1).astype(np.float32), np.ones(5, dtype=np.float32), mode="same")
    col_hits = np.convolve(mask.sum(axis=0).astype(np.float32), np.ones(5, dtype=np.float32), mode="same")
    valid_rows = row_hits > 3.0
    valid_cols = col_hits > 3.0
    mask &= valid_rows[:, None]
    mask &= valid_cols[None, :]

    if not np.any(mask):
        raise RuntimeError("Could not detect foreground contour from image.")
    return mask


def _longest_true_run(row_mask):
    idx = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    if idx.size == 0:
        return None
    split = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[split + 1]]
    ends = np.r_[idx[split], idx[-1]]
    lengths = ends - starts + 1
    j = int(np.argmax(lengths))
    return int(starts[j]), int(ends[j]), int(lengths[j])


def choose_random_probe_row(mask, rng, y_frac_min=1.0 / 3.0, y_frac_max=2.0 / 3.0, min_span_frac=0.14):
    h, w = mask.shape
    y0 = int(np.clip(np.floor(float(y_frac_min) * h), 0, h - 1))
    y1 = int(np.clip(np.ceil(float(y_frac_max) * h), 0, h - 1))
    if y1 < y0:
        y0, y1 = y1, y0

    min_span = max(16, int(round(float(min_span_frac) * w)))
    ys = np.arange(y0, y1 + 1, dtype=np.int32)
    ys = ys[rng.permutation(ys.shape[0])]

    best = None
    for y in ys:
        row = mask[int(y), :]
        # Dilate slightly in 1D to ignore tiny perforations from texture.
        row_s = np.convolve(row.astype(np.float32), np.ones(5, dtype=np.float32), mode="same") >= 1.0
        run = _longest_true_run(row_s)
        if run is None:
            continue
        x_left, x_right, span = run
        if span >= min_span:
            return int(y), int(x_left), int(x_right)
        if best is None or span > best[2]:
            best = (int(y), int(x_left), int(x_right), int(span))

    if best is not None:
        return best[0], best[1], best[2]

    ys_fg, xs_fg = np.where(mask)
    if xs_fg.size == 0:
        raise RuntimeError("Failed to find foreground intersections for probe row.")
    y = int(np.clip(np.median(ys_fg), y0, y1))
    return y, int(np.min(xs_fg)), int(np.max(xs_fg))


def build_cross_section_trial(stimulus_path, rng):
    path = Path(stimulus_path)
    raw_img = _read_image(path)
    img_u8 = _to_rgb_u8(raw_img)
    mask = estimate_foreground_mask(img_u8)
    y_px, x_left_px, x_right_px = choose_random_probe_row(mask, rng)
    ys_fg, _ = np.where(mask)
    fg_top_px = int(np.min(ys_fg))
    fg_bottom_px = int(np.max(ys_fg))

    h, w = mask.shape
    denom = float(max(1, w - 1))
    x_left_norm = float(x_left_px / denom)
    x_right_norm = float(x_right_px / denom)
    if x_right_norm <= x_left_norm:
        x_left_norm, x_right_norm = sorted([x_left_norm, x_right_norm])
        x_right_norm = float(min(0.99, x_right_norm + 0.01))

    # Five movable sample points strictly between fixed left/right anchors.
    x_inner_norm = np.linspace(x_left_norm, x_right_norm, 7, dtype=np.float32)[1:-1]
    x_inner_px = np.rint(x_inner_norm * (w - 1)).astype(np.int32)

    return {
        "path": path,
        # Always display as explicit RGB grayscale to avoid accidental colormaps.
        "img": img_u8,
        "height_px": int(h),
        "width_px": int(w),
        "fg_top_px": fg_top_px,
        "fg_bottom_px": fg_bottom_px,
        "probe_y_px": int(y_px),
        "x_left_px": int(x_left_px),
        "x_right_px": int(x_right_px),
        "x_left_norm": x_left_norm,
        "x_right_norm": x_right_norm,
        # Side task uses same left-right spacing as image intersections.
        "task_x_left_norm": x_left_norm,
        "task_x_right_norm": x_right_norm,
        "task_x_inner_norm": x_inner_norm.tolist(),
        "x_inner_px": x_inner_px.tolist(),
    }


def build_cross_section_trial_from_depth_grid_row(stimulus_path, depth_grid_row, trial_meta=None):
    path = Path(stimulus_path)
    raw_img = _read_image(path)
    img_u8 = _to_rgb_u8(raw_img)

    row = dict(depth_grid_row)

    h, w = img_u8.shape[:2]
    try:
        mask = estimate_foreground_mask(img_u8)
        ys_fg, _ = np.where(mask)
        fg_top_px = int(np.min(ys_fg))
        fg_bottom_px = int(np.max(ys_fg))
    except Exception:
        fg_top_px = 0
        fg_bottom_px = max(1, h - 1)

    y_frac = float(np.clip(row.get("probe_y_fraction", 0.5), 0.0, 1.0))
    if fg_bottom_px <= fg_top_px:
        y_px = int(round(y_frac * max(1, h - 1)))
    else:
        y_px = int(round(fg_top_px + y_frac * (fg_bottom_px - fg_top_px)))
    y_px = int(np.clip(y_px, 0, h - 1))

    x_left_norm = float(row.get("x_left_norm", 0.2))
    x_right_norm = float(row.get("x_right_norm", 0.8))
    if x_right_norm <= x_left_norm:
        x_right_norm = float(min(0.99, x_left_norm + 0.01))
    x_inner_norm = np.asarray(row.get("x_inner_norm", []), dtype=np.float32)
    if x_inner_norm.size != 5:
        x_inner_norm = np.linspace(x_left_norm, x_right_norm, 7, dtype=np.float32)[1:-1]

    x_left_px = int(np.clip(np.rint(x_left_norm * (w - 1)), 0, w - 1))
    x_right_px = int(np.clip(np.rint(x_right_norm * (w - 1)), 0, w - 1))
    x_inner_px = np.clip(np.rint(x_inner_norm * (w - 1)), 0, w - 1).astype(np.int32)

    return {
        "path": path,
        "img": img_u8,
        "height_px": int(h),
        "width_px": int(w),
        "fg_top_px": int(fg_top_px),
        "fg_bottom_px": int(fg_bottom_px),
        "probe_y_px": int(y_px),
        "x_left_px": int(x_left_px),
        "x_right_px": int(x_right_px),
        "x_left_norm": float(x_left_norm),
        "x_right_norm": float(x_right_norm),
        "task_x_left_norm": float(x_left_norm),
        "task_x_right_norm": float(x_right_norm),
        "task_x_inner_norm": x_inner_norm.astype(np.float32).tolist(),
        "x_inner_px": x_inner_px.tolist(),
        "depth_grid_row_index": int(row.get("row_index", -1)),
        "depth_grid_probe_y_fraction": float(y_frac),
        "depth_grid_probe_v": float(row.get("probe_v", float("nan"))),
        "depth_grid_shading_depth_near_with_anchors": list(row.get("shading_depth_near_with_anchors", [])),
        "depth_grid_texture_depth_near_with_anchors": list(row.get("texture_depth_near_with_anchors", [])),
        "trial_meta": dict(trial_meta) if trial_meta is not None else {},
    }


def build_cross_section_trial_from_depth_grid(stimulus_path, rng, depth_grid_rows, trial_meta=None):
    if not depth_grid_rows:
        raise RuntimeError("Depth grid has no rows.")
    idx = int(rng.integers(0, len(depth_grid_rows)))
    return build_cross_section_trial_from_depth_grid_row(
        stimulus_path=stimulus_path,
        depth_grid_row=depth_grid_rows[idx],
        trial_meta=trial_meta,
    )


def _load_combo_images_for_shape_dir(shape_dir):
    d = Path(shape_dir)
    if not d.exists() or not d.is_dir():
        raise RuntimeError(f'Shape directory not found: "{shape_dir}"')
    combos = {}
    for p in sorted(d.glob("*.png")):
        code = p.stem.strip().upper()
        if len(code) == 2 and code[0] in "ABC" and code[1] in "ABC":
            combos[code] = p
    return combos


def _arrange_trials_no_adjacent_duplicates(trials, key_fn, rng=None, randomize=True):
    """
    Reorder trials so no two adjacent trials share the same key.
    """
    import heapq

    groups = {}
    for t in trials:
        k = str(key_fn(t))
        groups.setdefault(k, []).append(t)
    for items in groups.values():
        if rng is not None and randomize:
            rng.shuffle(items)

    heap = []
    for k, items in groups.items():
        tie = float(rng.random()) if (rng is not None and randomize) else 0.0
        heapq.heappush(heap, (-len(items), tie, k))

    out = []
    last_key = None
    while heap:
        count1, _tie1, key1 = heapq.heappop(heap)
        if key1 == last_key:
            if not heap:
                raise RuntimeError("Could not arrange trials without adjacent duplicates.")
            count2, _tie2, key2 = heapq.heappop(heap)
            out.append(groups[key2].pop())
            last_key = key2
            count2 += 1
            if count2 < 0:
                tie = float(rng.random()) if (rng is not None and randomize) else 0.0
                heapq.heappush(heap, (count2, tie, key2))
            heapq.heappush(heap, (count1, _tie1, key1))
        else:
            out.append(groups[key1].pop())
            last_key = key1
            count1 += 1
            if count1 < 0:
                tie = float(rng.random()) if (rng is not None and randomize) else 0.0
                heapq.heappush(heap, (count1, tie, key1))
    return out


def _validate_two_shape_experiment_trials(
    trials,
    shape_dirs,
    combo_order,
    repeats,
    require_diff_shape_lines=True,
):
    expected_total = int(len(shape_dirs) * len(combo_order) * repeats)
    if len(trials) != expected_total:
        raise RuntimeError(
            f"Trial count mismatch: expected {expected_total}, got {len(trials)}."
        )

    counts = {}
    rows_by_shape = {}
    yfrac_by_shape = {}
    for t in trials:
        meta = t.get("trial_meta", {})
        shape_label = str(meta.get("shape_label", ""))
        combo = str(meta.get("combo_code", ""))
        key = (shape_label, combo)
        counts[key] = counts.get(key, 0) + 1

        if shape_label:
            rows_by_shape.setdefault(shape_label, set()).add(int(meta.get("depth_grid_row_index", -1)))
            yfrac_by_shape.setdefault(shape_label, set()).add(float(t.get("depth_grid_probe_y_fraction", float("nan"))))

    missing = []
    bad_counts = []
    for shape_i in range(1, len(shape_dirs) + 1):
        shape_label = f"shape{shape_i}"
        for combo in combo_order:
            c = counts.get((shape_label, combo), 0)
            if c == 0:
                missing.append(f"{shape_label}:{combo}")
            elif c != repeats:
                bad_counts.append(f"{shape_label}:{combo}={c}")
    if missing:
        raise RuntimeError(
            "Missing shape/combo trial(s): " + ", ".join(missing)
        )
    if bad_counts:
        raise RuntimeError(
            "Unexpected repeats per shape/combo: " + ", ".join(bad_counts)
        )

    for shape_label, rows in rows_by_shape.items():
        if len(rows) != 1:
            raise RuntimeError(
                f"{shape_label} does not use exactly one depth-grid row (rows={sorted(rows)})."
            )

    if require_diff_shape_lines and len(shape_dirs) == 2:
        y1 = next(iter(yfrac_by_shape.get("shape1", {float("nan")})))
        y2 = next(iter(yfrac_by_shape.get("shape2", {float("nan")})))
        if np.isfinite(y1) and np.isfinite(y2):
            if abs(float(y1) - float(y2)) < 1e-9:
                raise RuntimeError(
                    "shape1 and shape2 selected the same probe line (probe_y_fraction)."
                )
        else:
            r1 = next(iter(rows_by_shape.get("shape1", {-1})))
            r2 = next(iter(rows_by_shape.get("shape2", {-1})))
            if int(r1) == int(r2):
                raise RuntimeError(
                    "shape1 and shape2 selected the same depth-grid row index."
                )

    for i in range(1, len(trials)):
        prev = str(Path(trials[i - 1]["path"]).resolve())
        curr = str(Path(trials[i]["path"]).resolve())
        if prev == curr:
            raise RuntimeError(
                f"Adjacent duplicate stimulus detected at positions {i} and {i + 1}: {curr}"
            )


def build_two_shape_experiment_trials(
    shape_dirs,
    seed=1234,
    repeats_per_combo=4,
    exclude_codes=("AA",),
    shuffle=False,
):
    shape_dirs = [Path(p) for p in shape_dirs]
    if len(shape_dirs) != 2:
        raise RuntimeError("Expected exactly two shape directories for experiment mode.")

    repeats = max(1, int(repeats_per_combo))
    exclude = {str(c).upper() for c in exclude_codes}
    combo_order = [f"{a}{b}" for a in "ABC" for b in "ABC" if f"{a}{b}" not in exclude]
    if not combo_order:
        raise RuntimeError("No stimulus combinations selected for experiment.")

    rng = np.random.default_rng(seed)
    trials = []
    summary = []
    selected_shape_probe_y_fracs = []
    for shape_i, shape_dir in enumerate(shape_dirs, start=1):
        combos = _load_combo_images_for_shape_dir(shape_dir)
        missing = [c for c in combo_order if c not in combos]
        if missing:
            raise RuntimeError(
                f'Shape directory "{shape_dir}" is missing combo image(s): {", ".join(missing)}'
            )

        depth_grid_path = discover_depth_grid_json(list(combos.values()), explicit_path=str(shape_dir / "depth_grid.json"))
        depth_grid_data = load_depth_grid_metadata(depth_grid_path)
        payload = depth_grid_data.get("payload", {})
        shading_levels = payload.get("shading_levels", [])
        texture_levels = payload.get("texture_levels", [])
        rows = depth_grid_data["rows"]

        # One fixed row per shape, but force different line positions across the two shapes.
        row_idx = int(rng.integers(0, len(rows)))
        if selected_shape_probe_y_fracs:
            target_frac = float(selected_shape_probe_y_fracs[0])
            min_sep = 0.06
            if np.isfinite(target_frac):
                y_fracs = np.asarray(
                    [float(r.get("probe_y_fraction", float("nan"))) for r in rows],
                    dtype=np.float32,
                )
                finite_mask = np.isfinite(y_fracs)
                valid_idx = np.flatnonzero(finite_mask)
                if valid_idx.size > 0:
                    sep = np.abs(y_fracs[valid_idx] - target_frac)
                    good = valid_idx[sep >= float(min_sep)]
                    if good.size > 0:
                        row_idx = int(good[int(rng.integers(0, good.size))])
                    else:
                        # If no row clears min_sep, pick the farthest available row.
                        row_idx = int(valid_idx[int(np.argmax(sep))])
                elif len(rows) > 1:
                    # Fallback: at least avoid same row index when fractions are unavailable.
                    row_idx = int((int(rng.integers(1, len(rows))) + row_idx) % len(rows))
            elif len(rows) > 1:
                # Fallback when shape1 row fraction is invalid.
                row_idx = int((int(rng.integers(1, len(rows))) + row_idx) % len(rows))

        row = rows[row_idx]
        selected_shape_probe_y_fracs.append(float(row.get("probe_y_fraction", float("nan"))))

        def _level_lookup(levels, code):
            idx = max(0, min(2, ord(str(code).upper()) - ord("A")))
            if isinstance(levels, list) and idx < len(levels):
                entry = levels[idx]
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    return str(entry[0]), float(entry[1])
                if isinstance(entry, dict):
                    name = str(entry.get("name", entry.get("label", str(code))))
                    value = float(entry.get("value", float("nan")))
                    return name, value
            default_name = {"A": "Low", "B": "Med", "C": "High"}.get(str(code).upper(), str(code))
            return default_name, float("nan")

        shape_label = f"shape{shape_i}"
        summary.append(
            {
                "shape_label": shape_label,
                "shape_dir": str(shape_dir),
                "depth_grid_json": str(depth_grid_data["path"]),
                "depth_grid_row_index": int(row.get("row_index", row_idx)),
                "depth_grid_probe_y_fraction": float(row.get("probe_y_fraction", float("nan"))),
                "combos": list(combo_order),
                "repeats_per_combo": repeats,
            }
        )

        for combo in combo_order:
            for rep in range(1, repeats + 1):
                s_name, s_value = _level_lookup(shading_levels, combo[0])
                t_name, t_value = _level_lookup(texture_levels, combo[1])
                trial_meta = {
                    "shape_label": shape_label,
                    "shape_index": int(shape_i),
                    "shape_dir": str(shape_dir),
                    "combo_code": combo,
                    "shading_level_code": combo[0],
                    "shading_level_name": s_name,
                    "shading_level_value": float(s_value),
                    "texture_level_code": combo[1],
                    "texture_level_name": t_name,
                    "texture_level_value": float(t_value),
                    "repeat_index": int(rep),
                    "depth_grid_json": str(depth_grid_data["path"]),
                    "depth_grid_row_index": int(row.get("row_index", row_idx)),
                }
                t = build_cross_section_trial_from_depth_grid_row(
                    stimulus_path=combos[combo],
                    depth_grid_row=row,
                    trial_meta=trial_meta,
                )
                trials.append(t)

    if shuffle:
        rng.shuffle(trials)

    trials = _arrange_trials_no_adjacent_duplicates(
        trials,
        key_fn=lambda t: str(Path(t["path"]).resolve()),
        rng=rng,
        randomize=bool(shuffle),
    )

    for i, t in enumerate(trials, start=1):
        t.setdefault("trial_meta", {})
        t["trial_meta"]["experiment_trial_index"] = int(i)

    _validate_two_shape_experiment_trials(
        trials=trials,
        shape_dirs=shape_dirs,
        combo_order=combo_order,
        repeats=repeats,
        require_diff_shape_lines=True,
    )

    return trials, summary


def _cue_targets_from_depth_grid_trial(trial):
    shading = np.asarray(trial.get("depth_grid_shading_depth_near_with_anchors", []), dtype=np.float32)
    texture = np.asarray(trial.get("depth_grid_texture_depth_near_with_anchors", []), dtype=np.float32)
    if shading.size != 7 or texture.size != 7:
        raise RuntimeError("Depth grid row must contain exactly 7 shading and 7 texture depth values.")

    x_left = float(trial["task_x_left_norm"])
    x_right = float(trial["task_x_right_norm"])
    x_inner = np.asarray(trial["task_x_inner_norm"], dtype=np.float32)
    if x_inner.size != 5:
        x_inner = np.linspace(x_left, x_right, 7, dtype=np.float32)[1:-1]
    x_with_anchors = np.r_[np.array([x_left], dtype=np.float32), x_inner, np.array([x_right], dtype=np.float32)]

    return {
        "x_with_anchors_norm": x_with_anchors.tolist(),
        "x_inner_norm": x_inner.tolist(),
        "probe_v": float(trial.get("depth_grid_probe_v", float("nan"))),
        "probe_y_fraction": float(trial.get("depth_grid_probe_y_fraction", float("nan"))),
        "anchor_left_depth_near": float(shading[0]),
        "anchor_right_depth_near": float(shading[-1]),
        "anchor_left_depth_near_texture": float(texture[0]),
        "anchor_right_depth_near_texture": float(texture[-1]),
        "anchor_left_gap": float(texture[0] - shading[0]),
        "anchor_right_gap": float(texture[-1] - shading[-1]),
        "shading_depth_near_with_anchors": shading.tolist(),
        "texture_depth_near_with_anchors": texture.tolist(),
        "shading_depth_near_inner": shading[1:-1].tolist(),
        "texture_depth_near_inner": texture[1:-1].tolist(),
        "meta_shading": {"source": "depth_grid_json"},
        "meta_texture": {"source": "depth_grid_json"},
    }


def load_depth_grid_metadata(json_path):
    p = Path(json_path)
    payload = json.loads(p.read_text())
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f'Depth grid JSON "{p}" has no rows.')

    valid_rows = []
    for i, r in enumerate(rows):
        shading = np.asarray(r.get("shading_depth_near_with_anchors", []), dtype=np.float32).reshape(-1)
        texture = np.asarray(r.get("texture_depth_near_with_anchors", []), dtype=np.float32).reshape(-1)
        if shading.size != 7 or texture.size != 7:
            continue
        row = dict(r)
        row["row_index"] = int(row.get("row_index", i))
        row["probe_y_fraction"] = float(np.clip(row.get("probe_y_fraction", 0.5), 0.0, 1.0))
        row["x_left_norm"] = float(row.get("x_left_norm", 0.2))
        row["x_right_norm"] = float(row.get("x_right_norm", 0.8))
        row["x_inner_norm"] = np.asarray(row.get("x_inner_norm", []), dtype=np.float32).reshape(-1).tolist()
        row["probe_v"] = float(row.get("probe_v", float("nan")))
        row["shading_depth_near_with_anchors"] = shading.tolist()
        row["texture_depth_near_with_anchors"] = texture.tolist()
        valid_rows.append(row)

    if not valid_rows:
        raise RuntimeError(f'Depth grid JSON "{p}" has no valid 7-point rows.')

    return {"path": str(p), "payload": payload, "rows": valid_rows}


def discover_depth_grid_json(stimuli_paths, explicit_path=""):
    if explicit_path:
        p = Path(explicit_path)
        if p.exists() and p.is_file():
            return p
        raise RuntimeError(f'Depth grid JSON not found: "{explicit_path}"')

    candidates = []
    dirs = sorted({str(Path(p).parent) for p in stimuli_paths})
    for d in dirs:
        base = Path(d)
        candidates.extend([base / "depth_grid.json", base / "depth_grid_latest.json"])
    candidates.extend([Path("stimuli") / "depth_grid.json", Path("stimuli") / "depth_grid_latest.json"])

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def save_cross_section_preview(trial, out_path):
    _require_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(trial["img"])
    ax.axhline(float(trial["probe_y_px"]), color="red", linewidth=1.8, alpha=0.85)
    inner_x = np.asarray(trial["x_inner_px"], dtype=np.float32)
    if inner_x.size > 0:
        ax.scatter(
            inner_x,
            np.full(inner_x.shape[0], float(trial["probe_y_px"]), dtype=np.float32),
            s=42,
            c="red",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    ax.scatter(
        [float(trial["x_left_px"]), float(trial["x_right_px"])],
        [float(trial["probe_y_px"]), float(trial["probe_y_px"])],
        s=82,
        c="red",
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    ax.axis("off")
    ax.set_title(
        f'{trial["path"].name} | probe y={trial["probe_y_px"]} | '
        f'xL={trial["x_left_px"]}, xR={trial["x_right_px"]}'
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


class CrossSectionFiveDotTask:
    def __init__(
        self,
        stimuli_paths,
        participant_id,
        responses_dir,
        seed=1234,
        depth_grid_data=None,
        prebuilt_trials=None,
        allow_back=True,
        experiment_info=None,
    ):
        _require_matplotlib()
        self.stimuli_paths = [Path(p) for p in stimuli_paths]
        self.participant_id = participant_id
        self.responses_dir = Path(responses_dir)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.allow_back = bool(allow_back)
        self.experiment_info = dict(experiment_info) if experiment_info is not None else {}

        if prebuilt_trials is None and not self.stimuli_paths:
            raise ValueError("No stimuli images found.")

        self.rng = np.random.default_rng(seed)
        self.depth_grid_data = depth_grid_data if depth_grid_data is not None else None
        self.depth_grid_path = str(self.depth_grid_data["path"]) if self.depth_grid_data is not None else ""

        if prebuilt_trials is not None:
            self.trials = list(prebuilt_trials)
            if not self.trials:
                raise ValueError("No prebuilt trials provided.")
        elif self.depth_grid_data is not None:
            self.trials = [
                build_cross_section_trial_from_depth_grid(p, self.rng, self.depth_grid_data["rows"])
                for p in self.stimuli_paths
            ]
        else:
            self.trials = [build_cross_section_trial(p, self.rng) for p in self.stimuli_paths]

        self.n_trials = len(self.trials)
        self.cue_profile_seed = int(seed)
        self.cue_targets = [None for _ in range(self.n_trials)]
        self.cue_targets_available = False

        if prebuilt_trials is not None:
            try:
                self.cue_targets = [_cue_targets_from_depth_grid_trial(t) for t in self.trials]
                self.cue_targets_available = True
                if not self.depth_grid_path:
                    paths = sorted(
                        {
                            str(t.get("trial_meta", {}).get("depth_grid_json", "")).strip()
                            for t in self.trials
                            if str(t.get("trial_meta", {}).get("depth_grid_json", "")).strip()
                        }
                    )
                    if len(paths) == 1:
                        self.depth_grid_path = paths[0]
                print(
                    "Loaded cue targets from prebuilt trials "
                    f"(n_trials={self.n_trials}, forward_only={not self.allow_back})."
                )
            except Exception as exc:
                print(
                    "Warning: could not extract cue targets from prebuilt trials "
                    f"({exc})."
                )
        elif self.depth_grid_data is not None:
            try:
                self.cue_targets = [_cue_targets_from_depth_grid_trial(t) for t in self.trials]
                self.cue_targets_available = True
                print(
                    "Loaded shading/texture depth targets from saved depth grid JSON "
                    f"for cross-section mode ({self.depth_grid_path})."
                )
            except Exception as exc:
                print(
                    "Warning: could not use saved depth grid JSON targets "
                    f"({exc}). Falling back to geometry recomputation."
                )
                self.depth_grid_data = None
                self.depth_grid_path = ""
                self.trials = [build_cross_section_trial(p, self.rng) for p in self.stimuli_paths]
                self.n_trials = len(self.trials)
                self.cue_targets = [None for _ in range(self.n_trials)]

        if (prebuilt_trials is None) and (self.depth_grid_data is None):
            if _points_to_camera_coords is not None and make_conflicting_surface_pair is not None:
                try:
                    surf_a, surf_b = make_conflicting_surface_pair(
                        theta_res=300,
                        phi_res=300,
                        fixed_camera=FIXED_CAMERA,
                        seed=self.cue_profile_seed,
                    )
                    uvw_a = _points_to_camera_coords(surf_a.points, *FIXED_CAMERA).astype(np.float32)
                    uvw_b = _points_to_camera_coords(surf_b.points, *FIXED_CAMERA).astype(np.float32)
                    self.cue_targets = [_compute_cross_section_cue_targets(t, uvw_a, uvw_b) for t in self.trials]
                    self.cue_targets_available = True
                    max_anchor_gap = max(
                        max(abs(float(c["anchor_left_gap"])), abs(float(c["anchor_right_gap"]))) for c in self.cue_targets
                    )
                    print(
                        "Computed shading/texture absolute depth targets for cross-section mode "
                        f"(near-positive, geometry-native anchors, seed={self.cue_profile_seed}, "
                        f"max_anchor_gap={max_anchor_gap:.6f})."
                    )
                except Exception as exc:
                    print(
                        "Warning: could not compute shading/texture depth targets from texture_gen "
                        f"geometry ({exc})."
                    )
            else:
                print(
                    "Warning: texture_gen geometry helpers unavailable; shading/texture depth "
                    "targets were not computed."
                )

        self._print_true_depth_targets_startup()

        self.n_movable = 5
        if self.cue_targets_available:
            all_depths = []
            for cue in self.cue_targets:
                all_depths.extend(cue["shading_depth_near_with_anchors"])
                all_depths.extend(cue["texture_depth_near_with_anchors"])
            d_min = float(np.min(all_depths))
            d_max = float(np.max(all_depths))
            d_span = max(1e-4, d_max - d_min)
            pad = 0.12 * d_span
            self.y_min = d_min - pad
            self.y_max = d_max + pad
        else:
            self.y_min = -1.0
            self.y_max = 1.0

        self.y_mid = 0.5 * (self.y_min + self.y_max)

        self.responses_ui = np.zeros((self.n_trials, self.n_movable), dtype=np.float32)
        for i in range(self.n_trials):
            # UI initialization only: start movable dots centered vertically.
            # True cue-depth targets remain unchanged and are still saved in output.
            self.responses_ui[i, :] = np.float32(self.y_mid)
        self.trial_view_seconds = np.zeros(self.n_trials, dtype=np.float64)

        self.current = 0
        self.drag_idx = None
        self.current_x = np.linspace(0.30, 0.70, self.n_movable, dtype=np.float32)
        self._syncing_anchor_spacing = False
        self.trial_start_time = time.time()
        self.saved_path = None

        self.fig = plt.figure(figsize=(13.5, 7.5))
        gs = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[1.35, 1.0],
            left=0.04,
            right=0.98,
            top=0.91,
            bottom=0.15,
            wspace=0.12,
        )
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_line = self.fig.add_subplot(gs[0, 1])
        self.response_line = None

        self._load_trial()

        self.btn_prev = None
        if self.allow_back:
            ax_prev = self.fig.add_axes([0.55, 0.03, 0.10, 0.06])
            self.btn_prev = Button(ax_prev, "Previous")
            self.btn_prev.on_clicked(lambda _evt: self.go_prev())
            ax_next = self.fig.add_axes([0.67, 0.03, 0.10, 0.06])
        else:
            ax_next = self.fig.add_axes([0.61, 0.03, 0.14, 0.06])
        self.btn_next = Button(ax_next, "Next")
        self.btn_next.on_clicked(lambda _evt: self.go_next())

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

    @staticmethod
    def _format_depth_values(values):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        return "[" + ", ".join(f"{float(v):.6f}" for v in arr.tolist()) + "]"

    def _print_true_depth_targets_startup(self):
        if not self.cue_targets_available:
            print(
                "True shading/texture depth values unavailable at startup "
                "(cue-depth targets were not computed)."
            )
            return
        print(
            "True shading/texture depth values at startup "
            "(7 values each, near-positive, includes fixed anchors):"
        )
        for i, (trial, cue) in enumerate(zip(self.trials, self.cue_targets), start=1):
            shading_vals = np.asarray(cue["shading_depth_near_with_anchors"], dtype=np.float32)
            texture_vals = np.asarray(cue["texture_depth_near_with_anchors"], dtype=np.float32)
            print(f"  Trial {i:02d} | {trial['path'].name}")
            print(f"    shading_depths_7: {self._format_depth_values(shading_vals)}")
            print(f"    texture_depths_7: {self._format_depth_values(texture_vals)}")

    def _draw_image_panel(self):
        t = self.trials[self.current]
        self.ax_img.clear()
        self.ax_img.imshow(t["img"])

        y0 = float(t["probe_y_px"])
        x_inner = np.asarray(t["x_inner_px"], dtype=np.float32)
        if x_inner.size > 0:
            self.ax_img.scatter(
                x_inner,
                np.full(x_inner.shape[0], y0, dtype=np.float32),
                s=42,
                c="red",
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
        self.ax_img.scatter(
            [float(t["x_left_px"]), float(t["x_right_px"])],
            [y0, y0],
            s=82,
            c="red",
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        self.ax_img.axis("off")
        self.ax_img.set_title("")

    def _set_line_xlim_to_match_anchor_pixel_spacing(self, t, x_left, x_right):
        """
        Make graph anchor spacing (in display pixels) equal to image anchor spacing.
        """
        # Ensure transforms/bboxes reflect current artists before measuring.
        self.fig.canvas.draw()

        p_left_img = self.ax_img.transData.transform((float(t["x_left_px"]), float(t["probe_y_px"])))
        p_right_img = self.ax_img.transData.transform((float(t["x_right_px"]), float(t["probe_y_px"])))
        target_px = abs(float(p_right_img[0] - p_left_img[0]))

        dx = float(x_right - x_left)
        line_width_px = float(self.ax_line.bbox.width)
        if target_px <= 1e-6 or line_width_px <= 1e-6 or dx <= 1e-9:
            self.ax_line.set_xlim(0.0, 1.0)
            return

        required_span = (dx * line_width_px) / target_px
        # Keep anchors visible while preserving exact spacing as much as possible.
        required_span = max(required_span, dx + 1e-6)
        center = 0.5 * (float(x_left) + float(x_right))
        self.ax_line.set_xlim(center - 0.5 * required_span, center + 0.5 * required_span)

    def _sync_anchor_spacing_current_trial(self):
        if self._syncing_anchor_spacing:
            return
        if self.current < 0 or self.current >= self.n_trials:
            return

        t = self.trials[self.current]
        x_left = float(t["task_x_left_norm"])
        x_right = float(t["task_x_right_norm"])

        self._syncing_anchor_spacing = True
        try:
            self._set_line_xlim_to_match_anchor_pixel_spacing(t, x_left, x_right)
        finally:
            self._syncing_anchor_spacing = False

    def on_resize(self, _event):
        # Keep image/graph anchor spacing matched under dynamic window resize.
        self._sync_anchor_spacing_current_trial()
        self.fig.canvas.draw_idle()

    def _style_line_axis(self):
        t = self.trials[self.current]
        x_inner = np.asarray(t["task_x_inner_norm"], dtype=np.float32)
        if x_inner.size != self.n_movable:
            x_inner = np.linspace(
                float(t["task_x_left_norm"]),
                float(t["task_x_right_norm"]),
                self.n_movable + 2,
                dtype=np.float32,
            )[1:-1]
        self.current_x = x_inner
        x_left = float(t["task_x_left_norm"])
        x_right = float(t["task_x_right_norm"])
        # UI display: keep fixed endpoints on the centered horizontal line.
        y_left = float(self.y_mid)
        y_right = float(self.y_mid)

        self.ax_line.clear()
        self.ax_line.axhline(float(self.y_mid), color="#7a7a7a", linewidth=1.2, linestyle="--")
        for x in np.r_[np.array([x_left], dtype=np.float32), self.current_x, np.array([x_right], dtype=np.float32)]:
            self.ax_line.axvline(float(x), color="#ebebeb", linewidth=1.0, zorder=0)

        # Endpoints are fixed anchors (not draggable).
        self.ax_line.scatter(
            [x_left, x_right],
            [y_left, y_right],
            s=86,
            c="red",
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )

        self.ax_line.set_xlim(0.0, 1.0)
        self._sync_anchor_spacing_current_trial()
        self.ax_line.set_ylim(self.y_min, self.y_max)
        self.ax_line.set_xticks([])
        self.ax_line.set_yticks([])
        self.ax_line.set_xlabel("")
        self.ax_line.set_ylabel("")
        self.ax_line.set_title("")
        self.ax_line.grid(False)
        for spine in self.ax_line.spines.values():
            spine.set_visible(False)

        y = self.responses_ui[self.current, :]
        (self.response_line,) = self.ax_line.plot(
            self.current_x,
            y,
            linestyle="None",
            marker="o",
            color="red",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="red",
        )

    def _load_trial(self):
        self._draw_image_panel()
        self._style_line_axis()
        self.fig.suptitle(
            "Fixed endpoints and movable dots start on the centered line. Drag 5 interior dots.\n"
            "Nearer = higher; farther = lower.",
            fontsize=12,
        )
        self.trial_start_time = time.time()
        self.fig.canvas.draw_idle()

    def _close_trial_timer(self):
        dt = max(0.0, time.time() - self.trial_start_time)
        self.trial_view_seconds[self.current] += dt

    def _set_current_dot_y(self, idx, y):
        y_clipped = float(np.clip(y, self.y_min, self.y_max))
        self.responses_ui[self.current, idx] = y_clipped
        self.response_line.set_data(self.current_x, self.responses_ui[self.current, :])
        self.fig.canvas.draw_idle()

    def _pick_dot_index_from_event_x(self, event, max_px=20.0):
        if event.inaxes != self.ax_line or event.x is None:
            return None
        pts = np.column_stack([self.current_x, self.responses_ui[self.current, :]])
        x_px = self.ax_line.transData.transform(pts)[:, 0]
        dx = np.abs(x_px - float(event.x))
        idx = int(np.argmin(dx))
        if float(dx[idx]) <= float(max_px):
            return idx
        return None

    def on_press(self, event):
        idx = self._pick_dot_index_from_event_x(event)
        if idx is None:
            return
        self.drag_idx = idx
        if event.ydata is not None:
            self._set_current_dot_y(idx, event.ydata)

    def on_motion(self, event):
        if self.drag_idx is None:
            return
        if event.inaxes != self.ax_line or event.ydata is None:
            return
        self._set_current_dot_y(self.drag_idx, event.ydata)

    def on_release(self, _event):
        self.drag_idx = None

    def on_key(self, event):
        if event.key in ("right", "n", "enter", " ", "space"):
            self.go_next()
        elif self.allow_back and event.key in ("left", "p", "backspace"):
            self.go_prev()
        elif event.key in ("escape", "q"):
            plt.close(self.fig)

    def go_prev(self):
        if not self.allow_back:
            return
        self._close_trial_timer()
        if self.current > 0:
            self.current -= 1
        self._load_trial()

    def go_next(self):
        self._close_trial_timer()
        if self.current < self.n_trials - 1:
            self.current += 1
            self._load_trial()
        else:
            self.finish()

    def _build_output_rows(self):
        rows = []
        for i, t in enumerate(self.trials):
            resp = self.responses_ui[i, :].astype(np.float32)
            x_inner = np.asarray(t["task_x_inner_norm"], dtype=np.float32)
            cue = self.cue_targets[i] if i < len(self.cue_targets) else None
            if cue is None:
                shading_inner = np.full(self.n_movable, np.nan, dtype=np.float32)
                texture_inner = np.full(self.n_movable, np.nan, dtype=np.float32)
                anchor_left = float("nan")
                anchor_right = float("nan")
                anchor_left_texture = float("nan")
                anchor_right_texture = float("nan")
                anchor_left_gap = float("nan")
                anchor_right_gap = float("nan")
                probe_v = float("nan")
                probe_y_frac = float("nan")
            else:
                shading_inner = np.asarray(cue["shading_depth_near_inner"], dtype=np.float32)
                texture_inner = np.asarray(cue["texture_depth_near_inner"], dtype=np.float32)
                anchor_left = float(cue["anchor_left_depth_near"])
                anchor_right = float(cue["anchor_right_depth_near"])
                anchor_left_texture = float(cue["anchor_left_depth_near_texture"])
                anchor_right_texture = float(cue["anchor_right_depth_near_texture"])
                anchor_left_gap = float(cue["anchor_left_gap"])
                anchor_right_gap = float(cue["anchor_right_gap"])
                probe_v = float(cue["probe_v"])
                probe_y_frac = float(cue["probe_y_fraction"])

            row = {
                "participant_id": self.participant_id,
                "trial_index": i + 1,
                "stimulus_file": t["path"].name,
                "depth_grid_json": str(
                    t.get("trial_meta", {}).get("depth_grid_json", self.depth_grid_path if self.depth_grid_path else "")
                ),
                "depth_grid_row_index": int(t["depth_grid_row_index"]) if "depth_grid_row_index" in t else -1,
                "depth_grid_probe_y_fraction": float(t["depth_grid_probe_y_fraction"])
                if "depth_grid_probe_y_fraction" in t
                else float("nan"),
                "probe_y_px": int(t["probe_y_px"]),
                "x_left_px": int(t["x_left_px"]),
                "x_right_px": int(t["x_right_px"]),
                "x_left_norm": float(t["x_left_norm"]),
                "x_right_norm": float(t["x_right_norm"]),
                "task_x_left_norm": float(t["task_x_left_norm"]),
                "task_x_right_norm": float(t["task_x_right_norm"]),
                "anchor_left_depth_near": anchor_left,
                "anchor_right_depth_near": anchor_right,
                "anchor_left_depth_near_texture": anchor_left_texture,
                "anchor_right_depth_near_texture": anchor_right_texture,
                "anchor_left_gap": anchor_left_gap,
                "anchor_right_gap": anchor_right_gap,
                "cue_probe_v": probe_v,
                "cue_probe_y_frac": probe_y_frac,
                "trial_view_seconds": float(self.trial_view_seconds[i]),
            }
            trial_meta = t.get("trial_meta", {})
            if isinstance(trial_meta, dict):
                for k, v in trial_meta.items():
                    if k in row:
                        if row[k] == v:
                            continue
                        row[f"meta_{k}"] = v
                    else:
                        row[k] = v
            for j in range(self.n_movable):
                xj = float(x_inner[j]) if j < x_inner.shape[0] else float("nan")
                row[f"x{j + 1}_norm"] = xj
                row[f"response{j + 1}_ui"] = float(resp[j])
                row[f"response{j + 1}_depth_near"] = float(resp[j])
                row[f"shading_depth{j + 1}_near"] = float(shading_inner[j]) if j < shading_inner.shape[0] else float("nan")
                row[f"texture_depth{j + 1}_near"] = float(texture_inner[j]) if j < texture_inner.shape[0] else float("nan")
            rows.append(row)
        return rows

    def finish(self):
        self._close_trial_timer()
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"cross_section_task_{self.participant_id}_{ts}"
        out_csv = self.responses_dir / f"{stem}.csv"
        out_json = self.responses_dir / f"{stem}.json"

        rows = self._build_output_rows()
        fieldnames = [
            "participant_id",
            "trial_index",
            "stimulus_file",
            "depth_grid_json",
            "depth_grid_row_index",
            "depth_grid_probe_y_fraction",
            "probe_y_px",
            "x_left_px",
            "x_right_px",
            "x_left_norm",
            "x_right_norm",
            "task_x_left_norm",
            "task_x_right_norm",
            "anchor_left_depth_near",
            "anchor_right_depth_near",
            "anchor_left_depth_near_texture",
            "anchor_right_depth_near_texture",
            "anchor_left_gap",
            "anchor_right_gap",
            "cue_probe_v",
            "cue_probe_y_frac",
        ]
        fieldnames += [f"x{i + 1}_norm" for i in range(self.n_movable)]
        fieldnames += [f"response{i + 1}_ui" for i in range(self.n_movable)]
        fieldnames += [f"response{i + 1}_depth_near" for i in range(self.n_movable)]
        fieldnames += [f"shading_depth{i + 1}_near" for i in range(self.n_movable)]
        fieldnames += [f"texture_depth{i + 1}_near" for i in range(self.n_movable)]
        fieldnames += ["trial_view_seconds"]
        extra_keys = []
        seen = set(fieldnames)
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    extra_keys.append(k)
        fieldnames += extra_keys

        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        payload = {
            "participant_id": self.participant_id,
            "task_type": "cross_section_fixed_ends_five_movable",
            "depth_convention": "nearer_is_larger",
            "depth_grid_json": self.depth_grid_path if self.depth_grid_path else None,
            "forward_only": bool(not self.allow_back),
            "experiment_info": self.experiment_info,
            "cue_depth_targets_available": bool(self.cue_targets_available),
            "cue_depth_seed": int(self.cue_profile_seed),
            "response_values_are_absolute_depth_near": True,
            "y_range_ui": [self.y_min, self.y_max],
            "y_range_depth_near": [self.y_min, self.y_max],
            "responses": rows,
        }
        out_json.write_text(json.dumps(payload, indent=2))

        self.saved_path = out_csv
        print(f"Saved participant responses CSV:  {out_csv}")
        print(f"Saved participant responses JSON: {out_json}")
        plt.close(self.fig)

    def run(self):
        plt.show()
        return self.saved_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run depth-perception tasks on generated stimuli (legacy 5-dot or cross-section fixed-ends + 5 movable)."
        )
    )
    parser.add_argument(
        "--task-mode",
        choices=["cross-section", "five-dot"],
        default="cross-section",
        help="Task mode to run (default: cross-section).",
    )
    parser.add_argument("--stimuli-dir", default="stimuli", help="Directory containing stimulus images.")
    parser.add_argument(
        "--single-image",
        default="",
        help="Optional single image path to use as the only trial.",
    )
    parser.add_argument("--participant-id", default="p001", help="Participant identifier for saved responses.")
    parser.add_argument("--responses-dir", default="responses", help="Output directory for response files.")
    parser.add_argument(
        "--stimulus-glob",
        default="*.png",
        help='Glob for stimulus images inside --stimuli-dir (default: "*.png").',
    )
    parser.add_argument(
        "--depth-grid-json",
        default="",
        help=(
            "Optional path to texture_gen true-depth grid JSON. "
            "If omitted in cross-section mode, auto-discovery checks stimulus directories."
        ),
    )
    parser.add_argument(
        "--shape-dirs",
        nargs=2,
        metavar=("SHAPE_DIR_1", "SHAPE_DIR_2"),
        default=None,
        help=(
            "Experiment mode: two shape directories, each containing combo images "
            '(AA..CC) and depth_grid.json. Builds 2 x 8 x repeats trials (AA excluded by default).'
        ),
    )
    parser.add_argument(
        "--trials-per-combo",
        type=int,
        default=4,
        help="Repeats per combo in --shape-dirs experiment mode (default: 4).",
    )
    parser.add_argument(
        "--include-aa",
        action="store_true",
        help="Include AA combo (no-shading/no-texture) in --shape-dirs experiment mode.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Disable backward navigation (space/right/enter/n advance).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for Shape A/B geometry (five-dot profiles and cross-section cue-depth targets).",
    )
    parser.add_argument("--n-samples", type=int, default=401, help="Samples in each continuous depth number-line.")
    parser.add_argument("--profile-json", default="stimuli/shape_ab_midline_profiles.json")
    parser.add_argument("--profile-csv", default="stimuli/shape_ab_midline_profiles.csv")
    parser.add_argument("--profile-plot", default="stimuli/shape_ab_midline_profiles.png")
    parser.add_argument(
        "--no-task",
        action="store_true",
        help=(
            "Do not launch participant UI. "
            "In five-dot mode: only export Shape A/B profiles. "
            "In cross-section mode: save probe preview image(s)."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle stimulus order before presenting the participant task.",
    )
    args = parser.parse_args()

    try:
        _require_matplotlib()
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    use_shape_dirs_experiment = (args.task_mode == "cross-section") and (args.shape_dirs is not None)
    experiment_trials = None
    experiment_info = {}

    if use_shape_dirs_experiment:
        if args.single_image:
            raise SystemExit("--single-image cannot be combined with --shape-dirs experiment mode.")
        exclude = tuple() if args.include_aa else ("AA",)
        try:
            experiment_trials, shape_summary = build_two_shape_experiment_trials(
                shape_dirs=args.shape_dirs,
                seed=args.seed,
                repeats_per_combo=args.trials_per_combo,
                exclude_codes=exclude,
                shuffle=args.shuffle,
            )
        except Exception as exc:
            raise SystemExit(f"Could not build --shape-dirs experiment trials: {exc}")
        stimuli_paths = [Path(t["path"]) for t in experiment_trials]
        combos_per_shape = int(len(shape_summary[0]["combos"])) if shape_summary else 0
        print(
            "Built two-shape experiment trial set: "
            f"{len(stimuli_paths)} trial(s) = 2 shapes x {combos_per_shape} combos x {args.trials_per_combo} repeats."
        )
        for s in shape_summary:
            print(
                f"  {s['shape_label']}: row={s['depth_grid_row_index']} "
                f"from {s['depth_grid_json']} | combos={','.join(s['combos'])}"
            )
        experiment_info = {
            "mode": "two_shape_combo_repeats",
            "shape_dirs": [str(Path(d)) for d in args.shape_dirs],
            "repeats_per_combo": int(args.trials_per_combo),
            "exclude_codes": list(exclude),
            "shuffle": bool(args.shuffle),
            "shape_summary": shape_summary,
            "total_trials": int(len(stimuli_paths)),
        }
    else:
        if args.single_image:
            p = Path(args.single_image)
            if not p.exists() or not p.is_file():
                raise SystemExit(f'Single image not found: "{args.single_image}"')
            stimuli_paths = [p]
        else:
            stimuli_paths = list_stimuli_images(args.stimuli_dir, args.stimulus_glob)

        if not stimuli_paths:
            raise SystemExit(
                f'No images found in stimuli directory "{args.stimuli_dir}" '
                f'with glob "{args.stimulus_glob}".'
            )
        print(f"Loaded {len(stimuli_paths)} stimulus image(s).")

        if args.shuffle:
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(len(stimuli_paths))
            stimuli_paths = [stimuli_paths[i] for i in order]

    depth_grid_data = None
    if (args.task_mode == "cross-section") and (not use_shape_dirs_experiment):
        depth_grid_path = discover_depth_grid_json(stimuli_paths, explicit_path=args.depth_grid_json)
        if depth_grid_path is not None:
            try:
                depth_grid_data = load_depth_grid_metadata(depth_grid_path)
                print(
                    f"Loaded saved true depth grid JSON: {depth_grid_data['path']} "
                    f"({len(depth_grid_data['rows'])} row(s))."
                )
            except Exception as exc:
                if args.depth_grid_json:
                    raise SystemExit(f'Could not load depth grid JSON "{args.depth_grid_json}": {exc}')
                print(
                    "Warning: auto-discovered depth_grid.json could not be loaded; "
                    f"falling back to geometry recomputation ({exc})."
                )
                depth_grid_data = None
        else:
            print(
                "No depth_grid.json found for cross-section mode; "
                "falling back to geometry recomputation."
            )

    if args.task_mode == "five-dot":
        profiles = build_shape_ab_profiles(seed=args.seed, n_samples=args.n_samples)
        save_profiles(profiles, args.profile_json, args.profile_csv, args.profile_plot)

        if args.no_task:
            return

        task = MidlineMatchingTask(
            stimuli_paths=stimuli_paths,
            participant_id=args.participant_id,
            responses_dir=args.responses_dir,
            profile_data=profiles,
        )
        task.run()
        return

    # New cross-section mode
    if args.no_task:
        rng = np.random.default_rng(args.seed)
        preview_dir = Path(args.responses_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        if experiment_trials is not None:
            for i, trial in enumerate(experiment_trials, start=1):
                meta = trial.get("trial_meta", {})
                code = str(meta.get("combo_code", "XX"))
                shape_label = str(meta.get("shape_label", "shape"))
                rep = int(meta.get("repeat_index", 1))
                out = preview_dir / f"cross_section_preview_{i:03d}_{shape_label}_{code}_r{rep}.png"
                save_cross_section_preview(trial, out)
                print(f"Saved cross-section preview: {out}")
            return

        for p in stimuli_paths:
            if depth_grid_data is not None:
                trial = build_cross_section_trial_from_depth_grid(p, rng, depth_grid_data["rows"])
            else:
                trial = build_cross_section_trial(p, rng)
            out = preview_dir / f"cross_section_preview_{Path(p).stem}.png"
            save_cross_section_preview(trial, out)
            print(f"Saved cross-section preview: {out}")
        return

    task = CrossSectionFiveDotTask(
        stimuli_paths=stimuli_paths,
        participant_id=args.participant_id,
        responses_dir=args.responses_dir,
        seed=args.seed,
        depth_grid_data=depth_grid_data,
        prebuilt_trials=experiment_trials,
        allow_back=bool(False if experiment_trials is not None else (not args.forward_only)),
        experiment_info=experiment_info,
    )
    task.run()


if __name__ == "__main__":
    main()
