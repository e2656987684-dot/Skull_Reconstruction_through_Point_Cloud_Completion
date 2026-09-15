"""Figure for the write-up: case 039 as filled sagittal cross-sections, defective
input and completed prediction side by side with one legend.

The sections come from the distance field the qualitative meshes are extracted
from (mesh_viz.distance_field, locked RECON, one grid shared by input, prediction
and ground truth), not from slicing a mesh, so they have area.

  plane   normal (1, 0, 0) at x = mean X of the implant-labelled ground-truth
          points; both fields are linearly interpolated between the same two layers.
  region  section = field <= radius_mm / scale_mm on the bilinear slice;
          completion = prediction AND NOT defective.
  render  one YZ window for both panels, square pixels, +Y right, +Z up; grey
          #D9D9D9 shared, orange #E6C58A prediction only, #404040 boundaries.

Arrays are matched by skull id. No GPU, checkpoint or inference: the prediction
is the cached fold-0 one. Visualisation only; the printed areas are not a metric.

    python src/eval/fig_sagittal_sections.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "eval"))
sys.path.insert(0, os.path.join(REPO, "src", "data"))

import fig_qualitative_completion as fq         # noqa: E402
import mesh_viz as mv                           # noqa: E402
import paths                                    # noqa: E402

CASE = "039"
RECON = dict(res=128, radius_mm=6.0, sigma=2.5, taubin=60, pad=0.14)
EXPECTED_SOURCES = dict(cache=os.path.join("data", "cache", "skullfix_pairs_4096_6144.npz"),
                        preds=os.path.join("experiments_log", "preds_fold0.npz"),
                        labels=os.path.join("experiments_log", "defect_mask_labels.npz"),
                        config="cd_rep05_full")

OUT_DIR = os.path.join("reports", "figures")
OUT = dict(defective=os.path.join(OUT_DIR, "sagittal_section_case039_defective.png"),
           prediction=os.path.join(OUT_DIR, "sagittal_section_case039_prediction.png"))
FIGURE = os.path.join(OUT_DIR, "sagittal_section_case039.png")

WINDOW = (1800, 1600)
MARGIN = 0.09                      # per side, fraction of the frame, tighter axis
SSAA_FACTOR = 3
LINE_WIDTH = 2.0                   # output px
WHITE, GREY, ORANGE, LINE = "#FFFFFF", "#D9D9D9", "#E6C58A", "#404040"


def rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def check_sources():
    here = dict(cache=paths.DATA_CACHE, preds=fq.PREDS, labels=fq.LABELS, config=fq.CONFIG)
    if here != EXPECTED_SOURCES:
        raise ValueError(f"fig_qualitative_completion reads {here}, expected {EXPECTED_SOURCES}")
    if not (RECON == fq.RECON == mv.RECON):
        raise ValueError(f"RECON drift: here {RECON}, figure {fq.RECON}, mesh_viz {mv.RECON}")


def shared_bounds(inputs, pred, gt):
    """The grid fig_qualitative_completion reconstructs all three clouds on."""
    lo = np.min([c.min(0) for c in (inputs, pred, gt)], axis=0) - RECON["pad"]
    hi = np.max([c.max(0) for c in (inputs, pred, gt)], axis=0) + RECON["pad"]
    return lo, hi


def plane_slice(field, lo, hi, value, axis):
    """Linear interpolation between the two grid layers around `value` on `axis`."""
    n = field.shape[axis]
    step = (hi[axis] - lo[axis]) / (n - 1)
    t = (value - lo[axis]) / step
    k = int(np.floor(t))
    if not 0 <= k < n - 1:
        raise ValueError(f"{value} outside the grid on axis {axis}")
    w = t - k
    S = (1 - w) * np.take(field, k, axis=axis) + w * np.take(field, k + 1, axis=axis)
    return S, dict(t=t, k=k, w=w, step=step)


def sections(clouds, scale, lo, hi, value, axis):
    """Both fields sliced at one position; aborts if a section reaches the grid edge."""
    slices, where, level = {}, None, None
    for name, cloud in clouds.items():
        field, flo, fhi, r = mv.distance_field(cloud, scale, bounds=(lo, hi))
        if not (np.array_equal(flo, lo) and np.array_equal(fhi, hi)):
            raise RuntimeError("distance_field did not use the shared bounds")
        S, w = plane_slice(field, lo, hi, value, axis)
        if where is not None and w != where:
            raise RuntimeError("the two fields were sliced at different positions")
        if np.concatenate([S[[0, -1]].ravel(), S[:, [0, -1]].ravel()]).min() <= r:
            raise RuntimeError(f"{name} section touches the grid edge; it would be clipped")
        slices[name], where, level = S, w, r
    return slices, where, level


def bilinear(S, lo, hi, us, vs):
    """S[u, v] sampled at u = us (columns) and v = vs (rows); NaN outside the grid."""
    def weights(q, a, b, n):
        s = (q - a) / ((b - a) / (n - 1))
        i = np.clip(np.floor(s).astype(int), 0, n - 2)
        return i, s - i, (s >= 0) & (s <= n - 1)
    iu, wu, oku = weights(us, lo[0], hi[0], S.shape[0])
    iv, wv, okv = weights(vs, lo[1], hi[1], S.shape[1])
    A = S[iu] * (1 - wu)[:, None] + S[iu + 1] * wu[:, None]
    B = A[:, iv] * (1 - wv)[None, :] + A[:, iv + 1] * wv[None, :]
    B[~oku, :] = np.nan
    B[:, ~okv] = np.nan
    return B.T


def frame(slices, lo, hi, level):
    """One window around every section boundary, square pixels; "x" horizontal, "y" vertical."""
    from skimage import measure

    step = (hi - lo) / (next(iter(slices)).shape[0] - 1)
    pts = np.vstack([c for S in slices for c in measure.find_contours(S, level)]) * step + lo
    blo, bhi = pts.min(0), pts.max(0)
    centre, half = (blo + bhi) / 2, (bhi - blo) / 2
    aspect = WINDOW[0] / WINDOW[1]
    half_h = max(half[1], half[0] / aspect) / (1 - 2 * MARGIN)
    return dict(x=(centre[0] - half_h * aspect, centre[0] + half_h * aspect),
                y=(centre[1] - half_h, centre[1] + half_h))


def section_masks(slices, lo, hi, level, win):
    """Per-sample section masks on the supersampled frame (row 0 at the top)."""
    W, H = WINDOW[0] * SSAA_FACTOR, WINDOW[1] * SSAA_FACTOR
    px = (win["x"][1] - win["x"][0]) / W
    us = win["x"][0] + (np.arange(W) + 0.5) * px
    vs = win["y"][1] - (np.arange(H) + 0.5) * px
    masks = {n: np.nan_to_num(bilinear(S, lo, hi, us, vs), nan=np.inf) <= level
             for n, S in slices.items()}
    return masks, px


def draw(labels, colours):
    """Fill by label, boundaries where labels change, box-averaged to WINDOW."""
    from scipy.ndimage import distance_transform_edt

    img = np.empty(labels.shape + (3,))
    for value, colour in colours.items():
        img[labels == value] = rgb(colour)
    edge = np.zeros(labels.shape, bool)
    for axis in (0, 1):
        d = np.diff(labels, axis=axis) != 0
        sl = [slice(None)] * 2
        sl[axis] = slice(1, None)
        edge[tuple(sl)] |= d
        sl[axis] = slice(None, -1)
        edge[tuple(sl)] |= d
    # The change band is 2 samples wide; grow it to LINE_WIDTH output px.
    img[distance_transform_edt(~edge) <= (LINE_WIDTH * SSAA_FACTOR - 2) / 2] = rgb(LINE)
    f = SSAA_FACTOR
    return np.round(img.reshape(WINDOW[1], f, WINDOW[0], f, 3).mean(axis=(1, 3))).astype(np.uint8)


def render_panels(masks, px, scale, out):
    """Print areas, write both panels, return the images."""
    from PIL import Image

    defective, prediction = masks["defective"], masks["prediction"]
    completion = prediction & ~defective
    area = (px * scale) ** 2 / 100                                   # cm^2 per sample
    for label, m in (("defective", defective), ("prediction", prediction),
                     ("completion", completion), ("prediction AND defective", prediction & defective),
                     ("defective AND NOT prediction", defective & ~prediction)):
        print(f"area {label:30s} {m.sum() * area:8.2f} cm^2")

    labels = dict(defective=defective.astype(np.uint8),
                  prediction=np.where(completion, 2, np.where(prediction, 1, 0)).astype(np.uint8))
    images = dict(defective=draw(labels["defective"], {0: WHITE, 1: GREY}),
                  prediction=draw(labels["prediction"], {0: WHITE, 1: GREY, 2: ORANGE}))
    for name, img in images.items():
        Image.fromarray(img).save(os.path.join(REPO, out[name]))
        print(f"{name:10s} {img.shape[1]} x {img.shape[0]} px -> {out[name]}")
    return images


def load(axis_name, axis):
    check_sources()
    inputs, pred, gt, scale, implant = fq.load_case(CASE)
    cut = float(gt[implant, axis].astype(np.float64).mean())
    lo, hi = shared_bounds(inputs, pred, gt)
    print(f"{axis_name}_cut {cut:.8f} = mean {axis_name.upper()} of {int(implant.sum())} "
          f"implant-labelled ground-truth points; scale {scale:.5f} mm/unit")
    print(f"shared bounds lo {np.round(lo, 6)} hi {np.round(hi, 6)}")
    slices, where, level = sections(dict(defective=inputs, prediction=pred), scale, lo, hi, cut, axis)
    print(f"cut at layer index {where['t']:.6f}: layer {where['k']} x {1 - where['w']:.6f} + "
          f"layer {where['k'] + 1} x {where['w']:.6f}; section level {level:.8f}")
    return slices, lo, hi, level, scale


def combined_figure(images, out):
    """The two panels unchanged, side by side, one legend centred below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dpi, legend_h, font, box, pad = 100, 170, 44, 56, 22
    W, H = 2 * WINDOW[0], WINDOW[1] + legend_h
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    for i, name in enumerate(("defective", "prediction")):
        a = fig.add_axes([i * WINDOW[0] / W, legend_h / H, WINDOW[0] / W, WINDOW[1] / H])
        a.imshow(images[name], interpolation="nearest")
        a.set_axis_off()

    # Lay the legend out left to right in pixels, then shift the row to centre it.
    renderer = fig.canvas.get_renderer()
    y = legend_h / 2
    items, x = [], 0.0
    for colour, text in ((None, "Prediction panel:"), (GREY, "Shared with defective input"),
                         (ORANGE, "Prediction only")):
        if colour is not None:
            items.append(("box", x, colour))
            x += box + pad
        t = fig.text(0, 0, text, fontsize=font, color="#202020", va="center")
        items.append(("text", x, t))
        x += t.get_window_extent(renderer).width + (2.2 * pad if colour is None else 3 * pad)
    x0 = (W - (x - 3 * pad)) / 2
    for kind, dx, obj in items:
        if kind == "box":
            fig.patches.append(Rectangle(((x0 + dx) / W, (y - box / 2) / H), box / W, box / H,
                                         transform=fig.transFigure, facecolor=obj,
                                         edgecolor=LINE, linewidth=2))
        else:
            obj.set_position(((x0 + dx) / W, y / H))
    fig.savefig(os.path.join(REPO, out), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"figure {W} x {H} px -> {out}")


def main():
    slices, lo, hi, level, scale = load("x", 0)
    win = frame(slices.values(), lo[1:], hi[1:], level)       # in-plane (y, z)
    print(f"YZ window y {np.round(win['x'], 6)} z {np.round(win['y'], 6)}")
    masks, px = section_masks(slices, lo[1:], hi[1:], level, win)
    images = render_panels(masks, px, scale, OUT)
    combined_figure(images, FIGURE)


if __name__ == "__main__":
    main()
