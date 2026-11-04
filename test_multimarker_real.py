"""
Evaluate the multi-marker models (grayscale UNETWithDropoutMini segmentation + RegressorMobileNetV3
keypoints, trained by train_seg_multimarker_100k.py / train_kp_multimarker_100k.py) on the real
July-12 RealSense experiments (AprilTag 36h11 id 0, 1280x720).

Each frame is scored against the robot ground truth (tf_m_c.csv) and compared with OpenCV's AprilTag
detector (CCV) on the same frame, plus the previous LBCV runs stored in each dataset's results*/results.csv.

Preprocessing mirrors training (and deep-marker-estimation's build_lbcv_predictor):
  seg : BGR -> RGB -> gray, Normalize(0.485, 0.229), downscaled by --seg-scale (default 0.5 -> 640x360),
        mask upsampled back to 1280x720. At native size the tag is 142-363 px wide, above the training
        range (fx=733 @ 0.3-2.0 m -> 37-244 px), and close-range frames come back empty; at 0.5x it is 71-182 px.
  roi : largest blob -> overlay_render_utils.compute_roi (the exact crop that built the training ROIs)
  kp  : raw 0-255 RGB ROI, CHW, no normalization; 121 pts -> image space
  pose: pose_refinement.PoseRefiner. 'raw' = solvePnP IPPE on all 121 points (the unrefined pipeline), then the
        --stages in order (outliers -> orient -> subpix -> ecc), each one's pose scored on its own. The 'lbcv'
        columns are the last enabled stage; `--stages` with no names gives the unrefined pipeline.
Poses come out in the OpenCV ArUco convention, so no 180deg-about-Z correction is applied
(the old Isaac-trained model needed one, see real_data_processor.run_LBCV_fiducial_marker_detection).

Outputs in --out-dir (eval_outputs/multimarker_real/):  summary.csv, <dataset>/per_frame.csv, <dataset>/vis/*.png

Run from anywhere (Linux or Windows):
    python test_multimarker_real.py                          # all datasets, all refinement stages
    python test_multimarker_real.py --datasets distance skew --vis-every 5
    python test_multimarker_real.py --stages outliers orient # a subset of the refinement stages
    python test_multimarker_real.py --data-root D:/july12    # data somewhere else

Paths (each can be overridden with the environment variable in brackets):
    checkpoints : <this repo>/checkpoints/<name> if present, else the sibling deep-marker-estimation repo's
                  checkpoint folders                                            [LBCV_SEG_CKPT, LBCV_KP_CKPT]
    deep-marker-estimation : the folder next to this repo                       [DEEP_MARKER_ROOT]
    July-12 data : the lab machine's path, or --data-root                       [JULY12_DATA_ROOT]
overlay_render_utils (from deep-marker-estimation/data_generation) is only needed by this script's evaluation,
not by demo_live.py.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:  # model packages resolve when imported from another directory
    sys.path.insert(0, REPO_ROOT)

from segmentation_model.model import UNETWithDropoutMini
from keypoints_model.model import RegressorMobileNetV3
from pose_refinement import PoseRefiner, STAGES

# ---- data / checkpoints ----
DEEP_MARKER_ROOT = os.environ.get("DEEP_MARKER_ROOT",
                                  os.path.join(os.path.dirname(REPO_ROOT), "deep-marker-estimation"))
DATA_ROOT = os.environ.get("JULY12_DATA_ROOT", "/home/rp/dhanush_ws/sunrise-wrapper/data/marker_data_collection/july12")
DATASETS = ["distance", "skew", "truncation", "underexposure_box", "shadow_v2", "glare"]
MARKER_ID = 0  # the tag in every dataset


def _checkpoint(env_var, name, model_dir):
    """Env override, else <repo>/checkpoints/<name>, else the sibling deep-marker-estimation repo."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    local = os.path.join(REPO_ROOT, "checkpoints", name)
    return local if os.path.exists(local) else os.path.join(DEEP_MARKER_ROOT, model_dir, "checkpoints", name)


SEG_CKPT = _checkpoint("LBCV_SEG_CKPT", "seg_mini_multimarker_best.pth.tar", "segmentation_model")
KP_CKPT = _checkpoint("LBCV_KP_CKPT", "kp_mobilenetv3_multimarker_best.pth.tar", "keypoints_model")

sys.path.insert(0, os.path.join(DEEP_MARKER_ROOT, "data_generation"))
try:  # same crop / label mapping as the training ROIs
    from overlay_render_utils import compute_roi, roi_to_keypoints
except ImportError:
    compute_roi = roi_to_keypoints = None


def _require_overlay_utils():
    if compute_roi is None:
        raise SystemExit(f"overlay_render_utils not found in {os.path.join(DEEP_MARKER_ROOT, 'data_generation')}; "
                         "copy deep-marker-estimation next to this repo or set DEEP_MARKER_ROOT")
# KP_CKPT was trained on data_100k, generated before overlay_render_utils.keypoints_to_roi used the cv2.resize
# pixel-center convention, so its outputs are mapped back with the matching legacy inverse. Set False for a model
# trained on data regenerated with the fix.
KP_LEGACY_ROI_LABELS = True
OUT_DIR = "./eval_outputs/multimarker_real"

# ---- camera / marker (same values as ablations/analysis/real_data_processor.py) ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAMERA_MATRIX = np.array([[906.995, 0.0, 638.235],
                          [0.0, 906.995, 360.533],
                          [0.0, 0.0, 1.0]])
DIST_COEFFS = np.zeros(5)
MARKER_LENGTH = 0.100            # incl. white quiet border = keypoint grid extent
MARKER_LENGTH_NO_BORDER = 0.080  # black square, what cv2.aruco measures
GRID_N = 10                      # 11x11 = 121 keypoints
MIN_BLOB_AREA = 100              # px, same as build_lbcv_predictor

SEG_MEAN, SEG_STD = 0.485, 0.229  # albumentations Normalize(mean, std, max_pixel_value=255) as in training
ARUCO = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                cv2.aruco.DetectorParameters())

# 121 marker-frame points, ordered as compute_2D_gridpoints (x outer, y inner)
_u = np.linspace(-MARKER_LENGTH / 2, MARKER_LENGTH / 2, GRID_N + 1)
GRID_POINTS = np.array([[x, y, 0.0] for x in _u for y in _u])
_h = MARKER_LENGTH / 2
MARKER_CORNERS = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]], dtype=np.float64)
_h = MARKER_LENGTH_NO_BORDER / 2
CCV_CORNERS = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]], dtype=np.float64)  # TL,TR,BR,BL


def load_models():
    missing = [p for p in (SEG_CKPT, KP_CKPT) if not os.path.exists(p)]
    if missing:
        raise SystemExit("model checkpoint(s) not found:\n  " + "\n  ".join(missing) +
                         f"\ncopy them into {os.path.join(REPO_ROOT, 'checkpoints')} "
                         "(or set LBCV_SEG_CKPT / LBCV_KP_CKPT / DEEP_MARKER_ROOT)")
    seg_model = UNETWithDropoutMini(in_channels=1, out_channels=1).to(DEVICE)
    seg_model.load_state_dict(torch.load(SEG_CKPT, map_location=DEVICE)["state_dict"])
    kp_model = RegressorMobileNetV3().to(DEVICE)
    kp_model.load_state_dict(torch.load(KP_CKPT, map_location=DEVICE)["state_dict"])
    return seg_model.eval(), kp_model.eval()


def segment(seg_model, rgb, scale):
    """Binary {0,255} mask at the input resolution; the network sees the image resized by `scale`."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy((gray.astype(np.float32) / 255.0 - SEG_MEAN) / SEG_STD)[None, None].to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE):
        logit = seg_model(x)
    mask = (torch.sigmoid(logit)[0, 0].float().cpu().numpy() > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else mask


def largest_blob(mask):
    """(mask of the largest connected component >= MIN_BLOB_AREA or None, number of such blobs)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= MIN_BLOB_AREA)
    if keep.size == 0:
        return None, 0
    best = 1 + keep[np.argmax(areas[keep])]
    return (labels == best).astype(np.uint8) * 255, int(keep.size)


def predict_keypoints(kp_model, rgb, blob):
    """121x2 keypoints in image space and the 128x128 RGB ROI they were predicted from."""
    _require_overlay_utils()
    roi, roi_coords = compute_roi(blob, rgb)
    if roi is None:
        return None, None
    x = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # raw 0-255, as in training
    with torch.no_grad():
        kp_roi = kp_model(x)[0].float().cpu().numpy().reshape(-1, 2)
    return roi_to_image(kp_roi, roi_coords, roi.shape[:2]), roi


def roi_to_image(kp_roi, roi_coords, roi_shape):
    """Model keypoints in ROI space -> image space, with the label convention KP_CKPT was trained on."""
    if not KP_LEGACY_ROI_LABELS:
        _require_overlay_utils()
        return roi_to_keypoints(kp_roi, roi_coords, roi_shape)
    s = np.array(roi_shape)
    center = np.array([(roi_coords[0] + roi_coords[1]) / 2, (roi_coords[2] + roi_coords[3]) / 2])
    m = s / np.array([roi_coords[1] - roi_coords[0], roi_coords[3] - roi_coords[2]])
    return (kp_roi - s / 2) / m + center


def pose_to_tf(pose):
    tf = np.eye(4)
    tf[:3, :3] = cv2.Rodrigues(pose[0])[0]
    tf[:3, 3] = pose[1].ravel()
    return tf


def solve_pose(object_points, image_points, flags):
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points.astype(np.float64),
                                  CAMERA_MATRIX, DIST_COEFFS, flags=flags)
    return pose_to_tf((rvec, tvec)) if ok else None


def ccv_pose(bgr):
    corners, ids, _ = ARUCO.detectMarkers(bgr)
    if ids is None:
        return None
    return solve_pose(CCV_CORNERS, corners[0].reshape(4, 2), cv2.SOLVEPNP_IPPE_SQUARE)


def project(tf, points):
    rvec = cv2.Rodrigues(tf[:3, :3])[0]
    return cv2.projectPoints(points, rvec, tf[:3, 3], CAMERA_MATRIX, DIST_COEFFS)[0].reshape(-1, 2)


def orthonormalize(tf):
    """tf with its rotation projected onto SO(3). The distance / skew / truncation tf_m_c rotations have singular
    values ~0.9998, which pushes (trace - 1) / 2 above 1 and would clip rotation errors below ~1.1 deg to 0."""
    u, _, vt = np.linalg.svd(tf[..., :3, :3])
    out = tf.copy()
    out[..., :3, :3] = u @ vt
    return out


def tf_errors(tf_ref, tf_est):
    """(translation error mm, rotation error deg); tf_err as in ablations/analysis/utils.compute_tf_error."""
    err = np.linalg.inv(tf_ref) @ tf_est
    rot = np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1.0, 1.0)))
    return 1000 * np.linalg.norm(err[:3, 3]), rot


def draw_vis(bgr, mask, kp, gt_kp, roi, tf_lbcv, tf_ccv, text):
    vis = bgr.copy()
    vis[mask > 0] = (0.6 * vis[mask > 0] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)  # mask in yellow
    for (x, y) in gt_kp:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)   # GT keypoints green
    if kp is not None:
        for (x, y) in kp:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)  # predicted keypoints red
    if tf_lbcv is not None:
        cv2.drawFrameAxes(vis, CAMERA_MATRIX, DIST_COEFFS, cv2.Rodrigues(tf_lbcv[:3, :3])[0],
                          tf_lbcv[:3, 3], MARKER_LENGTH / 2, 3)
    if tf_ccv is not None:
        cv2.drawFrameAxes(vis, CAMERA_MATRIX, DIST_COEFFS, cv2.Rodrigues(tf_ccv[:3, :3])[0],
                          tf_ccv[:3, 3], MARKER_LENGTH / 3, 1)
    if roi is not None:  # ROI inset, top-left
        inset = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), (256, 256), interpolation=cv2.INTER_NEAREST)
        vis[40:296, 10:266] = inset
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def previous_lbcv(ds_dir, results_subdir):
    """Detection rate + per-frame (trans mm, rot deg) of a previous LBCV run, or None if absent."""
    path = os.path.join(ds_dir, results_subdir, "results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    det = df["detected_LBCV"].astype(bool)
    errs = []
    for _, r in df[det].iterrows():
        err = np.eye(4)
        err[:3, :3] = np.array([[r[f"tf_error_LBCV_R{a}{b}"] for b in "xyz"] for a in "xyz"])
        err[:3, 3] = [r["tf_error_LBCV_tx"], r["tf_error_LBCV_ty"], r["tf_error_LBCV_tz"]]
        errs.append(tf_errors(np.eye(4), err))
    errs = np.array(errs).reshape(-1, 2)
    return det.mean(), errs[:, 0], errs[:, 1]


def evaluate_dataset(name, seg_model, kp_model, refiner, max_frames, vis_every, seg_scale, out_dir, data_root=DATA_ROOT):
    ds_dir = os.path.join(data_root, name)
    tf_true_all = orthonormalize(np.linalg.inv(pd.read_csv(os.path.join(ds_dir, "tf_m_c.csv")).values.reshape(-1, 4, 4)))
    n = len(tf_true_all) if max_frames is None else min(max_frames, len(tf_true_all))
    vis_dir = os.path.join(out_dir, name, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    stages = ["raw"] + refiner.stages

    rows = []
    for idx in tqdm(range(n), desc=name):
        bgr = cv2.imread(os.path.join(ds_dir, "images", f"picture_{idx}.png"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tf_true = tf_true_all[idx]
        row = {"idx": idx}

        mask = segment(seg_model, rgb, seg_scale)
        gt_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(gt_mask, project(tf_true, MARKER_CORNERS).astype(np.int32), 255)
        union = np.count_nonzero((mask > 0) | (gt_mask > 0))
        row["mask_iou"] = np.count_nonzero((mask > 0) & (gt_mask > 0)) / union if union else np.nan

        blob, row["n_blobs"] = largest_blob(mask)
        kp, roi, tfs = None, None, {}
        if blob is not None:
            kp, roi = predict_keypoints(kp_model, rgb, blob)
        if kp is not None:
            poses, info = refiner(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), kp)
            tfs = {s: pose_to_tf(p) for s, p in poses.items()}
            row.update(info)
        tf_lbcv = tfs.get(stages[-1])
        gt_kp = project(tf_true, GRID_POINTS)
        row["detected_lbcv"] = tf_lbcv is not None
        row["kp_err_px"] = np.linalg.norm(kp - gt_kp, axis=1).mean() if kp is not None else np.nan
        row["trans_err_lbcv_mm"], row["rot_err_lbcv_deg"] = tf_errors(tf_true, tf_lbcv) if tf_lbcv is not None else (np.nan, np.nan)

        tf_ccv = ccv_pose(bgr)
        row["detected_ccv"] = tf_ccv is not None
        row["trans_err_ccv_mm"], row["rot_err_ccv_deg"] = tf_errors(tf_true, tf_ccv) if tf_ccv is not None else (np.nan, np.nan)
        # ground-truth-free agreement with CCV (skew's tf_m_c has a ~35 mm translation offset, CCV included)
        both = tf_lbcv is not None and tf_ccv is not None
        row["trans_lbcv_vs_ccv_mm"], row["rot_lbcv_vs_ccv_deg"] = tf_errors(tf_ccv, tf_lbcv) if both else (np.nan, np.nan)
        for s in stages:  # every refinement stage scored on its own
            tf = tfs.get(s)
            row[f"trans_err_{s}_mm"], row[f"rot_err_{s}_deg"] = tf_errors(tf_true, tf) if tf is not None else (np.nan, np.nan)
            row[f"rot_{s}_vs_ccv_deg"] = tf_errors(tf_ccv, tf)[1] if tf is not None and tf_ccv is not None else np.nan
        rows.append(row)

        if vis_every and idx % vis_every == 0:
            text = (f"{name} #{idx}  LBCV[{stages[-1]}] t={row['trans_err_lbcv_mm']:.1f}mm r={row['rot_err_lbcv_deg']:.1f}deg "
                    f"kp={row['kp_err_px']:.1f}px IoU={row['mask_iou']:.2f} | "
                    f"CCV t={row['trans_err_ccv_mm']:.1f}mm r={row['rot_err_ccv_deg']:.1f}deg")
            cv2.imwrite(os.path.join(vis_dir, f"frame_{idx:05d}.png"),
                        draw_vis(bgr, mask, kp, gt_kp, roi, tf_lbcv, tf_ccv, text))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, name, "per_frame.csv"), index=False)

    summary = {
        "dataset": name, "frames": n,
        "det_lbcv": df["detected_lbcv"].mean(), "det_ccv": df["detected_ccv"].mean(),
        "t_lbcv_mm": df["trans_err_lbcv_mm"].median(), "r_lbcv_deg": df["rot_err_lbcv_deg"].median(),
        "t_ccv_mm": df["trans_err_ccv_mm"].median(), "r_ccv_deg": df["rot_err_ccv_deg"].median(),
        "t_vs_ccv_mm": df["trans_lbcv_vs_ccv_mm"].median(), "r_vs_ccv_deg": df["rot_lbcv_vs_ccv_deg"].median(),
        "kp_err_px": df["kp_err_px"].median(), "mask_iou": df["mask_iou"].median(),
        "flips_lbcv": int((df["rot_err_lbcv_deg"] > 90).sum()), "multi_blob": int((df["n_blobs"] > 1).sum()),
    }
    for s in stages:
        summary[f"r_{s}_deg"] = df[f"rot_err_{s}_deg"].median()
        summary[f"t_{s}_mm"] = df[f"trans_err_{s}_mm"].median()
        summary[f"r_{s}_vs_ccv_deg"] = df[f"rot_{s}_vs_ccv_deg"].median()
        summary[f"flips_{s}"] = int((df[f"rot_err_{s}_deg"] > 90).sum())
    det = df["detected_lbcv"]
    diag = df.reindex(columns=["tag_conf", "tag_id", "quarter_turns", "n_subpix", "ecc_used"])  # absent if nothing detected
    decoded = diag["tag_conf"] >= refiner.min_conf
    summary["decoded"] = decoded[det].mean()                                   # of detections, confidently decoded
    summary["wrong_id"] = int((decoded & (diag["tag_id"] != MARKER_ID)).sum())
    summary["reoriented"] = int((diag["quarter_turns"] > 0).sum())
    summary["subpix_used"] = (diag["n_subpix"] > 0)[det].mean()
    summary["ecc_used"] = diag["ecc_used"].eq(True)[det].mean()
    for tag, subdir in [("prev", "results"), ("prev_mini", "results_seg_mini")]:
        prev = previous_lbcv(ds_dir, subdir) if max_frames is None else None
        summary[f"det_{tag}"] = prev[0] if prev else np.nan
        summary[f"t_{tag}_mm"] = np.median(prev[1]) if prev and len(prev[1]) else np.nan
        summary[f"r_{tag}_deg"] = np.median(prev[2]) if prev and len(prev[2]) else np.nan
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--max-frames", type=int, default=None, help="per dataset (skips previous-run comparison)")
    ap.add_argument("--vis-every", type=int, default=10, help="save a visualization every N frames (0 = none)")
    ap.add_argument("--seg-scale", type=float, default=0.5, help="resize factor for the segmentation input")
    ap.add_argument("--stages", nargs="*", default=list(STAGES),
                    help=f"pose refinement stages, any of {' '.join(STAGES)} (run in that order; none = unrefined)")
    ap.add_argument("--min-conf", type=float, default=0.85,
                    help="tag decode confidence needed to re-orient and to run subpix / ecc")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--data-root", default=DATA_ROOT, help="folder holding the July-12 datasets")
    args = ap.parse_args()
    if set(args.stages) - set(STAGES):
        ap.error(f"unknown stages {sorted(set(args.stages) - set(STAGES))}; choose from {' '.join(STAGES)}")
    if not os.path.isdir(args.data_root):
        ap.error(f"data root {args.data_root} not found; pass --data-root or set JULY12_DATA_ROOT")
    _require_overlay_utils()

    os.makedirs(args.out_dir, exist_ok=True)
    seg_model, kp_model = load_models()
    refiner = PoseRefiner(CAMERA_MATRIX, DIST_COEFFS, MARKER_LENGTH, stages=args.stages, min_conf=args.min_conf)
    stages = ["raw"] + refiner.stages
    print(f"[INFO] device={DEVICE}  seg_scale={args.seg_scale}  stages={' -> '.join(stages)}\n"
          f"  seg={SEG_CKPT}\n  kp ={KP_CKPT}")

    summary = pd.DataFrame([evaluate_dataset(d, seg_model, kp_model, refiner, args.max_frames, args.vis_every,
                                             args.seg_scale, args.out_dir, args.data_root)
                            for d in args.datasets])
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    tables = [("median rotation error vs GT (deg)", "r_{}_deg"), ("median translation error vs GT (mm)", "t_{}_mm"),
              ("median rotation difference vs CCV (deg)", "r_{}_vs_ccv_deg"), ("frames with rotation error > 90 deg", "flips_{}")]
    stage_cols = [col.format(s) for _, col in tables for s in stages]
    print(f"\n[RESULTS] medians over detected frames, final stage ({stages[-1]}); "
          f"t = translation error (mm), r = rotation error (deg)")
    print(summary.drop(columns=stage_cols).round(3).to_string(index=False))
    by_dataset = summary.set_index("dataset")
    for title, col in tables:
        print(f"\n[STAGES] {title}")
        print(by_dataset[[col.format(s) for s in stages]].set_axis(stages, axis=1).round(2).to_string())
    print(f"\n[INFO] per-frame CSVs and visualizations in {args.out_dir}/<dataset>/")


if __name__ == "__main__":
    main()
