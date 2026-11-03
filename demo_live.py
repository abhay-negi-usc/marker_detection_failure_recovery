"""
Live camera demo of the multi-marker LBCV pipeline (grayscale segmentation + keypoint regressor), with
OpenCV's ArUco/AprilTag detector (CCV) drawn alongside for reference. Uses the same models / preprocessing as
test_multimarker_real.py. Every segmented blob is processed, so several markers can be shown at once.

Window: yellow = segmentation mask, red dots = keypoints (the network's; with refine on, all 121 reprojected
from the pose fitted to the refined corners), green dots = sub-pixel refined corners,
thick axes = LBCV pose, thin axes + cyan outline = CCV pose. Each LBCV marker is labelled with its decoded tag
id and confidence, and with the LBCV-vs-CCV rotation difference when CCV sees the same marker.

Confidence: the 10x10 cells of the white-bordered tag are sampled at the estimated pose and binarized against
a planar white/black fit from the quiet zone and the tag border (tolerates shading). confidence =
min(fraction of the 64 quiet-zone/border cells that are right, fraction of the 36 data bits matching the
nearest 36h11 code the models were trained on, ids 0-15); that code is the reported tag id. 0.85 means at most
5 wrong data bits, inside 36h11's unique-decoding radius (codes differ in >= 11 bits), so other tag families
(e.g. ArUco 4x4, which the models also segment) are rejected instead of given a made-up id.
Markers below --min-conf are not drawn.

Pose of a confident marker (each stage toggleable):
  refine (r): the decoded id says which of the 121 grid points are real L/X corners of the tag pattern; those
              keypoints are snapped with cornerSubPix (dropped if they move > half a cell) and the pose is
              re-solved from them alone, then polished with solvePnPRefineLM. Removes most of the regressor's
              per-frame bias and jitter; falls back to the 121 network keypoints when < MIN_REFINED corners lock.
              The drawn keypoints are then the 121 grid points reprojected from that pose, so they sit on the
              refined corners and fill in the flat / edge points cornerSubPix cannot lock onto.
  flips:      a near-frontal tag gives two planar IPPE solutions with near-tied reprojection error, tilted
              ~10 deg either way about the line of sight; picking the lower-error one flickers between them.
              A frame whose best solution is ANCHOR_RATIO x better than its mirror is unambiguous and anchors
              the tilt direction for TRACK_TIMEOUT: tied frames take the solution closer to it (moving or tilted
              tags). With no recent anchor (a still, near-frontal tag) the two solutions' rotational midpoint is
              used. July-12 data vs robot GT: 0.6-1.9 deg median on the frontal sets instead of 10-13 (jitter
              0.01-0.4 deg instead of 2-7), 4.6 instead of 6.0 deg on the tilted skew set.
  filter (f): One Euro filter per tag id (translation per axis, rotation on SO(3)): heavy smoothing when the
              marker is still, little lag when it moves (cutoff grows with speed).

The segmentation input scale is adapted every frame so the largest marker spans ~TARGET_SIDE px, inside
the training range (37-244 px); too-large markers otherwise come back as patchy, fragmented masks.

Keys: q/Esc quit | m mask | k keypoints | c CCV | a auto scale | [ / ] manual seg scale -/+ 0.1
      - / = confidence threshold -/+ 0.05 | r refine | f filter | d scan all OpenCV dictionaries | s snapshot

Poses need the camera intrinsics: pass --calib <npz with camera_matrix, distortion_coeffs> (the format
real_data_processing/apriltag_calibration.py writes, at the capture resolution). Without it, fx is
approximated from --hfov and poses are only indicative; mask, keypoints and ids are unaffected.

Run from this repo root:
    cd /home/rp/abhay_ws/marker_detection_failure_recovery
    python demo_live.py --tag-size 0.08            # side of the black tag square in meters
    python demo_live.py --source video.mp4         # or replay a video file
"""
import os
import time
import argparse
import numpy as np
import cv2
import torch

from test_multimarker_real import load_models, segment, roi_to_image, DEVICE

ROI_SIZE, ROI_PADDING = 128, 5  # overlay_render_utils.compute_roi defaults
TARGET_SIDE = 140               # px the largest marker should span at the segmentation input (auto scale)
TRAINED_IDS = list(range(16))   # 36h11 ids the models were trained on
MIN_REFINED = 8                 # refined corners needed to solve the pose from them alone
AMBIGUITY_RATIO = 1.5           # IPPE solutions count as near-tied below this reprojection-error ratio
ANCHOR_RATIO = 3.0              # ... and as unambiguous (tilt-direction anchor) above this one
TRACK_TIMEOUT = 0.5             # s an anchor / a marker's filter state stays valid without a detection
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
OUT_DIR = "./eval_outputs/live_demo"

# 10x10 cells of the white-bordered tag: outer ring = white quiet zone, next ring = black tag border
QUIET = np.ones((10, 10), bool)
QUIET[1:9, 1:9] = False
BORDER = np.zeros((10, 10), bool)
BORDER[1:9, 1:9] = True
BORDER[2:8, 2:8] = False
STRUCTURE = (QUIET | BORDER).ravel()  # id-independent cells (64)
DATA = ~STRUCTURE                     # 6x6 code bits (36)
_rr, _cc = np.mgrid[0:10, 0:10]
CELL_DESIGN = np.c_[np.ones(100), _rr.ravel(), _cc.ravel()]  # planar fit a + b*row + c*col


def open_source(source, width, height):
    if not source.isdigit():
        return cv2.VideoCapture(source)
    cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 30 fps at 1280x720 on the NexiGo N60
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame
    return cap


def camera_intrinsics(calib, hfov, w, h):
    """(K, dist, calibrated)."""
    if calib:
        d = np.load(calib)
        return d["camera_matrix"].astype(np.float64), d["distortion_coeffs"].astype(np.float64).ravel(), True
    f = (w / 2) / np.tan(np.radians(hfov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]), np.zeros(5), False


def grid_points(tag_size):
    """121 marker-frame points spanning the white-bordered patch (10 cells = tag_size * 10/8)."""
    u = np.linspace(-0.5, 0.5, 11) * tag_size * 10 / 8
    return np.array([[x, y, 0.0] for x in u for y in u])


def cell_points(tag_size, sub=(-0.25, 0.0, 0.25)):
    """9 marker-frame sample points per cell of the 10x10 patch, row 0 = top (+Y), as cv2.aruco's corners."""
    p = tag_size * 10 / 8
    c = p / 10
    return np.array([[-p / 2 + (col + 0.5 + dx) * c, p / 2 - (row + 0.5 + dy) * c, 0.0]
                     for row in range(10) for col in range(10) for dy in sub for dx in sub])


def tag_patterns(ids):
    """(n, 100) bool cells, True = white: 1-cell quiet zone + the cv2.aruco 36h11 tag, as textured for training."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    pats = np.ones((len(ids), 10, 10), bool)
    for k, i in enumerate(ids):
        pats[k, 1:9, 1:9] = cv2.aruco.generateImageMarker(dictionary, i, 8) > 127
    return pats.reshape(len(ids), -1)


def corner_masks(patterns, grid, tag_size):
    """(n, 121) bool: grid points where the tag has a real image corner (an L or X junction of the 4 cells
    around the point), i.e. where cornerSubPix can lock on. Flat areas, straight edges and the outer
    boundary of the quiet zone are excluded."""
    p = tag_size * 10 / 8
    c = p / 10
    masks = np.zeros((len(patterns), len(grid)), bool)
    for g, (x, y, _) in enumerate(grid):
        rows = [int(np.floor((p / 2 - y + d) / c)) for d in (-c / 2, c / 2)]  # cell above, below
        cols = [int(np.floor((x + p / 2 + d) / c)) for d in (-c / 2, c / 2)]  # cell left, right
        if min(rows + cols) < 0 or max(rows + cols) > 9:
            continue
        q = patterns[:, [rows[0] * 10 + cols[0], rows[0] * 10 + cols[1], rows[1] * 10 + cols[0], rows[1] * 10 + cols[1]]]
        h_edge = (q[:, 0] == q[:, 1]) & (q[:, 2] == q[:, 3])  # also true for flat
        v_edge = (q[:, 0] == q[:, 2]) & (q[:, 1] == q[:, 3])
        masks[:, g] = ~(h_edge | v_edge)
    return masks


def refine_keypoints(gray, kp, corner_mask):
    """(grid indices, sub-pixel points) for the keypoints on real tag corners; points that moved more than
    half a cell (locked onto a neighbouring corner) are dropped."""
    g = kp.reshape(11, 11, 2)
    cell_px = min(np.median(np.linalg.norm(np.diff(g, axis=0), axis=2)),
                  np.median(np.linalg.norm(np.diff(g, axis=1), axis=2)))
    win = int(np.clip(0.4 * cell_px, 2, 15))  # half-window stays inside one cell
    h, w = gray.shape
    idx = np.flatnonzero(corner_mask)
    idx = idx[(kp[idx, 0] > win) & (kp[idx, 0] < w - win - 1) & (kp[idx, 1] > win) & (kp[idx, 1] < h - win - 1)]
    if idx.size == 0:
        return idx, np.empty((0, 2))
    pts = kp[idx].astype(np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), CORNER_CRITERIA)
    pts = pts.reshape(-1, 2).astype(np.float64)
    keep = np.linalg.norm(pts - kp[idx], axis=1) < 0.5 * cell_px
    return idx[keep], pts[keep]


class OneEuroPose:
    """One Euro filter (Casiez et al., CHI 2012) on a 6-DoF pose: translation per axis, rotation on SO(3) via
    the rotation vector from the filtered rotation to the measurement. The cutoff frequency grows with the
    (low-passed) speed, so a still marker is smoothed hard while a moving one is followed with little lag."""

    def __init__(self, min_cutoff, beta_trans, beta_rot, d_cutoff=1.0):
        self.min_cutoff, self.beta_trans, self.beta_rot, self.d_cutoff = min_cutoff, beta_trans, beta_rot, d_cutoff
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def __call__(self, rvec, tvec, t):
        R, p = cv2.Rodrigues(rvec)[0], tvec.ravel()
        if self.t is None or not 0 < t - self.t <= TRACK_TIMEOUT:  # new track, gap, or time went backwards
            self.R, self.p, self.v, self.w, self.t = R, p, np.zeros(3), np.zeros(3), t
            return rvec, tvec
        dt, self.t = t - self.t, t
        a_d = self._alpha(self.d_cutoff, dt)
        self.v += a_d * ((p - self.p) / dt - self.v)  # linear velocity
        self.p = self.p + self._alpha(self.min_cutoff + self.beta_trans * np.linalg.norm(self.v), dt) * (p - self.p)
        delta = cv2.Rodrigues(self.R.T @ R)[0].ravel()
        self.w += a_d * (delta / dt - self.w)  # angular velocity
        a = self._alpha(self.min_cutoff + self.beta_rot * np.linalg.norm(self.w), dt)
        self.R = self.R @ cv2.Rodrigues(a * delta)[0]
        return cv2.Rodrigues(self.R)[0], self.p.reshape(3, 1)


def tag_confidence(gray, pose, K, D, cells_3d, patterns):
    """(min(structure cells right, data bits matching the nearest code), index of that code) at the pose;
    `patterns` are the (n, 36) data bits."""
    uv = cv2.projectPoints(cells_3d, *pose, K, D)[0].reshape(-1, 2).astype(np.float32)
    h, w = gray.shape
    if np.mean((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)) < 0.5:
        return 0.0, None
    v = cv2.remap(gray, np.ascontiguousarray(uv[:, :1]), np.ascontiguousarray(uv[:, 1:]),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cells = v.reshape(10, 10, -1).mean(-1)
    white = CELL_DESIGN @ np.linalg.lstsq(CELL_DESIGN[QUIET.ravel()], cells[QUIET], rcond=None)[0]
    black = CELL_DESIGN @ np.linalg.lstsq(CELL_DESIGN[BORDER.ravel()], cells[BORDER], rcond=None)[0]
    bits = cells.ravel() > (white + black) / 2
    structure = (bits[STRUCTURE] == QUIET.ravel()[STRUCTURE]).mean()
    match = (bits[DATA][None] == patterns).mean(1)
    return float(min(structure, match.max())), int(match.argmax())


def blob_rois(rgb, mask, min_area, max_markers):
    """(labels, [(label, 128x128 RGB roi, roi_coords)] largest blob first). Same crop arithmetic as
    overlay_render_utils.compute_roi, but only the ROI is copied instead of padding the whole frame."""
    # fill holes / absorb fragments inside a blob (large or high-contrast tags come back patchy);
    # each blob's bounding box, and so its ROI, is unchanged
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = cv2.drawContours(np.zeros(mask.shape, np.uint8), contours, -1, 1, cv2.FILLED)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    H, W = mask.shape
    rois = []
    for lab in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:max_markers]:
        x, y, bw, bh, area = stats[lab]
        if area < min_area:
            break
        min_x, max_x, min_y, max_y = x, x + bw - 1, y, y + bh - 1
        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
        side = max(max_y - min_y, max_x - min_x) + 2 * ROI_PADDING
        x0, x1, y0, y1 = cx - side // 2, cx + side // 2, cy - side // 2, cy + side // 2
        crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)  # zero fill outside the frame, as compute_roi's border
        sx0, sx1, sy0, sy1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
        rois.append((lab, cv2.resize(crop, (ROI_SIZE, ROI_SIZE)), np.array([x0, x1, y0, y1])))
    return labels, rois


def predict_keypoints_batch(kp_model, rois):
    """Image-space 121x2 keypoints for every ROI, in one forward pass."""
    x = torch.from_numpy(np.stack([roi for _, roi, _ in rois])).float().permute(0, 3, 1, 2).to(DEVICE)
    with torch.no_grad():
        kp_roi = kp_model(x).float().cpu().numpy().reshape(len(rois), -1, 2)
    return [roi_to_image(kp, coords, (ROI_SIZE, ROI_SIZE)) for kp, (_, _, coords) in zip(kp_roi, rois)]


def solve_pose(object_points, image_points, K, D, flags):
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points.astype(np.float64), K, D, flags=flags)
    return (rvec, tvec) if ok else None


def solve_planar_pose(object_points, image_points, K, D, anchor_rvec=None):
    """(pose, unambiguous) from IPPE. If its two planar solutions are near-tied: the one closer to
    `anchor_rvec` (a recent unambiguous pose of this marker), else their rotational midpoint (not LM-polished,
    which would just fall back into one of the two minima). Otherwise the lower-error solution + LM polish."""
    img = image_points.astype(np.float64)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(object_points, img, K, D, flags=cv2.SOLVEPNP_IPPE)
    if n == 0:
        return None, False
    ratio = errs[1, 0] / max(errs[0, 0], 1e-9) if n == 2 else np.inf
    best = 0
    if ratio < AMBIGUITY_RATIO:
        if anchor_rvec is None:
            R0 = cv2.Rodrigues(rvecs[0])[0]
            half = cv2.Rodrigues(0.5 * cv2.Rodrigues(R0.T @ cv2.Rodrigues(rvecs[1])[0])[0])[0]
            return (cv2.Rodrigues(R0 @ half)[0], (tvecs[0] + tvecs[1]) / 2), False
        best = int(np.argmin([rotation_diff_deg(anchor_rvec, r) for r in rvecs]))
    pose = cv2.solvePnPRefineLM(object_points, img, K, D, rvecs[best].copy(), tvecs[best].copy())
    return pose, ratio >= ANCHOR_RATIO


def rotation_diff_deg(rvec_a, rvec_b):
    R = cv2.Rodrigues(rvec_a)[0].T @ cv2.Rodrigues(rvec_b)[0]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def aruco_detector(name):
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)),
                                   cv2.aruco.DetectorParameters())


def scan_dictionaries(frame):
    """{dictionary name: detected ids} for every OpenCV predefined dictionary that finds a marker."""
    found = {}
    # one name per dictionary (cv2 also exposes upper-case aliases such as DICT_APRILTAG_36H11)
    names = {getattr(cv2.aruco, n): n for n in sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))}
    for name in names.values():
        _, ids, _ = aruco_detector(name).detectMarkers(frame)
        if ids is not None:
            found[name] = sorted(set(ids.ravel().tolist()))
    return found


def put_text(img, text, org, scale, color, thickness=2):
    """Text with a dark outline, legible on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw(frame, mask, dets, ccv, K, D, axis_len, show, hud, notice):
    vis = frame.copy()
    if show["mask"]:
        vis[mask] = (0.6 * vis[mask] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
    for m in ccv:
        cv2.polylines(vis, [m["corners"].astype(np.int32)], True, (255, 255, 0), 1)
        if m["pose"] is not None:
            cv2.drawFrameAxes(vis, K, D, *m["pose"], axis_len * 0.6, 1)
        x, y = m["corners"].min(0).astype(int)
        put_text(vis, f"CCV {m['id']}", (x, m["corners"][:, 1].max().astype(int) + 20), 0.6, (255, 255, 0), 2)
    for d in dets:
        if show["kp"]:
            for (x, y) in d["kp"]:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
            for (x, y) in d["refined"] if d["refined"] is not None else []:
                cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        cv2.drawFrameAxes(vis, K, D, *d["pose"], axis_len, 3)
        label = f"id {d['tag_id']} ({d['conf']:.2f})  z={d['pose'][1][2, 0]:.2f}m"
        if d["refined"] is not None:
            label += f"  {len(d['refined'])} corners"
        if d["ccv_id"] is not None and d["ccv_id"] != d["tag_id"]:
            label += f"  CCV says id {d['ccv_id']}"
        if d["drot"] is not None:
            label += f"  dRot vs CCV {d['drot']:.0f}deg"
        x, y = d["kp"].min(0).astype(int)
        put_text(vis, label, (x, max(y - 8, 15)), 0.6, (255, 255, 255), 2)
    put_text(vis, hud, (10, 28), 0.7, (255, 255, 255), 2)
    if notice:
        put_text(vis, notice, (10, 58), 0.6, (0, 255, 255), 2)
    put_text(vis, "q quit | m mask | k kp | c CCV | a auto scale | [ ] seg scale | - = conf | r refine | f filter"
                  " | d dict scan | s snapshot",
                (10, vis.shape[0] - 12), 0.55, (200, 200, 200), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tag-size", type=float, default=0.08, help="black tag square side (m)")
    ap.add_argument("--calib", default=None, help="npz with camera_matrix, distortion_coeffs")
    ap.add_argument("--hfov", type=float, default=78.0, help="horizontal FOV (deg) used when --calib is not given")
    ap.add_argument("--min-conf", type=float, default=0.85, help="hide LBCV estimates below this confidence")
    ap.add_argument("--dict", default="DICT_APRILTAG_36h11", help="cv2.aruco dictionary used by CCV")
    ap.add_argument("--no-refine", action="store_true", help="start with corner refinement off")
    ap.add_argument("--no-filter", action="store_true", help="start with the One Euro filter off")
    ap.add_argument("--min-cutoff", type=float, default=1.0, help="One Euro cutoff (Hz) for a still marker")
    ap.add_argument("--beta-trans", type=float, default=20.0, help="One Euro cutoff gain per m/s")
    ap.add_argument("--beta-rot", type=float, default=1.0, help="One Euro cutoff gain per rad/s")
    ap.add_argument("--seg-scale", type=float, default=0.5, help="initial / fallback segmentation scale")
    ap.add_argument("--fixed-scale", action="store_true", help="disable auto scale")
    ap.add_argument("--min-area", type=int, default=400, help="min blob area (px) to count as a marker")
    ap.add_argument("--max-markers", type=int, default=8)
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until quit)")
    ap.add_argument("--no-window", action="store_true", help="no display; print throughput (benchmark)")
    args = ap.parse_args()

    cap = open_source(args.source, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read from source {args.source}")
    h, w = frame.shape[:2]
    K, D, calibrated = camera_intrinsics(args.calib, args.hfov, w, h)
    grid, cells_3d, patterns = grid_points(args.tag_size), cell_points(args.tag_size), tag_patterns(TRAINED_IDS)
    data_patterns, corners_of = patterns[:, DATA], corner_masks(patterns, grid, args.tag_size)
    s = args.tag_size / 2
    ccv_corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)  # TL,TR,BR,BL
    axis_len = args.tag_size * 10 / 16
    aruco = aruco_detector(args.dict)
    seg_model, kp_model = load_models()
    print(f"[INFO] {w}x{h}  device={DEVICE}  fx={K[0, 0]:.0f}{'' if calibrated else ' (approx. from --hfov)'}  "
          f"CCV dictionary={args.dict}")

    show = {"mask": True, "kp": True, "ccv": True}
    seg_scale, auto_scale, min_conf = args.seg_scale, not args.fixed_scale, args.min_conf
    refine, filtering = not args.no_refine, not args.no_filter
    filters, anchors = {}, {}  # per tag id: OneEuroPose, (rvec, time) of the last unambiguous pose
    from_file = not args.source.isdigit()
    fps, n, n_shown, n_hidden, drots = 0.0, 0, 0, 0, []
    notice, notice_until = "", 0.0
    t_start = t_prev = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.no_window:
        cv2.namedWindow("LBCV live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LBCV live", w, h)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_frame = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if from_file else time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = segment(seg_model, rgb, seg_scale)
        labels, rois = blob_rois(rgb, mask, args.min_area, args.max_markers)
        dets = []
        for (lab, _, _), kp in zip(rois, predict_keypoints_batch(kp_model, rois) if rois else []):
            pose = solve_pose(grid, kp, K, D, cv2.SOLVEPNP_IPPE)
            conf, idx = tag_confidence(gray, pose, K, D, cells_3d, data_patterns) if pose is not None else (0.0, None)
            dets.append({"label": lab, "kp": kp, "pose": pose, "conf": conf, "pattern": idx, "refined": None,
                         "tag_id": None if idx is None else TRAINED_IDS[idx], "ccv_id": None, "drot": None})
        shown = sorted((d for d in dets if d["pattern"] is not None and d["conf"] >= min_conf), key=lambda d: -d["conf"])

        # final pose of each confident marker: corner refinement -> IPPE (flip handling) -> One Euro filter
        seen = set()  # tag ids already tracked this frame (a duplicate id is shown untracked)
        for d in shown:
            obj, img = grid, d["kp"]
            if refine:
                idx, pts = refine_keypoints(gray, d["kp"], corners_of[d["pattern"]])
                if len(idx) >= MIN_REFINED:
                    obj, img, d["refined"] = grid[idx], pts, pts
            tid = d["tag_id"]
            anchor = anchors.get(tid) if tid not in seen else None
            anchor_rvec = anchor[0] if anchor is not None and t_frame - anchor[1] <= TRACK_TIMEOUT else None
            pose, unambiguous = solve_planar_pose(obj, img, K, D, anchor_rvec)
            pose = pose or d["pose"]
            if d["refined"] is not None:  # all 121 keypoints onto the pose fitted to the refined corners (pre-filter)
                d["kp"] = cv2.projectPoints(grid, *pose, K, D)[0].reshape(-1, 2)
            if tid not in seen:
                if unambiguous:
                    anchors[tid] = (pose[0], t_frame)
                if filtering:
                    pose = filters.setdefault(tid, OneEuroPose(args.min_cutoff, args.beta_trans, args.beta_rot))(*pose, t_frame)
                seen.add(tid)
            d["pose"] = pose

        ccv = []
        if show["ccv"]:
            corners, ids, _ = aruco.detectMarkers(frame)
            for c, i in zip(corners, ids.ravel() if ids is not None else []):
                c = c.reshape(4, 2)
                pose = solve_pose(ccv_corners, c, K, D, cv2.SOLVEPNP_IPPE_SQUARE)
                ccv.append({"id": int(i), "corners": c, "pose": pose})
                cx, cy = np.clip(c.mean(0).astype(int), 0, [w - 1, h - 1])
                for d in dets:  # the LBCV blob containing this marker's center
                    if d["label"] == labels[cy, cx]:
                        d["ccv_id"] = int(i)
                        if d["pose"] is not None and pose is not None:
                            d["drot"] = rotation_diff_deg(d["pose"][0], pose[0])

        hud = (f"seg x{seg_scale:.2f}{' auto' if auto_scale else ''} | LBCV {len(shown)} shown, "
               f"{len(dets) - len(shown)} below conf {min_conf:.2f} | refine {'on' if refine else 'off'} | "
               f"filter {'on' if filtering else 'off'} | CCV {len(ccv) if show['ccv'] else 'off'}"
               + ("" if calibrated else " | uncalibrated"))
        if auto_scale:  # steer the largest marker towards TARGET_SIDE px for the next frame
            if rois:
                side = max(x1 - x0 for _, _, (x0, x1, _, _) in rois)
                seg_scale = float(np.clip(0.6 * seg_scale + 0.4 * TARGET_SIDE / side, 0.2, 1.0))
            else:
                seg_scale = args.seg_scale

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if n else 1 / max(now - t_prev, 1e-6)
        t_prev = now
        n += 1
        n_shown += len(shown)
        n_hidden += len(dets) - len(shown)
        drots += [d["drot"] for d in shown if d["drot"] is not None]
        if args.frames and n >= args.frames:
            break
        if args.no_window:
            continue

        shown_mask = np.isin(labels, [d["label"] for d in shown]) & (mask > 0)
        vis = draw(frame, shown_mask, shown, ccv, K, D, axis_len, show, f"{fps:4.1f} fps | " + hud,
                   notice if now < notice_until else "")
        cv2.imshow("LBCV live", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty("LBCV live", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("m"):
            show["mask"] = not show["mask"]
        elif key == ord("k"):
            show["kp"] = not show["kp"]
        elif key == ord("c"):
            show["ccv"] = not show["ccv"]
        elif key == ord("a"):
            auto_scale = not auto_scale
        elif key == ord("["):
            auto_scale, seg_scale = False, max(0.2, round(seg_scale - 0.1, 1))
        elif key == ord("]"):
            auto_scale, seg_scale = False, min(1.0, round(seg_scale + 0.1, 1))
        elif key == ord("-"):
            min_conf = max(0.0, round(min_conf - 0.05, 2))
        elif key in (ord("="), ord("+")):
            min_conf = min(1.0, round(min_conf + 0.05, 2))
        elif key == ord("r"):
            refine = not refine
        elif key == ord("f"):
            filtering = not filtering
        elif key == ord("d"):
            found = scan_dictionaries(frame)
            notice = ("dict scan: " + ", ".join(f"{k.replace('DICT_', '')} {v}" for k, v in found.items())
                      if found else "dict scan: no OpenCV dictionary detects a marker in this frame")
            notice_until = time.time() + 8
            print(f"[INFO] {notice}")
        elif key == ord("s"):
            path = os.path.join(OUT_DIR, f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(path, vis)
            cv2.imwrite(path.replace(".png", "_raw.png"), frame)
            print(f"[INFO] saved {path} (+ _raw)")

    elapsed = time.time() - t_start
    print(f"[INFO] {n} frames in {elapsed:.1f}s ({n / elapsed:.1f} fps), {n_shown / max(n, 1):.2f} LBCV markers "
          f"shown/frame, {n_hidden} hidden below conf {min_conf:.2f}, {len(drots)} shown matched to CCV"
          + (f" (dRot median {np.median(drots):.1f}, p90 {np.percentile(drots, 90):.1f} deg)" if drots else ""))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
