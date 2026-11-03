"""
Marker pose from the keypoint regressor's 121 grid points, refined in stages (used by test_multimarker_real.py).
Each stage starts from the pose the previous one handed on, and hands that pose on unchanged when it cannot run:

  outliers : least-median-of-squares homography over the 121 points; points further than OUTLIER_SIGMAS robust
             standard deviations from it are dropped and the pose is IPPE + Levenberg-Marquardt on the rest, so
             keypoints thrown off by glare, shadow or truncation no longer drag the pose. The noise scale comes from
             the median residual, so the regressor's ordinary scatter is kept (a fixed px threshold discards most
             of the grid on small markers and makes the pose worse).
  orient   : the 10x10 cells of the white-bordered tag are sampled at the pose (planar white/black fit, as in
             demo_live.tag_confidence) and matched against the 36h11 codes in all four in-plane orientations. If
             the best confident match is in a turned orientation, the regressor read the grid in the wrong order
             and the pose is turned by that many quarter turns about the marker normal. 36h11 codes stay >= 11 bits
             apart under rotation, so a match with <= 5 wrong bits is unique.
  subpix   : the decoded code says which grid points are real L/X corners of the pattern; those are snapped with
             cornerSubPix (starting from the pose's reprojection; points that move more than half a cell are
             dropped) and the pose is re-solved from them alone, as demo_live's refinement does.
  ecc      : the decoded tag is rendered as a template and aligned to the image with ECC (homography motion model,
             invariant to global gain / offset), so every pixel of the tag constrains the pose, not just corners.
             Kept only when it converges above ECC_MIN_CC and moves no grid point by more than ECC_MAX_SHIFT_CELLS.

subpix and ecc need the code, so they only run when the decode confidence reaches min_conf.
Conventions as in demo_live.py / overlay_render_utils: marker frame +X right, +Y up, +Z toward the camera;
121 points x-outer / y-inner spanning the white-bordered 10x10-cell patch; cell row 0 = top (+Y).
"""
import numpy as np
import cv2

STAGES = ("outliers", "orient", "subpix", "ecc")
TRAINED_IDS = list(range(16))   # 36h11 ids the models were trained on
MIN_POINTS = 8                  # inliers / snapped corners needed to re-solve the pose from them alone
OUTLIER_SIGMAS = 3.0            # inlier threshold in robust std devs of the homography residual
ECC_MIN_CC = 0.5                # correlation an ECC alignment must reach to be kept
ECC_MAX_SHIFT_CELLS = 0.5       # ... and the most it may move any grid point, in cells
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

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


def tag_patterns(ids):
    """(n, 100) bool cells, True = white: 1-cell quiet zone + the cv2.aruco 36h11 tag, as textured for training."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    pats = np.ones((len(ids), 10, 10), bool)
    for k, i in enumerate(ids):
        pats[k, 1:9, 1:9] = cv2.aruco.generateImageMarker(dictionary, i, 8) > 127
    return pats.reshape(len(ids), -1)


def corner_masks(patterns, grid, patch):
    """(n, 121) bool: grid points where the tag has a real image corner (an L or X junction of the 4 cells
    around the point), i.e. where cornerSubPix can lock on. Flat areas, straight edges and the outer
    boundary of the quiet zone are excluded."""
    c = patch / 10
    masks = np.zeros((len(patterns), len(grid)), bool)
    for g, (x, y, _) in enumerate(grid):
        rows = [int(np.floor((patch / 2 - y + d) / c)) for d in (-c / 2, c / 2)]  # cell above, below
        cols = [int(np.floor((x + patch / 2 + d) / c)) for d in (-c / 2, c / 2)]  # cell left, right
        if min(rows + cols) < 0 or max(rows + cols) > 9:
            continue
        q = patterns[:, [rows[0] * 10 + cols[0], rows[0] * 10 + cols[1], rows[1] * 10 + cols[0], rows[1] * 10 + cols[1]]]
        h_edge = (q[:, 0] == q[:, 1]) & (q[:, 2] == q[:, 3])  # also true for flat
        v_edge = (q[:, 0] == q[:, 2]) & (q[:, 1] == q[:, 3])
        masks[:, g] = ~(h_edge | v_edge)
    return masks


def rot_z(quarter_turns):
    c, s = [[1, 0], [0, 1], [-1, 0], [0, -1]][quarter_turns % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rotation_perm(points, quarter_turns):
    """perm with points[perm[i]] == Rz(quarter_turns * 90deg) @ points[i]; `points` is symmetric under quarter turns."""
    d = np.linalg.norm((points @ rot_z(quarter_turns).T)[:, None, :2] - points[None, :, :2], axis=2)
    perm = d.argmin(1)
    assert np.allclose(d[np.arange(len(points)), perm], 0, atol=1e-9)
    return perm


def cell_px(kp):
    """Median spacing (px) of neighbouring points of the 11x11 grid in the image."""
    g = kp.reshape(11, 11, 2)
    return min(np.median(np.linalg.norm(np.diff(g, axis=0), axis=2)),
               np.median(np.linalg.norm(np.diff(g, axis=1), axis=2)))


class PoseRefiner:
    """Per-stage marker poses from the 121 regressed grid points; `patch` is the side (m) of the white-bordered
    10x10-cell patch the grid spans (= black tag side * 10/8)."""

    def __init__(self, K, D, patch, ids=TRAINED_IDS, stages=STAGES, min_conf=0.85):
        self.K, self.D, self.patch, self.ids, self.min_conf = K, D, patch, ids, min_conf
        self.stages = [s for s in STAGES if s in stages]
        u = np.linspace(-0.5, 0.5, 11) * patch
        self.grid = np.array([[x, y, 0.0] for x in u for y in u])
        c = patch / 10
        centers = np.array([[-patch / 2 + (col + 0.5) * c, patch / 2 - (row + 0.5) * c, 0.0]
                            for row in range(10) for col in range(10)])
        sub = (-0.25, 0.0, 0.25)  # 9 samples per cell
        self.cells_3d = np.array([[x + dx * c, y - dy * c, 0.0] for x, y, _ in centers for dy in sub for dx in sub])
        self.patterns = tag_patterns(ids)
        self.data_patterns = self.patterns[:, DATA]
        self.corner_masks = corner_masks(self.patterns, self.grid, patch)
        self.cell_perm = [rotation_perm(centers, k) for k in range(4)]

    def project(self, pose):
        return cv2.projectPoints(self.grid, *pose, self.K, self.D)[0].reshape(-1, 2)

    def solve(self, obj, img):
        """(rvec, tvec): the lower-error IPPE solution polished with Levenberg-Marquardt, or None."""
        img = img.astype(np.float64)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, self.K, self.D, flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            return None
        b = int(np.argmin(errs.ravel()))
        return cv2.solvePnPRefineLM(obj, img, self.K, self.D, rvecs[b].copy(), tvecs[b].copy())

    def inliers(self, kp):
        """Grid indices within OUTLIER_SIGMAS robust std devs (at least 1 px) of an LMedS homography of the plane."""
        hom, _ = cv2.findHomography(self.grid[:, :2], kp, cv2.LMEDS)
        if hom is None:
            return np.arange(len(kp))
        r = np.linalg.norm(cv2.perspectiveTransform(self.grid[:, None, :2], hom).reshape(-1, 2) - kp, axis=1)
        sigma = np.median(r) / 1.1774  # median |r| of 2-D Gaussian noise is 1.1774 sigma (Rayleigh)
        return np.flatnonzero(r < max(OUTLIER_SIGMAS * sigma, 1.0))

    def decode(self, gray, pose):
        """((4, n) fraction of the 36 data bits matching each code with the pose turned by k quarter turns about
        the marker normal, fraction of the 64 quiet-zone / border cells that are right), or (None, 0.0) when
        under half the cell samples fall inside the frame."""
        uv = cv2.projectPoints(self.cells_3d, *pose, self.K, self.D)[0].reshape(-1, 2).astype(np.float32)
        h, w = gray.shape
        if np.mean((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)) < 0.5:
            return None, 0.0
        v = cv2.remap(gray, np.ascontiguousarray(uv[:, :1]), np.ascontiguousarray(uv[:, 1:]),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cells = v.reshape(10, 10, -1).mean(-1)
        white = CELL_DESIGN @ np.linalg.lstsq(CELL_DESIGN[QUIET.ravel()], cells[QUIET], rcond=None)[0]
        black = CELL_DESIGN @ np.linalg.lstsq(CELL_DESIGN[BORDER.ravel()], cells[BORDER], rcond=None)[0]
        bits = cells.ravel() > (white + black) / 2
        structure = (bits[STRUCTURE] == QUIET.ravel()[STRUCTURE]).mean()  # the same in every orientation
        # turned by k quarter turns, cell i reads the cell the unturned pose sees at Rz(k*90deg) @ center_i
        match = np.stack([(bits[perm][DATA][None] == self.data_patterns).mean(1) for perm in self.cell_perm])
        return match, float(structure)

    def subpix(self, gray, pose, code):
        """(grid indices, sub-pixel points) of the code's real corners, snapped with cornerSubPix from the pose's
        reprojection; points that moved more than half a cell (onto a neighbouring corner) are dropped."""
        kp = self.project(pose)
        cpx = cell_px(kp)
        win = int(np.clip(0.4 * cpx, 2, 15))  # half-window stays inside one cell
        h, w = gray.shape
        idx = np.flatnonzero(self.corner_masks[code])
        idx = idx[(kp[idx, 0] > win) & (kp[idx, 0] < w - win - 1) & (kp[idx, 1] > win) & (kp[idx, 1] < h - win - 1)]
        if idx.size == 0:
            return idx, np.empty((0, 2))
        pts = kp[idx].astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), CORNER_CRITERIA)
        pts = pts.reshape(-1, 2).astype(np.float64)
        keep = np.linalg.norm(pts - kp[idx], axis=1) < 0.5 * cpx
        return idx[keep], pts[keep]

    def ecc(self, gray, pose, code):
        """(image points of the 121 grid points, correlation) after aligning the code's rendered tag to the image
        with ECC, starting from the pose's reprojection. Points are None when ECC fails, ends below ECC_MIN_CC or
        moves a point by more than ECC_MAX_SHIFT_CELLS."""
        kp = self.project(pose)
        cpx = cell_px(kp)
        ppc = int(np.clip(round(cpx), 6, 24))  # template px per cell, about the tag's scale in the image
        tmpl = np.kron(self.patterns[code].reshape(10, 10), np.ones((ppc, ppc))).astype(np.uint8) * 255
        side = 10 * ppc  # pixel centers at integers, so the patch edge x = -patch/2 is at u = -0.5
        uv = np.c_[(self.grid[:, 0] / self.patch + 0.5) * side - 0.5, (0.5 - self.grid[:, 1] / self.patch) * side - 0.5]
        warp = cv2.findHomography(uv, kp)[0].astype(np.float32)  # template px -> image px
        try:
            cc, warp = cv2.findTransformECC(tmpl, gray, warp, cv2.MOTION_HOMOGRAPHY, ECC_CRITERIA, None, 5)
        except cv2.error:  # did not converge
            return None, np.nan
        pts = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), warp.astype(np.float64)).reshape(-1, 2)
        if cc < ECC_MIN_CC or np.linalg.norm(pts - kp, axis=1).max() > ECC_MAX_SHIFT_CELLS * cpx:
            return None, cc
        return pts, cc

    def __call__(self, gray, kp):
        """({stage: (rvec, tvec)}, diagnostics) for 'raw' (plain IPPE on all 121 points, the unrefined baseline)
        and each enabled stage in order; ({}, {}) when not even the raw pose can be solved."""
        ok, rvec, tvec = cv2.solvePnP(self.grid, kp.astype(np.float64), self.K, self.D, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return {}, {}
        pose = (rvec, tvec)
        poses, info = {"raw": pose}, {}
        if "outliers" in self.stages:
            idx = self.inliers(kp)
            info["n_inliers"] = idx.size
            if idx.size >= MIN_POINTS:
                pose = self.solve(self.grid[idx], kp[idx]) or pose
            poses["outliers"] = pose

        match, structure = self.decode(gray, pose)
        k, code, conf = 0, None, 0.0
        if match is not None:
            if "orient" not in self.stages:
                match = match[:1]
            k, code = np.unravel_index(match.argmax(), match.shape)
            conf = min(structure, float(match[k, code]))
        confident = conf >= self.min_conf
        info.update(tag_id=self.ids[code] if code is not None else np.nan, tag_conf=conf)
        if "orient" in self.stages:
            info["quarter_turns"] = int(k) if confident else 0
            if confident and k:
                pose = (cv2.Rodrigues(cv2.Rodrigues(pose[0])[0] @ rot_z(k))[0], pose[1])
            poses["orient"] = pose

        if "subpix" in self.stages:
            info["n_subpix"] = 0
            if confident:
                idx, pts = self.subpix(gray, pose, code)
                refined = self.solve(self.grid[idx], pts) if idx.size >= MIN_POINTS else None
                if refined is not None:
                    pose, info["n_subpix"] = refined, idx.size
            poses["subpix"] = pose

        if "ecc" in self.stages:
            info["ecc_cc"], info["ecc_used"] = np.nan, False
            if confident:
                pts, info["ecc_cc"] = self.ecc(gray, pose, code)
                refined = self.solve(self.grid, pts) if pts is not None else None
                if refined is not None:
                    pose, info["ecc_used"] = refined, True
            poses["ecc"] = pose
        return poses, info
