# Album Cover Classifier — Prototyping Handoff

## Project Goal
Build an image classifier/matcher that identifies a specific album cover (out of
a personal collection, ~100 eventually — currently 11 sample covers in
`Gold_Dots/`) from a submitted photo. Photos are taken through a plastic
protective sleeve, introducing glare, and at varying angles. Eventually
deploying on AWS; currently prototyping locally in a Jupyter notebook
(`Preprocessing.ipynb` — this is the actual working file; there are no
standalone `.py` modules, everything lives inline in notebook cells).

## Overall Approach (Decided)
Not a trained softmax classifier — with ~100 classes and limited real-world
training data, a classifier would require heavy retraining every time a cover
is added. Instead:

- **Embedding + nearest-neighbor matching**: generate a feature embedding
  (planned: CLIP or similar pretrained backbone) for each of the ~100
  reference covers, then compare a submitted photo's embedding via cosine
  similarity against the reference set. Top match above a confidence
  threshold = predicted album. Below threshold = "no confident match."
- This approach was chosen for robustness to glare/angle variation and because
  it scales without retraining when covers are added/changed.
- AWS migration path (future): SageMaker endpoint for embedding generation,
  OpenSearch k-NN or DynamoDB for reference vector storage (trivial at 100
  items), S3 for image intake, Lambda for orchestration. Rekognition Custom
  Labels flagged as a lower-effort alternative worth evaluating later.

## Current Focus: Preprocessing (Perspective Correction)
Before embedding, submitted photos need perspective correction to normalize
angle/tilt, since this is expected to meaningfully improve match quality.
This has been the entire focus of work so far, across multiple sessions.

### Known real-world constraints on submitted photos
- Covers are behind plastic sleeves → **glare** is a persistent, significant
  issue, and tends to fragment edge detection — the **bottom edge in
  particular** is observed to break up more often than the top edge (this is
  the motivation behind both the top-line-informed Hough matching and the new
  ROI+line-fit approach described below)
- Photos may be taken at an **angle**, not straight-on
- **Hand may be in frame** holding the album — currently unresolved, deferred
  (see "Deferred / Known Issues" below)
- **Left/right edges are often cut off** by the camera frame (inconsistent —
  sometimes 3 edges are in frame, sometimes all 4). Currently working under the
  assumption that **top and bottom edges are reliably in frame**; left/right
  may or may not be. Full edge-case handling (detecting which edges are
  present, reconstructing missing corners) is deferred until the
  top/bottom-anchored approach is working end-to-end.

## Pipeline Built So Far (all in `Preprocessing.ipynb`)

### 1. HEIC Ingestion
- `pillow-heif`'s `register_heif_opener()` registers a HEIC opener with PIL
  (OpenCV has no native HEIC support)
- `ImageOps.exif_transpose` corrects orientation (phone photos store rotation
  as metadata, not baked into pixels), then `.convert("RGB")`
- PIL's RGB output array is reversed on the last axis (`[..., ::-1]`) to get
  OpenCV's expected BGR channel order

### 2. Downscale Before Detection
- Original full-resolution array is preserved as `rgb_array_original` /
  `bgr_array_original` **before** any resizing — this full-res array is what
  the final perspective warp is applied against, to avoid quality loss.
- A working copy is downscaled so the long edge is ~1500px
  (`scale_factor = 1500 / max(full_res_shape)`, `cv2.resize(..., fx=fs, fy=fs,
  interpolation=cv2.INTER_AREA)`) before Canny/Hough — necessary because at
  full resolution (~24MP), glare-induced gaps in the edge map translate to
  very large pixel gaps that blow past `maxLineGap`, and standard/tutorial
  Hough parameters stop behaving predictably. Confirmed: downscaling produces
  much cleaner, more continuous top/bottom edge lines.

### 3. Grayscale → Blur → Canny Edge Detection
- `cv2.cvtColor` (BGR2GRAY) → `cv2.GaussianBlur` (5x5) → `cv2.Canny` (40, 120),
  run on the downscaled image
- Canny's two-threshold hysteresis: pixels above the upper threshold are
  definite edges; below the lower threshold, discarded; in between, kept only
  if connected to a definite edge. Lowering the **lower** threshold is the
  targeted fix for glare-broken edges (reconnects weak-but-connected
  segments) rather than lowering both thresholds indiscriminately.

### 4. Top-Edge Detection (Hough-based) — `detect_top_bottom_lines(edges, image_shape)`
- Runs `cv2.HoughLinesP` (threshold=150, minLineLength=width*0.3,
  maxLineGap=150) on the Canny output, filters for roughly-horizontal lines
  (±10°/170° tolerance).
- **Top line**: candidates are restricted to lines whose average y falls in
  the **top 20% of the image height** (tightened from an earlier "top half"
  rule), then the longest candidate in that band is picked. Restricting to
  the top 20% keeps interior horizontal lines (text, artwork detail) from
  being mistaken for the top edge. The top edge is reliably detected in
  practice.
- `visualize_lines()`: draws detected top (green) / bottom (red) lines for
  visual sanity-checking.

#### Top-line-informed bottom-edge detection (Hough path, still active)
This function still runs inside `detect_top_bottom_lines` and produces the
`bottom_line` currently used by the main warp pipeline (cells using
`resolve_corners`/`warp_to_rectangle`, and the `sample_files` batch loop).
Motivated by the observation that the top edge is detected reliably far more
often than the bottom edge, and that an album cover's bottom edge should be
close to the same angle and length as its top edge. Helper functions:
- `merge_collinear_segments(segments, y_tolerance=20)`: clusters Hough
  segments by average y-position and merges each cluster into one segment
  spanning its extreme left/right endpoints — bridges glare-fragmented pieces
  of the same physical edge back into one line.
- `find_matching_bottom_line(bottom_candidates, top_line, image_height,
  angle_tolerance=5, min_separation_frac=0.3)`: filters bottom-half candidates
  to those (a) within `angle_tolerance` degrees of the top line's angle, and
  (b) at least `min_separation_frac` (30%) of the image height away from the
  top line — both checked as a **pairing constraint** on each
  (top_line, candidate) pair rather than as an upfront absolute-position
  filter, since "far enough" depends on where the top line actually landed,
  not on a fixed row in the frame. Matching candidates are merged via the
  above, then the merged candidate whose length is closest to the top line's
  length is picked. Falls back to the old "longest candidate in bottom half"
  approach if nothing matches on angle+separation.
- **Status**: implemented, wired in, angle_tolerance/min_separation_frac
  tunable. Not fully validated yet against a real glare-fragmented photo via
  `visualize_lines()`. A length-extension idea (stretch a bottom line that's
  <80% of the top line's length back out to match it) was tried and then
  explicitly reverted at the user's request — not in the current pipeline.

### 5. Bottom-Edge Detection (new: ROI mask + robust line fit)
A second, more direct approach to the bottom edge was added this session,
running in parallel with (not yet replacing) the Hough-based
`find_matching_bottom_line` path above. Motivation: Hough's segment-based
detection struggles when glare breaks the bottom edge into many small
pieces; this approach instead masks down to just the expected bottom-edge
region and fits a line directly through the raw Canny edge pixels in that
region.
- `build_bottom_roi_mask(top_line, image_shape, band_start_frac=0.7,
  band_end_frac=0.98)`: builds a binary mask isolating a band roughly the
  bottom ~28% of the frame (from 70% down to 98% down the image height,
  measured from the top edge's average y), **tilted to follow the top edge's
  slope** column-by-column so the band tracks the expected tilt of a
  (roughly parallel) bottom edge rather than being a flat horizontal strip.
- `extract_edge_points(edges, roi_mask)`: `cv2.bitwise_and`s the Canny edge
  map with the ROI mask and returns the surviving edge pixel coordinates.
- `fit_bottom_line(points)`: fits a line through those points via
  `cv2.fitLine` with `DIST_HUBER` (robust to outlier points — stray marks,
  noise — unlike an ordinary least-squares fit), returns `(slope,
  intercept)`. Requires at least 10 points or returns `None`.
- `bottom_corners_from_fit(slope, intercept, width)`: extrapolates the fitted
  line to the image's left (`x=0`) and right (`x=width`) borders to get
  bottom-left/bottom-right corner points directly — no separate Hough
  segment endpoints needed.
- `resolve_bottom_edge(top_line, edges, image_shape)`: orchestrates the above
  four steps into one call.
- `visualize_fit()`: sanity-check plot — green-tinted ROI band, yellow
  candidate edge points, red fitted/extrapolated line.
- **Status: newly added, exercised in one scratch cell
  (`top_line, bottom_line = detect_top_bottom_lines(...)` →
  `build_bottom_roi_mask` → `extract_edge_points` → `fit_bottom_line` →
  `bottom_corners_from_fit` → `visualize_fit`), not yet wired into the main
  `resolve_corners`/`warp_to_rectangle` pipeline.** The scratch cell ends with
  a comment noting the remaining step: combine `bottom_left`/`bottom_right`
  from this path with the existing top-line corner logic into a single
  `corners` array for `warp_to_rectangle`. Not yet validated visually against
  a real glare-fragmented photo.

### 6. Corner Resolution + Perspective Warp (current main pipeline)
- `resolve_corners(top_line, bottom_line, image_shape)`: extracts 4 corner
  points directly from the top/bottom line endpoints (left-to-right sorted).
  No separate frame-border fallback logic needed — a Hough segment cut off by
  the frame naturally has its endpoint sitting at the image boundary anyway.
  Returns `None` if either line is missing. Currently still fed by the
  Hough-based `bottom_line` from `detect_top_bottom_lines` (Section 4), not
  yet by the new ROI+fit path from Section 5.
- `rescale_corners(corners, scale_factor)`: maps corners found on the
  downscaled image back to full-resolution coordinate space
  (`corners / scale_factor`).
- `warp_to_rectangle(image_bgr, corners, output_size)`: standard
  `getPerspectiveTransform` + `warpPerspective` to flatten to a uniform
  square. Called with the **original full-resolution image**
  (`bgr_array_original`) and the rescaled corners — confirmed working
  end-to-end.
- `visualize_warp()`: displays the warped result.

### 7. Multi-photo batch test (`sample_files` loop)
A cell that runs the full per-image pipeline (load → exif-correct → downscale
→ edge/line detect → resolve corners → rescale → warp) across multiple real
sample photos in a loop, displaying results side by side. Currently exercises
8 of the 11 available files in `Gold_Dots/`. Skips (prints a message, doesn't
crash) any file where corner resolution fails rather than halting the whole
batch — useful for surfacing which covers/angles are still problematic. Still
uses the Hough-based bottom-edge path (Section 4/6), not the new ROI+fit path.

## Bugs Found & Fixed
1. **`sorted_endpoints()` malformed ternary** (earlier session) — fixed to an
   explicit if/else.
2. **`corners_full = rescale_corners(corners_small, scale_factor)` warped
   against the wrong array**: the downscale cell was reassigning `bgr_array`
   to the *downscaled* image, clobbering the original full-res array, so the
   "full-res warp" was actually warping the small image with full-res-scaled
   corner coordinates (a coordinate-space mismatch). Fixed by capturing
   `bgr_array_original`/`rgb_array_original` before any downscaling and
   warping against that instead. Follow-up: `bgr_array_original` was itself
   briefly built from a stray reference to `rgb_array` (a variable not yet
   defined at that point in a fresh kernel — only "worked" via leftover state
   from a previous run). Fixed to reference `rgb_array_original` correctly.
3. **`sample_files` batch loop crashed with `TypeError: unsupported operand
   type(s) for /: 'NoneType' and 'float'`**: the loop ran edge/line detection
   directly on full-resolution images (skipped the downscale step entirely),
   so Hough detection failed for some files (`None` line), and it also reused
   `small`/`scale_factor`/`bgr_array_original` left over from the single-image
   pipeline above it instead of computing them per-file. Fixed: each loop
   iteration now downscales its own image, tracks its own
   `file_scale_factor` and full-res array, and skips with a printed message
   instead of crashing if corner resolution fails.
4. **`find_matching_bottom_line`/`detect_top_bottom_lines` argument mismatch**
   (introduced and fixed within this session): while adding the 30%
   min-separation pairing constraint, a later revert (undoing an unrelated
   length-extension experiment) accidentally rolled `find_matching_bottom_line`
   back to its pre-separation-constraint signature (no `image_height`/
   `min_separation_frac` params), while `detect_top_bottom_lines` still called
   it with `height` as a third positional argument — which silently landed in
   `angle_tolerance` instead of raising an error, disabling the angle filter
   rather than crashing. Re-fixed by restoring the full signature.

## Deferred / Known Issues (Not Yet Addressed)
1. **Hand in frame**: user's photos sometimes include their hand holding the
   album, which gets picked up in edge detection. Discussed options (contour
   shape/area filtering, aspect-ratio filtering, skin-tone masking, or simply
   reframing photos to exclude the hand). Decision: deprioritized for now,
   revisit later.
2. **Left/right edge cut-off handling (inconsistent per-photo)**: current
   pipeline assumes top/bottom are always present and only anchors on those.
   Full case-based handling (detecting which of the 4 edges are actually
   present vs. cut off by frame, reconstructing missing corners
   mathematically) was scoped out in detail but explicitly deferred until the
   top/bottom-anchored majority case is working end-to-end.
3. **Glare handling at the embedding stage** (not just preprocessing):
   originally discussed masking specular highlights or multi-crop embedding
   strategies — not yet started, still just a plan.
4. Embedding generation, reference vector storage, and similarity
   matching/thresholding have not been started — all work so far has been on
   preprocessing/perspective correction only.
5. **Two parallel bottom-edge detection approaches now coexist** (Hough-based
   `find_matching_bottom_line` vs. the new ROI-mask + robust-fit
   `resolve_bottom_edge`) — need to decide whether one replaces the other or
   they're combined (e.g. ROI+fit as the primary method with Hough matching
   as a fallback), then wire the winner into `resolve_corners`.

## Next Steps (Suggested Order)
1. Validate the new ROI-mask + robust-line-fit bottom-edge approach
   (`resolve_bottom_edge` / `visualize_fit`) against a real photo with a
   glare-fragmented bottom edge, and compare its output against the
   Hough-based `find_matching_bottom_line` path on the same photos. Tune
   `band_start_frac`/`band_end_frac` if the ROI band is missing the true edge.
2. Decide how the two bottom-edge approaches relate (replace vs. fallback vs.
   combine), then wire the chosen one into `resolve_corners` /
   `warp_to_rectangle` as the main pipeline.
3. Re-validate the Hough-based top-line-informed bottom-edge detection
   (angle + 30% separation pairing constraint) if it remains in use, tuning
   `angle_tolerance`/`min_separation_frac` as needed.
4. Run the `sample_files` batch loop across all 11 available sample covers
   (currently only 8 are listed) and review the grid of warped outputs for
   failure patterns — which covers/angles still fail corner resolution, and
   why.
5. Return to left/right edge-cut-off handling for the remaining edge cases.
6. Return to hand-in-frame handling.
7. Move to embedding generation (CLIP) and reference vector setup for the
   full album catalog.
8. Build similarity comparison + thresholding logic.
9. Begin AWS migration planning once local prototype is validated.
