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
  particular** is observed to break up more often than the top edge (see
  "Top-line-informed bottom-edge detection" below, added to address this)
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

### 4. Line Detection — `detect_top_bottom_lines(edges, image_shape)`
- Runs `cv2.HoughLinesP` (threshold=50, minLineLength=width*0.3,
  maxLineGap=150) on the Canny output, filters for roughly-horizontal lines
  (±10°/170° tolerance), splits candidates into top-half/bottom-half by
  average y-coordinate.
- **Top line**: picked as the longest candidate in the top half — this edge
  is reliably detected in practice.
- **Bottom line** (new this session — see below): instead of independently
  picking "longest candidate in the bottom half," now uses the top line as a
  prior via `find_matching_bottom_line()`, falling back to the old
  longest-candidate approach only if nothing matches.
- `visualize_lines()`: draws detected top (green) / bottom (red) lines for
  visual sanity-checking.

#### Top-line-informed bottom-edge detection (new)
Motivated by the observation that the top edge is detected reliably far more
often than the bottom edge, and that an album cover's bottom edge should be
close to the same angle and length as its top edge. Two new helper functions:
- `merge_collinear_segments(segments, y_tolerance=15)`: clusters Hough
  segments by average y-position (within `y_tolerance` px on the downscaled
  image) and merges each cluster into one segment spanning its extreme
  left/right endpoints — bridges glare-fragmented pieces of the same physical
  edge back into one line.
- `find_matching_bottom_line(bottom_candidates, top_line, angle_tolerance=5)`:
  filters bottom-half candidates to those within `angle_tolerance` degrees of
  the top line's angle (filters out unrelated lines — a hand, a shadow, a
  reflection), merges matching fragments via the above, then picks whichever
  merged candidate's length is closest to the top line's length.
- **Status: implemented and wired into `detect_top_bottom_lines`, not yet
  visually validated by the user** against a real photo with a
  glare-fragmented bottom edge. Next thing to check: run `visualize_lines()`
  on such a photo and confirm the reconstructed bottom line tracks the true
  edge. `angle_tolerance` (5°) and `y_tolerance` (15px) are the two knobs to
  tune if it's too strict/loose.

### 5. Corner Resolution + Perspective Warp
- `resolve_corners(top_line, bottom_line, image_shape)`: extracts 4 corner
  points directly from the top/bottom line endpoints (left-to-right sorted).
  No separate frame-border fallback logic needed — a Hough segment cut off by
  the frame naturally has its endpoint sitting at the image boundary anyway.
  Returns `None` if either line is missing.
- `rescale_corners(corners, scale_factor)`: maps corners found on the
  downscaled image back to full-resolution coordinate space
  (`corners / scale_factor`).
- `warp_to_rectangle(image_bgr, corners, output_size)`: standard
  `getPerspectiveTransform` + `warpPerspective` to flatten to a uniform
  square. Called with the **original full-resolution image**
  (`bgr_array_original`) and the rescaled corners — confirmed working
  end-to-end.
- `visualize_warp()`: displays the warped result.

### 6. Multi-photo batch test (`sample_files` loop)
A cell that runs the full per-image pipeline (load → exif-correct → downscale
→ edge/line detect → resolve corners → rescale → warp) across multiple real
sample photos in a loop, displaying results side by side. Currently exercises
7 of the 11 available files in `Gold_Dots/`. Skips (prints a message, doesn't
crash) any file where corner resolution fails rather than halting the whole
batch — useful for surfacing which covers/angles are still problematic.

## Bugs Found & Fixed This Session
1. **`sorted_endpoints()` malformed ternary** (earlier session) — fixed to an
   explicit if/else.
2. **`corners_full = rescale_corners(corners_small, scale_factor)` warped
   against the wrong array**: the downscale cell was reassigning `bgr_array`
   to the *downscaled* image, clobbering the original full-res array, so the
   "full-res warp" was actually warping the small image with full-res-scaled
   corner coordinates (a coordinate-space mismatch). Fixed by capturing
   `bgr_array_original`/`rgb_array_original` before any downscaling and
   warping against that instead.
2b. Follow-up: `bgr_array_original` was itself briefly built from a stray
   reference to `rgb_array` (a variable not yet defined at that point in a
   fresh kernel — only "worked" via leftover state from a previous run).
   Fixed to reference `rgb_array_original` correctly.
3. **`sample_files` batch loop crashed with `TypeError: unsupported operand
   type(s) for /: 'NoneType' and 'float'`**: the loop ran edge/line detection
   directly on full-resolution images (skipped the downscale step entirely),
   so Hough detection failed for some files (`None` line), and it also reused
   `small`/`scale_factor`/`bgr_array_original` left over from the single-image
   pipeline above it instead of computing them per-file — meaning even a
   successful run would have warped every photo against the same (wrong)
   image. Fixed: each loop iteration now downscales its own image, tracks its
   own `file_scale_factor` and full-res array, and skips with a printed
   message instead of crashing if corner resolution fails.

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
   top/bottom-anchored majority case is working end-to-end. See prior
   conversation for the originally proposed branching logic (full detection →
   3-edge right-angle reconstruction → 2-opposite-edges frame-border fallback
   → unprocessable flag).
3. **Glare handling at the embedding stage** (not just preprocessing):
   originally discussed masking specular highlights or multi-crop embedding
   strategies — not yet started, still just a plan.
4. Embedding generation, reference vector storage, and similarity
   matching/thresholding have not been started — all work so far has been on
   preprocessing/perspective correction only.

## Next Steps (Suggested Order)
1. Validate the new top-line-informed bottom-edge detection
   (`find_matching_bottom_line` / `merge_collinear_segments`) against a real
   photo with a glare-fragmented bottom edge — confirm via `visualize_lines()`
   that the reconstructed line tracks the true edge, and tune
   `angle_tolerance`/`y_tolerance` if needed.
2. Run the `sample_files` batch loop across all 11 available sample covers
   (currently only 7 are listed) and review the grid of warped outputs for
   failure patterns — which covers/angles still fail corner resolution, and
   why.
3. Return to left/right edge-cut-off handling for the remaining edge cases.
4. Return to hand-in-frame handling.
5. Move to embedding generation (CLIP) and reference vector setup for the
   full album catalog.
6. Build similarity comparison + thresholding logic.
7. Begin AWS migration planning once local prototype is validated.
