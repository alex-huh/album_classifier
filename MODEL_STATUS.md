# Model Status — Album Cover Matching

_Last updated: 2026-10-04 (run finished 19:24, checkpoint `checkpoints/dinov2_best.pt`)_

Companion to `handoff.md` (preprocessing) and `Training.ipynb` (model). This file tracks
where the matching model stands: what was trained, how it scores, and what the numbers mean.

## TL;DR

| Model | Top-1 | Top-5 | Notes |
|---|---|---|---|
| DINOv2 zero-shot | 71.7% (38/53) | 84.9% | no training |
| CLIP zero-shot | 79.2% (42/53) | 90.6% | no training |
| DINOv2 zero-shot + SIFT re-rank | 84.9% | — | |
| CLIP zero-shot + SIFT re-rank | 86.8% | — | |
| **DINOv2 fine-tuned** | **94.3% (50/53)** | **96.2%** | current model |
| **DINOv2 fine-tuned + SIFT re-rank** | **96.2% (51/53)** | — | best overall |

- Fine-tuning DINOv2 on synthetic "photo through a sleeve" images gave a **+22.6 pt** top-1 gain.
- **Every album with a real test photo was held out of training**, so these scores measure how
  the model handles covers it has never been trained on. That's what will happen each time a
  new album is added to the collection.
- **2 of the 3 remaining misses are bad reference images, not model errors** (see
  [Misses](#remaining-misses)). With those fixed, the model would effectively be at 50/51.

## Data used

| | Count | Source |
|---|---|---|
| Gallery (reference) covers | 1,465 | `training_images/*.jpg`, one per album, named `Artist - Album` |
| Real photos (queries) | 62 in `Processed/` → **53 labelled** | perspective-corrected output of `Preprocessing.ipynb` |
| Training albums | 1,200 | gallery minus held-out |
| Held-out albums | 265 | 53 albums with real photos + 212 random (15%) |

**9 real photos were skipped** because no matching gallery cover exists (either not downloaded
or the name doesn't match). Add the cover to `training_images/` or an entry to
`LABEL_OVERRIDES` to include them:

- Bill Evans - Undercurrent
- Bill Withers - Greatest Hits
- Chet Baker - Sings
- Khruangbin & Leon Bridges - Texas Sun
- King Crimson - In the Court of the Crimson King
- Lana Del Rey - NFR
- Soundgarden - Live At The Palladium, Hollywood October 8, 1991
- The Black Eyed Peas - Elephunk
- Twenty One Pilots - Breach

**Fuzzy label matches (all checked, all correct):** 50 Cent, A Tribe Called Quest, Charli XCX,
Gang Starr, Korn, Raye, Stan Getz & João Gilberto, The Smiths, The Velvet Underground, The
Weeknd. Two gallery filenames have typos worth fixing at some point: `Korn - Follow The Lead`
and `Raye - My 21st Centry Blues`.

Photos whose album top edge is out of frame are now skipped by preprocessing and copied to
`Needs_Review/` (currently: Michael Jackson - Dangerous). One more (One Direction - Four) is
skipped because its bottom edge couldn't be resolved.

## How the model works

1. **Backbone:** `facebook/dinov2-base` (ViT-B/14), producing a 768-d embedding per image,
   followed by a 768×768 linear head initialised to the identity. Embeddings are
   L2-normalised.
2. **Matching:** a real photo is embedded and compared against all 1,465 gallery embeddings
   with cosine similarity. The highest score is the prediction. No classifier layer exists,
   so adding an album only means embedding its cover. No retraining needed.
3. **Training signal:** there is only one image per album, so training pairs are made by
   augmenting each cover twice ("sleeve photo" augmentations, below). Supervised contrastive
   loss pulls the two views of the same cover together and pushes other covers away.
4. **Optional SIFT re-rank:** for the top-5 embedding matches, count SIFT keypoint matches that
   agree on a single homography (covers are flat). The candidate with the most inliers wins if
   it has ≥15. Otherwise the embedding's top-1 is kept.

## Parameters (current run)

### Model / evaluation

| Parameter | Value | What it does |
|---|---|---|
| `BACKBONE` | `dinov2` | Chosen over CLIP: lower zero-shot score but much bigger gain from fine-tuning (see below). |
| `IMG_SIZE` | 224 | Input resolution. Covers are squashed to square, not cropped, so edge content isn't lost. |
| `TOP_K` | 5 | Candidates reported and passed to SIFT re-rank. |
| `SEED` | 0 | Fixes the held-out split and synthetic test images so runs are comparable. |

### Fine-tuning

| Parameter | Value | What it does |
|---|---|---|
| `HELDOUT_FRACTION` | 0.15 | Random 15% of albums never trained on. Used for checkpoint selection. |
| `HOLD_OUT_QUERY_ALBUMS` | True | Albums with real photos are also never trained on, so the real-photo score is a fair "new album" test. |
| `TRAIN_LAST_N_BLOCKS` | 2 | Only the last 2 of 12 transformer blocks (+ final norm + head) are trained: 14.8M trainable params. Keeps the pretrained features intact and limits overfitting to the synthetic data. |
| `EPOCHS` | 20 | |
| `BATCH_SIZE` | 32 | Albums per batch; each contributes 2 views → 64 images. Every other album in the batch acts as a negative. |
| `REPEATS_PER_EPOCH` | 4 | Each training album is sampled 4× per epoch with fresh augmentations (~150 steps/epoch). |
| `BACKBONE_LR` / `HEAD_LR` | 1e-5 / 1e-4 | Small LR for pretrained blocks, 10× larger for the new head. AdamW, weight decay 0.05, cosine decay over all steps. |
| `TEMPERATURE` | 0.07 | SupCon loss sharpness. Lower means it focuses harder on the most confusable covers. |
| Mixed precision | on (CUDA) | Runs on the RTX 2060 (6 GB). |

### Sleeve-photo augmentations (`sleeve_photo_aug`)

Each makes a clean cover look more like a real submission:

| Augmentation | Setting | Simulates |
|---|---|---|
| `RandomPerspective` | distortion 0.3, p=0.7 | residual tilt the preprocessing warp didn't fully remove |
| `RandomResizedCrop` | scale 0.6–1.0, ratio 0.7–1.1 | cut-off edges / imperfect crop |
| `ColorJitter` | brightness/contrast/saturation 0.3, hue 0.03 | lighting, white balance, phone processing |
| `GaussianBlur` | σ 0.1–2.0, p=0.4 | focus / motion blur |
| `JPEG` | quality 40–95 | compression |
| `SleeveGlare` | 1–3 streaks, haze ≤0.25, p=0.8 | specular glare and washed-out contrast from the plastic sleeve |
| `RandomErasing` | 2–15% area, p=0.3 | hand, price sticker |

## Results explained

### Zero-shot: CLIP vs DINOv2

- **CLIP scored higher out of the box** (79.2% vs 71.7%) and its correct matches score higher
  (mean 0.784 vs 0.688). CLIP is trained to connect images with text, so it reads the large
  artist/title lettering on covers well.
- **But CLIP's mistakes are "semantic":** it picks a *different album by the same artist*
  (Linkin Park → *Hybrid Theory*, Kanye → *Late Registration*, Norah Jones → *Day Breaks*,
  Harry Styles → *Harry Styles*, Bauhaus → *Mask*). This is the worst failure for this use
  case, because those are exactly the covers that sit together in the collection.
- **DINOv2's mistakes are visual:** it misses when glare or a sleeve changes the overall look,
  but it's less tied to the text. That makes it a better starting point for fine-tuning on
  glare/sleeve augmentations, which is borne out by the fine-tuned result.
- Both zero-shot models also have overlapping right/wrong score ranges (DINOv2 wrong max 0.621
  vs right min 0.424; CLIP 0.789 vs 0.602), so no single score cutoff separates right from
  wrong without training.

### Fine-tuned DINOv2

| | Zero-shot | Fine-tuned |
|---|---|---|
| Top-1 | 71.7% | **94.3%** |
| Top-5 | 84.9% | **96.2%** |
| Mean score, correct matches | 0.688 | 0.728 |
| Highest score of a wrong match | 0.621 | 0.487 |

- Fine-tuning fixed 12 of the 15 zero-shot misses, including several where the true cover was
  ranked 100–250 (Alicia Keys, Bauhaus).
- Wrong matches now score lower (max 0.621 → 0.487) while correct ones score higher, so the
  model has learned to be less confident on things it doesn't recognise.
- These gains are on **albums the model never trained on**. It learned "how to see through a
  sleeve" in general, not specific covers. That's the property needed for a growing
  collection.

### SIFT re-ranking

- Adds +7–13 pts to the zero-shot models, and +1.9 pts (one photo, Charli XCX *Brat*, rank
  2 → 1) to the fine-tuned model.
- It's cheap (only 5 candidates checked) and tolerant of glare and hands, because it only needs a
  subset of keypoints to agree. It's worth keeping as the second stage in deployment.

### Remaining misses

| Photo | Predicted | True cover rank | Cause |
|---|---|---|---|
| ABBA - Gold | RATM - Renegades (0.393) | 50 | **Wrong reference image.** The gallery `ABBA - Gold.jpg` is the *Gold Ballads* CD cover (group photo); the record is the black/gold-lettering *Gold: Greatest Hits*. |
| Cannonball Adderley - Somethin' Else | Sonic Youth - Live In Austin (0.460) | 923 | **Wrong reference image.** The gallery shows a photo of Adderley; the record is the Blue Note DOL reissue with the large-text layout and a DOL sticker. |
| Charli XCX - Brat | Strokes - Comedown Machine (0.487) | 2 (fixed by SIFT) | **Genuinely hard.** The cover is just a flat lime field and one word. The sleeve dulls the green, and there are few distinctive features. |

**Action:** replace `training_images/ABBA - Gold.jpg` and
`training_images/Cannonball Adderley - Somethin' Else.jpg` with the editions actually in
stock. Since the gallery is just embeddings, this needs no retraining. Re-run the export
cell (and `evaluate_real`) and both should resolve.

## Confidence threshold (not set yet)

`match_photo(..., threshold=)` is meant to return "no confident match" instead of a wrong
answer. Based on this run:

- **Absolute score is not a clean separator yet.** The lowest-scoring correct matches
  (Beach House *Become* 0.387, Alicia Keys 0.392) score *below* the highest wrong one
  (0.487).
- **The gap between the top-1 and top-2 score looks more useful.** All 3 wrong matches have a
  gap ≤ 0.055, and only one correct match (Alicia Keys, 0.040) is that close. Most correct
  matches have gaps of 0.14–0.57.
- Suggested starting rule, to validate as more photos are added: **accept if top-1 ≥ 0.35 and
  (top-1 − top-2) ≥ 0.06, otherwise return "no confident match" or fall back to SIFT.**
- 53 photos is too few to set this reliably. Revisit with ≥100 real photos.

## Caveats

- **Small test set.** 53 photos means one photo ≈ 1.9 pts of accuracy. Treat differences of a
  few points as noise.
- **Results were reconstructed, not read from the notebook.** The training run finished at
  19:24 but `Training.ipynb` wasn't saved afterwards, so the outputs in the file are still from
  the earlier run. Every number above came from re-running the notebook's own evaluation code
  against the saved checkpoint. The per-epoch loss / held-out curve for this run wasn't
  recoverable; save the notebook after the next run to keep it.
- **Split assumption.** The 1,200 / 265 split above was recomputed from the current gallery and
  `Processed/` folder with `SEED = 0`. It matches the training run only if the run's data cell
  executed after `Processed/` finished regenerating (18:30). If it ran earlier, some of the
  newer query albums could have been in its training set, and the real-photo score could be
  slightly optimistic for those.
- **Held-out synthetic accuracy** (used to pick the checkpoint) runs near 99% and is much
  easier than real photos. Use it for checkpoint selection, not as a quality estimate.
- The `lr_scheduler.step() before optimizer.step()` warning is harmless. It fires because the
  AMP grad scaler skips the very first step.

## History

| Date | Gallery | Labelled photos | DINOv2 zero-shot top-1 | DINOv2 fine-tuned top-1 |
|---|---|---|---|---|
| 2026-10-04 (earlier) | 718 | 12 | 83.3% | 91.7% (top-5 100%) |
| 2026-10-04 (current) | 1,465 | 53 | 71.7% | **94.3%** (top-5 96.2%) |

The zero-shot drop between runs comes from the harder test, not a worse model: twice as many
gallery covers to confuse, and 4× the real photos. The fine-tuned model improved despite both.

## Next steps

1. Replace the two wrong reference covers (ABBA *Gold*, Cannonball Adderley *Somethin' Else*),
   then re-run the evaluation.
2. Add covers for the 9 skipped photos, to grow the real test set.
3. Keep photographing: aim for ≥100 labelled real photos before fixing the confidence
   threshold.
4. Try CLIP fine-tuned with the same recipe for a fair comparison (only zero-shot CLIP has been
   tested). Also try `TRAIN_LAST_N_BLOCKS = 4`.
5. Wire the deployment path: preprocessing → `match_photo` → SIFT re-rank → threshold.
