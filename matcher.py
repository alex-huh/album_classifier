"""Match a perspective-corrected cover photo against the gallery of reference covers.

Extracted from Training.ipynb: the Embedder model, gallery embeddings, and SIFT re-ranking.
Training still lives in the notebook; this module only loads its checkpoint.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

from preprocess import load_image

MODEL_NAMES = {
    "dinov2": "facebook/dinov2-base",
    "clip": "openai/clip-vit-base-patch16",
}
BACKBONE = "dinov2"
GALLERY_DIR = Path("training_images")  # one reference cover per album ("Artist - Album.jpg")
CHECKPOINT_DIR = Path("checkpoints")
IMG_SIZE = 224
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SIFT re-rank: a top-k candidate replaces the embedding's top-1 if it has this many homography inliers
MIN_INLIERS = 15
# Starting confidence rule from MODEL_STATUS.md, to revisit with >=100 labelled photos
MIN_SCORE = 0.35
MIN_GAP = 0.06


class Embedder(nn.Module):
    """Pretrained backbone -> L2-normalized embedding, plus an optional trainable linear head.

    The head starts as the identity, so an untrained Embedder gives the zero-shot embedding.
    """

    def __init__(self, backbone=BACKBONE, use_head=True):
        super().__init__()
        self.backbone = backbone
        name = MODEL_NAMES[backbone]
        if backbone == "clip":
            self.model = CLIPVisionModelWithProjection.from_pretrained(name)
            dim = self.model.config.projection_dim
        else:
            self.model = AutoModel.from_pretrained(name)
            dim = self.model.config.hidden_size
        self.head = nn.Linear(dim, dim) if use_head else nn.Identity()
        if use_head:
            nn.init.eye_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        processor = AutoImageProcessor.from_pretrained(name)
        self.mean, self.std = processor.image_mean, processor.image_std

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        feats = out.image_embeds if self.backbone == "clip" else out.pooler_output
        return F.normalize(self.head(feats), dim=-1)


def eval_transform(model):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),  # covers are square; squash, don't crop
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(model.mean, model.std),
    ])


@torch.no_grad()
def embed_images(model, images, transform, batch_size=64):
    """Embed a list of paths or PIL images -> [N, D] tensor on the CPU."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        batch = [load_image(x) if isinstance(x, Path) else x for x in images[i:i + batch_size]]
        pixels = torch.stack([transform(im) for im in batch]).to(DEVICE)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            out.append(model(pixels).float().cpu())
    return torch.cat(out)


_sift = cv2.SIFT_create(nfeatures=2000)


def sift_features(image, max_side=800):
    """(keypoint coordinates, descriptors) for an RGB PIL image."""
    gray = np.array(image.convert("L"))
    scale = max_side / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = _sift.detectAndCompute(gray, None)
    return np.float32([kp.pt for kp in keypoints]), descriptors


def homography_inliers(query_features, gallery_features, ratio=0.75):
    q_pts, q_des = query_features
    g_pts, g_des = gallery_features
    if q_des is None or g_des is None or len(q_pts) < 4 or len(g_pts) < 4:
        return 0
    pairs = cv2.BFMatcher().knnMatch(q_des, g_des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 4:
        return 0
    src = q_pts[[m.queryIdx for m in good]]
    dst = g_pts[[m.trainIdx for m in good]]
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


class Matcher:
    """Loads the fine-tuned checkpoint and gallery embeddings once, then matches photos."""

    def __init__(self, backbone=BACKBONE, gallery_dir=GALLERY_DIR, checkpoint_dir=CHECKPOINT_DIR):
        self.gallery_dir = Path(gallery_dir)
        self.model = Embedder(backbone).to(DEVICE)
        checkpoint = Path(checkpoint_dir) / f"{backbone}_best.pt"
        self.model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        self.model.eval()
        self.transform = eval_transform(self.model)

        self.gallery_path = Path(checkpoint_dir) / f"{backbone}_gallery.npz"
        self.names, self.embeddings = self._load_gallery()
        self._gallery_sift = {}

    def _load_gallery(self):
        """Saved gallery embeddings, re-embedded if training_images/ has changed since they were saved."""
        names = [p.stem for p in sorted(self.gallery_dir.glob("*.jpg"))]  # same order as Training.ipynb
        if self.gallery_path.exists():
            saved = np.load(self.gallery_path)
            if saved["names"].tolist() == names:
                return names, torch.from_numpy(saved["embeddings"])
        print(f"Embedding {len(names)} gallery covers (training_images/ changed or no saved gallery)...")
        embeddings = self.rebuild_gallery(names)
        return names, embeddings

    def rebuild_gallery(self, names=None):
        if names is None:
            names = [p.stem for p in sorted(self.gallery_dir.glob("*.jpg"))]  # same order as Training.ipynb
        paths = [self.gallery_dir / f"{n}.jpg" for n in names]
        embeddings = embed_images(self.model, paths, self.transform)
        np.savez(self.gallery_path, embeddings=embeddings.numpy(), names=np.array(names))
        self.names, self.embeddings = names, embeddings
        return embeddings

    def _sift_for(self, idx):
        if idx not in self._gallery_sift:
            self._gallery_sift[idx] = sift_features(load_image(self.gallery_dir / f"{self.names[idx]}.jpg"))
        return self._gallery_sift[idx]

    def match(self, cover, k=TOP_K, rerank=True):
        """
        Match a cover image against the gallery.

        Args:
            cover: RGB PIL image, ideally perspective-corrected (preprocess.crop_cover)
            k: number of candidates to return
            rerank: re-check the top k with SIFT + homography inliers

        Returns:
            dict with "best" (album name), "confident" (bool), and "candidates": a list of
            {"album", "score", "inliers"} in embedding-score order (inliers None if not reranked)
        """
        query_emb = embed_images(self.model, [cover], self.transform)
        scores, idx = (query_emb @ self.embeddings.T)[0].topk(k)
        scores, idx = scores.tolist(), idx.tolist()

        inliers = [None] * k
        best = 0
        sift_confirmed = False
        if rerank:
            query_features = sift_features(cover)
            inliers = [homography_inliers(query_features, self._sift_for(i)) for i in idx]
            top = int(np.argmax(inliers))
            if inliers[top] >= MIN_INLIERS:
                best, sift_confirmed = top, True

        gap = scores[0] - scores[1] if k > 1 else scores[0]
        confident = sift_confirmed or (scores[0] >= MIN_SCORE and gap >= MIN_GAP)
        return {
            "best": self.names[idx[best]],
            "confident": confident,
            "candidates": [
                {"album": self.names[i], "score": round(s, 3), "inliers": n}
                for i, s, n in zip(idx, scores, inliers)
            ],
        }
