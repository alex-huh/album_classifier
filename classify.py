"""Identify albums from phone photos: preprocessing + matching together.

    python classify.py Testing/PearlJam-Ten.HEIC "Testing/50 Cent - Get Rich Or Die Tryin'.HEIC"
    python classify.py Testing/            # every photo in a folder
    python classify.py --rebuild-gallery   # after adding or replacing covers in training_images/
"""
import argparse
from pathlib import Path

from matcher import Matcher
from preprocess import center_square, crop_cover, load_image

PHOTO_SUFFIXES = {".heic", ".jpg", ".jpeg", ".png"}


def identify(path, matcher):
    """
    Identify the album in one photo.

    Returns:
        matcher.match's dict plus "photo" (the path), "cropped" (whether the cover was found and
        perspective-corrected) and "crop_failure" (why not, or None). If cropping fails the
        centered square of the photo is matched instead, so there's always an answer.
    """
    image = load_image(path)
    cover, crop_failure = crop_cover(image)
    if cover is None:
        cover = center_square(image)
    result = matcher.match(cover)
    result.update(photo=str(path), cropped=crop_failure is None, crop_failure=crop_failure)
    return result


def expand(paths):
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", nargs="*", help="photo files or folders")
    parser.add_argument("--rebuild-gallery", action="store_true", help="re-embed every cover in training_images/")
    args = parser.parse_args()

    matcher = Matcher()
    if args.rebuild_gallery:
        matcher.rebuild_gallery()
        print(f"Re-embedded {len(matcher.names)} gallery covers")

    for path in expand(args.photos):
        result = identify(path, matcher)
        flag = "" if result["confident"] else "  [not confident]"
        crop = "" if result["cropped"] else f"  [uncropped: {result['crop_failure']}]"
        print(f"{path.name}\n  -> {result['best']}{flag}{crop}")
        for c in result["candidates"]:
            inliers = "" if c["inliers"] is None else f"  inliers={c['inliers']}"
            print(f"       {c['score']:.3f}  {c['album']}{inliers}")


if __name__ == "__main__":
    main()
