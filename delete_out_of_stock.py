"""Delete album art in training_images for albums with no copies in the store
or the warehouse (third and fourth columns of RPM Gold Dot List.xlsx).

Dry run by default: prints what would be deleted. Pass --delete to delete.
Counts that aren't plain numbers (e.g. "?", "4+?") are treated as possibly in
stock, so those photos are kept.
"""
import sys

from download_covers import OUT_DIR, load_albums, safe_filename


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    delete = "--delete" in sys.argv

    to_remove = [
        OUT_DIR / f"{safe_filename(f'{artist} - {album}')}.jpg"
        for artist, album in load_albums(stocked_only=False).itertuples(index=False)
    ]
    to_remove = [p for p in to_remove if p.exists()]

    for path in to_remove:
        if delete:
            path.unlink()
        print(f"{'Deleted' if delete else 'Would delete'}: {path.name}")
    print(f"{len(to_remove)} photos {'deleted' if delete else 'would be deleted (dry run; pass --delete)'}")


if __name__ == "__main__":
    main()
