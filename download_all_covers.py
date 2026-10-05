"""Download album art for every album in RPM Gold Dot List.xlsx, in stock or not.

Same lookup as download_covers.py, but without the store/warehouse filter.
Albums that already have art in training_images are skipped, so this only
fetches the remaining ones. Albums with no match or no cover are logged to
training_images/missing.csv.

Note: delete_out_of_stock.py --delete would remove the out-of-stock covers
this downloads.
"""
from download_covers import load_albums, main

if __name__ == "__main__":
    main(load_albums(stocked_only=None))
