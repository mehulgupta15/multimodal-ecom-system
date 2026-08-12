import os
import sys
# Force UTF-8 on Windows stdout to avoid CP1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import gzip
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_TARGET      = 25_000   # overshoot so we land 20k+ after image filtering
MAX_PER_CATEGORY  = 1_500    # diversity cap per product type
DOWNLOAD_WORKERS  = 32       # parallel image download threads
LISTING_FILES     = list(range(10))  # listings_0.json.gz … listings_9.json.gz
ABO_BASE_URL      = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com"
# ─────────────────────────────────────────────────────────────────────────────


def download_single_image(args):
    """Worker: download one product JPG. Returns True on success."""
    img_url, local_path = args
    if os.path.exists(local_path) and os.path.getsize(local_path) > 500:
        return True  # already have it
    try:
        req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp, \
             open(local_path, "wb") as out:
            out.write(resp.read())
        return os.path.getsize(local_path) > 500
    except Exception:
        return False


def fetch_amazon_physical_catalog(
    total_target: int = TOTAL_TARGET,
    max_per_category: int = MAX_PER_CATEGORY,
):
    print("=" * 60)
    print("  Amazon ABO Multi-Listing Scraper  --  target:", total_target)
    print("=" * 60)

    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir  = os.path.join(base_dir, "data")
    img_dir   = os.path.join(data_dir, "data_images")
    csv_path  = os.path.join(data_dir, "products_catalog.csv")
    os.makedirs(img_dir, exist_ok=True)

    catalog_rows    = []
    download_tasks  = []
    category_counts = defaultdict(int)   # diversity tracker

    # ── Parse listing files until we hit the target ───────────────────────────
    for file_idx in LISTING_FILES:
        if len(catalog_rows) >= total_target:
            break

        gz_url       = f"{ABO_BASE_URL}/listings/metadata/listings_{file_idx}.json.gz"
        temp_gz_path = os.path.join(data_dir, f"listings_temp_{file_idx}.json.gz")

        print(f"\n[{file_idx+1}/{len(LISTING_FILES)}] Downloading metadata: listings_{file_idx}.json.gz …")
        try:
            urllib.request.urlretrieve(gz_url, temp_gz_path)
        except Exception as e:
            print(f"  ✗ Failed: {e}  — skipping this file.")
            continue

        parsed_this_file = 0
        with gzip.open(temp_gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                if len(catalog_rows) >= total_target:
                    break
                try:
                    item = json.loads(line)

                    # ── Title ──────────────────────────────────────────────
                    names = item.get("item_name", [])
                    title = ""
                    if isinstance(names, list) and names:
                        title = names[0].get("value", "")
                    elif isinstance(names, str):
                        title = names
                    if not title:
                        continue

                    # ── Category ───────────────────────────────────────────
                    raw_cat = item.get("product_type", [{}])
                    category = (raw_cat[0].get("value", "General")
                                if raw_cat else "General")
                    category = category.replace("_", " ").title()

                    # ── Diversity cap ──────────────────────────────────────
                    if category_counts[category] >= max_per_category:
                        continue

                    # ── Image ──────────────────────────────────────────────
                    main_image = item.get("main_image_id", "")
                    if not main_image:
                        continue

                    item_id        = item.get("item_id", f"AMZ_{len(catalog_rows)}")
                    img_url        = f"https://m.media-amazon.com/images/I/{main_image}.jpg"
                    clean_filename = f"AMZ_{item_id}.jpg"
                    local_img_path = os.path.join(img_dir, clean_filename)
                    rel_img_path   = os.path.join(".", "data", "data_images", clean_filename)

                    # ── Description ────────────────────────────────────────
                    bullets   = item.get("bullet_point", [])
                    desc_text = " ".join(
                        b.get("value", "") for b in bullets if isinstance(b, dict)
                    ) if isinstance(bullets, list) else ""

                    download_tasks.append((img_url, local_img_path))
                    catalog_rows.append({
                        "product_id":         f"AMZ_{item_id}",
                        "title":              title[:150],
                        "category":           category,
                        "image_path":         rel_img_path,
                        "description":        desc_text[:300],
                        "predicted_category": None,
                        "color":              None,
                        "material":           None,
                        "style":              None,
                    })
                    category_counts[category] += 1
                    parsed_this_file += 1

                except Exception:
                    continue

        # Cleanup temp gz
        if os.path.exists(temp_gz_path):
            os.remove(temp_gz_path)

        print(f"  [OK] Parsed {parsed_this_file} products  |  Running total: {len(catalog_rows)}")

    print(f"\n{'-'*60}")
    print(f"Total products queued : {len(catalog_rows)}")
    print(f"Unique categories     : {len(category_counts)}")
    top5 = sorted(category_counts.items(), key=lambda x: -x[1])[:5]
    print(f"Top 5 categories      : {top5}")
    print(f"{'-'*60}")

    # ── Parallel Image Download ───────────────────────────────────────────────
    print(f"\nDownloading {len(download_tasks)} images on {DOWNLOAD_WORKERS} threads...")
    successful = 0
    failed     = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_single_image, t): t for t in download_tasks}
        for i, future in enumerate(as_completed(futures), 1):
            ok = future.result()
            if ok:
                successful += 1
            else:
                failed += 1
            if i % 1000 == 0 or i == len(download_tasks):
                print(f"  Progress: {i}/{len(download_tasks)}  OK={successful}  FAIL={failed}")

    # ── Save CSV (only rows with verified images) ─────────────────────────────
    print("\nBuilding final catalog CSV ...")
    df = pd.DataFrame(catalog_rows)
    df["_local"] = df["image_path"].apply(
        lambda p: os.path.join(base_dir, p.lstrip("./").lstrip("\\"))
    )
    df = df[df["_local"].apply(
        lambda p: os.path.exists(p) and os.path.getsize(p) > 500
    )].drop(columns=["_local"]).reset_index(drop=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("  [DONE] Scrape Complete!")
    print(f"  Products in catalog : {len(df)}")
    print(f"  Images downloaded   : {successful}")
    print(f"  CSV saved to        : {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    fetch_amazon_physical_catalog()
