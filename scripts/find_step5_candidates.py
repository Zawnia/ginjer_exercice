from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import bigquery

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ginjer_exercice.config import get_settings
from ginjer_exercice.data_access.bigquery_client import BigQueryClient
from ginjer_exercice.schemas.ad import Ad, Brand


BRANDS = (Brand.BALENCIAGA, Brand.LOUIS_VUITTON)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def _build_bigquery_client() -> BigQueryClient:
    settings = get_settings()
    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

    project_id = settings.gcp_project_id or "ginjer-440122"
    return BigQueryClient(bq_client=bigquery.Client(project=project_id))


def _truncate(value: str | None, max_chars: int = 100) -> str:
    if not value:
        return "-"

    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _media_kind(url: str) -> str:
    path = urlparse(url).path.lower()
    return "video" if path.endswith(VIDEO_EXTENSIONS) else "image"


def _media_summary(ad: Ad) -> str:
    counts = {"image": 0, "video": 0}
    for url in ad.media_urls:
        counts[_media_kind(url)] += 1

    return f"total={len(ad.media_urls)} image={counts['image']} video={counts['video']}"


def _print_ad(ad: Ad, index: int) -> None:
    print(f"{index}. ad_id={ad.platform_ad_id}")
    print(f"   media: {_media_summary(ad)}")

    if not ad.texts:
        print("   text: -")
    for text_index, text in enumerate(ad.texts, start=1):
        print(f"   text[{text_index}]")
        print(f"     body_text: {_truncate(text.body_text)}")
        print(f"     caption:   {_truncate(text.caption)}")
        print(f"     url:       {_truncate(text.url)}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find real BigQuery ads that are useful for manual step5_fallback checks."
    )
    parser.add_argument("--limit", type=int, default=5, help="Ads to list per brand.")
    args = parser.parse_args()

    print("BigQueryClient method used: fetch_ads_by_brand(brand: Brand, limit: int)")
    print("Brand enum values used: BALENCIAGA, LOUIS_VUITTON")
    print("Media type display: video by file extension; every other media URL shown as image")
    print()

    client = _build_bigquery_client()
    for brand in BRANDS:
        ads = client.fetch_ads_by_brand(brand, limit=args.limit)

        print(f"== {brand.value} ({len(ads)} ads) ==")
        if not ads:
            print("No ads returned.")
            print()
            continue

        for index, ad in enumerate(ads, start=1):
            _print_ad(ad, index)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
