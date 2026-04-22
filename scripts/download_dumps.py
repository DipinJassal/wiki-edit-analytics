#!/usr/bin/env python3
"""Download Wikimedia XML stub-meta-history dumps."""

import argparse
import os
import sys
import requests


BASE_URL = "https://dumps.wikimedia.org/{wiki}/{date}/{wiki}-{date}-stub-meta-history{part}.xml.gz"


def get_available_parts(wiki, date, session):
    index_url = f"https://dumps.wikimedia.org/{wiki}/{date}/{wiki}-{date}-stub-meta-history.xml.gz"
    parts = []
    # Try numbered parts first (enwiki has many), then single file
    for part_num in range(1, 30):
        url = BASE_URL.format(wiki=wiki, date=date, part=part_num)
        r = session.head(url, allow_redirects=True)
        if r.status_code == 200:
            parts.append((part_num, url, int(r.headers.get('content-length', 0))))
        else:
            # Try without part number (single file wikis)
            if part_num == 1:
                url_single = f"https://dumps.wikimedia.org/{wiki}/{date}/{wiki}-{date}-stub-meta-history.xml.gz"
                r2 = session.head(url_single, allow_redirects=True)
                if r2.status_code == 200:
                    parts.append((None, url_single, int(r2.headers.get('content-length', 0))))
            break
    return parts


def download_file(url, dest_path, session):
    if os.path.exists(dest_path):
        remote_size = int(session.head(url).headers.get('content-length', 0))
        local_size = os.path.getsize(dest_path)
        if remote_size > 0 and local_size == remote_size:
            print(f"  Skipping {os.path.basename(dest_path)} (already complete)")
            return

    print(f"  Downloading {os.path.basename(dest_path)} ...")
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.1f}%)", end='', flush=True)
        print()
    print(f"  Done: {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Wikimedia XML dumps")
    parser.add_argument('--wiki', nargs='+', default=['enwiki'], help='Wiki(s) to download')
    parser.add_argument('--date', default='20260201', help='Dump date (YYYYMMDD)')
    parser.add_argument('--output', default='data/dumps', help='Output directory')
    parser.add_argument('--parts', type=int, default=1, help='Number of part files to download per wiki (0 = all)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'wiki-edit-analytics/1.0'})

    for wiki in args.wiki:
        print(f"\n=== {wiki} ===")
        wiki_dir = os.path.join(args.output, wiki)
        os.makedirs(wiki_dir, exist_ok=True)

        parts = get_available_parts(wiki, args.date, session)
        if not parts:
            print(f"  No dump files found for {wiki} on {args.date}")
            continue

        limit = args.parts if args.parts > 0 else len(parts)
        for part_num, url, size in parts[:limit]:
            fname = os.path.basename(url)
            dest = os.path.join(wiki_dir, fname)
            print(f"  Part {part_num}: {size / 1e6:.0f} MB  →  {dest}")
            download_file(url, dest, session)


if __name__ == '__main__':
    main()
