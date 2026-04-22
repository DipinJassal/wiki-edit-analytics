"""Parse Wikimedia XML stub-meta-history dumps into Parquet.

Uses streaming iterparse instead of Spark/spark-xml to avoid OOM on
large unsplittable .xml.gz files. Writes Parquet in batches via PyArrow.
"""

import gzip
import os
import sys
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from xml.etree import ElementTree as ET

NS = 'http://www.mediawiki.org/xml/export-0.11/'

SCHEMA = pa.schema([
    pa.field('wiki',             pa.string()),
    pa.field('page_id',          pa.int64()),
    pa.field('page_title',       pa.string()),
    pa.field('page_namespace',   pa.int32()),
    pa.field('revision_id',      pa.int64()),
    pa.field('parent_id',        pa.int64()),
    pa.field('timestamp',        pa.string()),
    pa.field('contributor_id',   pa.int64()),
    pa.field('contributor_name', pa.string()),
    pa.field('is_anonymous',     pa.bool_()),
    pa.field('comment',          pa.string()),
    pa.field('text_bytes',       pa.int64()),
])

BATCH_SIZE = 50_000


def tag(local):
    return f'{{{NS}}}{local}'


def parse_dump(path, wiki, out_dir, batch_size=BATCH_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{wiki}.parquet')
    writer = None

    rows = {k: [] for k in SCHEMA.names}
    total = 0

    opener = gzip.open if path.endswith('.gz') else open

    with opener(path, 'rb') as f:
        context = ET.iterparse(f, events=('end',))

        current_page = {}

        for _, elem in context:
            t = elem.tag

            if t == tag('page'):
                ns = int(current_page.get('ns', -1))
                if ns == 0:
                    pass  # revisions were already emitted
                current_page = {}
                elem.clear()

            elif t == tag('ns'):
                current_page['ns'] = elem.text or '0'

            elif t == tag('title'):
                current_page['title'] = elem.text or ''

            elif t == tag('id') and 'page_id' not in current_page:
                current_page['page_id'] = int(elem.text or 0)

            elif t == tag('revision'):
                ns = int(current_page.get('ns', -1))
                if ns != 0:
                    elem.clear()
                    continue

                rev_id    = elem.findtext(tag('id')) or '0'
                parent_id = elem.findtext(tag('parentid')) or '0'
                timestamp = elem.findtext(tag('timestamp')) or ''
                comment   = elem.findtext(tag('comment')) or ''

                contrib = elem.find(tag('contributor'))
                if contrib is not None:
                    username = contrib.findtext(tag('username'))
                    ip       = contrib.findtext(tag('ip'))
                    cid_text = contrib.findtext(tag('id'))
                    if username:
                        cname  = username
                        cid    = int(cid_text) if cid_text else 0
                        is_anon = False
                    else:
                        cname  = ip or ''
                        cid    = 0
                        is_anon = True
                else:
                    cname, cid, is_anon = '', 0, True

                # Skip anonymous for similarity analysis
                if cid == 0:
                    elem.clear()
                    continue

                text_elem  = elem.find(tag('text'))
                text_bytes = int(text_elem.get('bytes', 0)) if text_elem is not None else 0

                rows['wiki'].append(wiki)
                rows['page_id'].append(current_page.get('page_id', 0))
                rows['page_title'].append(current_page.get('title', ''))
                rows['page_namespace'].append(0)
                rows['revision_id'].append(int(rev_id))
                rows['parent_id'].append(int(parent_id))
                rows['timestamp'].append(timestamp)
                rows['contributor_id'].append(cid)
                rows['contributor_name'].append(cname)
                rows['is_anonymous'].append(is_anon)
                rows['comment'].append(comment)
                rows['text_bytes'].append(text_bytes)

                total += 1
                elem.clear()

                if total % batch_size == 0:
                    batch = pa.record_batch(
                        [pa.array(rows[k], type=SCHEMA.field(k).type) for k in SCHEMA.names],
                        schema=SCHEMA,
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, SCHEMA, compression='snappy')
                    writer.write_batch(batch)
                    rows = {k: [] for k in SCHEMA.names}
                    print(f'\r  {wiki}: {total:,} revisions written...', end='', flush=True)

        # flush remainder
        if any(rows[k] for k in SCHEMA.names):
            batch = pa.record_batch(
                [pa.array(rows[k], type=SCHEMA.field(k).type) for k in SCHEMA.names],
                schema=SCHEMA,
            )
            if writer is None:
                writer = pq.ParquetWriter(out_path, SCHEMA, compression='snappy')
            writer.write_batch(batch)

    if writer:
        writer.close()

    print(f'\n  Done: {wiki} → {out_path}  ({total:,} revisions)')
    return total


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    wikis = sys.argv[1:] if len(sys.argv) > 1 else cfg['wikis']

    for wiki in wikis:
        dump_dir = os.path.join(cfg['paths']['dumps'], wiki)
        out_dir  = cfg['paths']['parquet']

        files = sorted(
            os.path.join(dump_dir, fn)
            for fn in os.listdir(dump_dir)
            if fn.endswith('.xml.gz') or fn.endswith('.xml')
        )
        if not files:
            print(f'No dump files found in {dump_dir}')
            continue

        for path in files:
            print(f'Parsing {path} ...')
            parse_dump(path, wiki, out_dir)


if __name__ == '__main__':
    main()
