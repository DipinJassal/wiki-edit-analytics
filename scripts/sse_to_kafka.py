#!/usr/bin/env python3
"""Connect to Wikimedia EventStreams SSE and publish to Kafka."""

import json
import time
import logging
import requests
from collections import defaultdict
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

STREAM_URL = 'https://stream.wikimedia.org/v2/stream/recentchange'
KAFKA_SERVERS = 'localhost:9092'
TOPIC = 'wiki-edits'
TARGET_WIKIS = {'enwiki', 'dewiki', 'frwiki', 'jawiki', 'eswiki'}

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=10,
    batch_size=32768,
)

counts = defaultdict(int)
total = 0
last_report = time.time()
last_event_id = None


def iter_sse(response):
    """Minimal SSE parser over a streaming requests response."""
    data_lines = []
    event_id = None
    for raw in response.iter_lines(decode_unicode=True):
        if raw.startswith('id:'):
            event_id = raw[3:].strip()
        elif raw.startswith('data:'):
            data_lines.append(raw[5:].strip())
        elif raw == '' and data_lines:
            yield event_id, '\n'.join(data_lines)
            data_lines = []
            event_id = None


def report_stats():
    global last_report
    now = time.time()
    if now - last_report >= 30:
        log.info(f"Total events sent: {total} | Per wiki: {dict(counts)}")
        last_report = now


while True:
    try:
        log.info(f"Connecting to {STREAM_URL} ...")
        headers = {
            'Accept': 'text/event-stream',
            'User-Agent': 'wiki-edit-analytics/1.0 (https://github.com/student/wiki-edit-analytics; dipinjassal189@gmail.com)',
        }
        if last_event_id:
            headers['Last-Event-ID'] = last_event_id

        with requests.get(STREAM_URL, stream=True, headers=headers, timeout=60) as resp:
            resp.raise_for_status()
            for eid, data in iter_sse(resp):
                if eid:
                    last_event_id = eid
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                wiki = event.get('wiki')
                if (wiki in TARGET_WIKIS
                        and event.get('namespace') == 0
                        and event.get('type') == 'edit'):
                    producer.send(TOPIC, value=event)
                    counts[wiki] += 1
                    total += 1
                    report_stats()

    except Exception as e:
        log.error(f"Connection lost: {e}. Reconnecting in 5s...")
        time.sleep(5)
