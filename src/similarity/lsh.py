"""LSH banding for candidate pair generation from MinHash signatures."""

from collections import defaultdict


class LSH:
    def __init__(self, num_bands=32, rows_per_band=4):
        """
        Args:
            num_bands: number of bands (b)
            rows_per_band: rows per band (r)
            Signature length must equal b * r
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets = [defaultdict(set) for _ in range(num_bands)]

    def index(self, item_id, signature):
        """Add an item to the LSH index.

        Args:
            item_id: identifier (e.g., wiki name or article title)
            signature: MinHash signature array of length num_bands * rows_per_band
        """
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_key = hash(tuple(signature[start:end].tolist()))
            self.buckets[band_idx][band_key].add(item_id)

    def get_candidates(self):
        """Return all candidate pairs sharing at least one bucket.

        Returns:
            set of (item_a, item_b) tuples where item_a < item_b
        """
        candidates = set()
        for band_buckets in self.buckets:
            for items in band_buckets.values():
                if len(items) >= 2:
                    sorted_items = sorted(items)
                    for i in range(len(sorted_items)):
                        for j in range(i + 1, len(sorted_items)):
                            candidates.add((sorted_items[i], sorted_items[j]))
        return candidates

    def reset(self):
        self.buckets = [defaultdict(set) for _ in range(self.num_bands)]
