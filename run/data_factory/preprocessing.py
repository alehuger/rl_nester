"""
Build the land-polygon training database (NPZ) from OSM data.
"""

from __future__ import annotations

import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import shapefile
from shapely.affinity import scale, translate
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

USE_PREPROCESSED_PKL = True
SHAPEFILE_PATH = PROJECT_ROOT / "data/raw_data/land-polygons-complete-3857/land_polygons.shp"
POLYGON_PKL_PATH = PROJECT_ROOT / "data/raw_data/land_polygon_big.pkl"
OUTPUT_DIR = PROJECT_ROOT / "data/final_db"
MAX_POLYGONS = None  # e.g. 10_000 for a quick dry run

COARSE_SIMPLIFY_TOLERANCE = 1.0
COARSE_MAX_VERTICES = 80  # only on raw shapefile; must stay before fine simplify
FINE_SIMPLIFY_TOLERANCE = 0.01
MIN_VERTICES = 4
MAX_VERTICES = 30
MIN_ISOPERIMETRIC_QUOTIENT = 0.15
TARGET_PERIMETER_MEAN = 10.0
TARGET_PERIMETER_STD = 1.0

TRAIN_TEST_SIZE = 240
RANDOM_SEED = 42
NUM_WORKERS = mp.cpu_count()
BATCH_SIZE = 2048


def preprocess_polygon(ring: np.ndarray, seed: int, *, from_raw_shapefile: bool) -> np.ndarray | None:
    if len(ring) < 3:
        return None
    try:
        polygon = Polygon(ring)
    except Exception:
        return None
    if not polygon.is_valid or polygon.is_empty:
        return None

    if from_raw_shapefile:
        polygon = polygon.simplify(COARSE_SIMPLIFY_TOLERANCE, preserve_topology=True)
        if len(polygon.exterior.coords) < 3 or len(polygon.exterior.coords) > COARSE_MAX_VERTICES:
            return None

    polygon = polygon.simplify(FINE_SIMPLIFY_TOLERANCE, preserve_topology=True)
    if polygon.length == 0:
        return None
    if (4 * np.pi * polygon.area) / (polygon.length**2) < MIN_ISOPERIMETRIC_QUOTIENT:
        return None

    centroid = polygon.centroid
    polygon = translate(polygon, -centroid.x, -centroid.y)

    np.random.seed(seed)
    target_perimeter = max(1.0, float(np.random.normal(TARGET_PERIMETER_MEAN, TARGET_PERIMETER_STD)))
    polygon = scale(
        polygon,
        xfact=target_perimeter / polygon.length,
        yfact=target_perimeter / polygon.length,
        origin="center",
    )
    if polygon.geom_type != "Polygon" or polygon.is_empty:
        return None

    ring = np.asarray(polygon.exterior.coords, dtype=np.float64)
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < MIN_VERTICES or len(ring) > MAX_VERTICES:
        return None
    return ring


def process_batch(batch: list[tuple]) -> list[np.ndarray]:
    pos = process_batch.pkl_pos  # set by pool initializer when input is pickle indices
    kept = []
    for ring_or_index, seed, from_raw_shapefile in batch:
        ring = np.asarray(pos[ring_or_index], dtype=np.float64) if pos is not None else ring_or_index
        result = preprocess_polygon(ring, seed, from_raw_shapefile=from_raw_shapefile)
        if result is not None:
            kept.append(result)
    return kept


process_batch.pkl_pos = None


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    pkl_pos = None
    dataset = None

    if USE_PREPROCESSED_PKL:
        dataset = np.load(POLYGON_PKL_PATH, allow_pickle=True)["pos"]
        n = len(dataset) if MAX_POLYGONS is None else min(MAX_POLYGONS, len(dataset))
        dataset = dataset[:n]
        pkl_pos = dataset
    else:
        reader = shapefile.Reader(str(SHAPEFILE_PATH))
        n = len(reader) if MAX_POLYGONS is None else min(MAX_POLYGONS, len(reader))
        dataset = []
        for shape in reader.shapes()[:n]:
            points, parts = shape.points, shape.parts
            ring = points[parts[0] : parts[1]] if len(parts) > 1 else points
            if len(ring) >= 2 and ring[0] == ring[-1]:
                ring = ring[:-1]
            dataset.append(np.asarray(ring, dtype=np.float64))

    seeds = rng.integers(0, 1_000_000, size=n, dtype=np.int64)
    batches = []
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        if pkl_pos is not None:
            batches.append([(i, int(seeds[i]), False) for i in range(start, end)])
        else:
            batches.append([(dataset[i], int(seeds[i]), True) for i in range(start, end)])

    pool_kwargs: dict = {"processes": NUM_WORKERS}
    if pkl_pos is not None:
        pool_kwargs["initializer"] = partial(setattr, process_batch, "pkl_pos")
        pool_kwargs["initargs"] = (pkl_pos,)

    polygons = []
    with mp.Pool(**pool_kwargs) as pool:
        for batch_out in pool.imap_unordered(process_batch, batches, chunksize=1):
            polygons.extend(batch_out)

    # Export padded arrays and train / test split
    n_out = len(polygons)
    lengths = np.fromiter((p.shape[0] for p in polygons), dtype=np.int64, count=n_out)
    padded = np.zeros((n_out, MAX_VERTICES, 2), dtype=np.float64)
    for i, poly in enumerate(polygons):
        padded[i, : poly.shape[0]] = poly

    perm = rng.permutation(n_out)
    n_test = min(TRAIN_TEST_SIZE, n_out)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "dataset.npz", polygons=padded, lengths=lengths)
    np.savez(OUTPUT_DIR / "train_db.npz", polygons=padded[train_idx], lengths=lengths[train_idx])
    np.savez(OUTPUT_DIR / "test_db.npz", polygons=padded[test_idx], lengths=lengths[test_idx])

    print(f"Saved {n_out} polygons to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
