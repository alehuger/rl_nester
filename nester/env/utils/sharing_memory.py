from multiprocessing import shared_memory

import numpy as np

from nester.env.utils.utils import npz_load_polygons


class SharedPolygonData:
    def __init__(self, data_path: str, max_polygons: int):
        self.data_path = data_path
        self.max_polygons = max_polygons

    def create_shared_memories(self):
        data = npz_load_polygons(self.data_path, self.max_polygons)
        shared_polygons_memory = shared_memory.SharedMemory(
            create=True, size=data["polygons"].nbytes
        )
        shared_lengths_memory = shared_memory.SharedMemory(
            create=True, size=data["lengths"].nbytes
        )
        shared_polygons_array = np.ndarray(
            data["polygons"].shape, dtype=np.float32, buffer=shared_polygons_memory.buf
        )
        shared_lengths_array = np.ndarray(
            data["lengths"].shape,
            dtype=np.int32,
            buffer=shared_lengths_memory.buf,
        )
        shared_polygons_array[:] = data["polygons"]
        shared_lengths_array[:] = data["lengths"]

        self.shared_polygons_memory_name = shared_polygons_memory.name
        self.shared_lengths_memory_name = shared_lengths_memory.name

        return {
            "polygons_name": self.shared_polygons_memory_name,
            "lengths_name": self.shared_lengths_memory_name,
            "shape": data["polygons"].shape,
        }

    def close(self):
        shm_polygons = shared_memory.SharedMemory(name=self.shared_polygons_memory_name)
        shm_lengths = shared_memory.SharedMemory(name=self.shared_lengths_memory_name)
        shm_polygons.close()
        shm_lengths.close()
        shm_polygons.unlink()
        shm_lengths.unlink()


class PolygonsShmView:
    """One POSIX shared segment per run; workers mmap read-only and copy only sampled rows."""

    def __init__(self, shared_memory_dict: dict):
        self._polygons_shm = shared_memory.SharedMemory(name=shared_memory_dict["polygons_name"])
        self._lengths_shm = shared_memory.SharedMemory(name=shared_memory_dict["lengths_name"])
        self._shape = tuple(shared_memory_dict["shape"])
        self._polygons = np.ndarray(self._shape, dtype=np.float32, buffer=self._polygons_shm.buf)
        self._lengths = np.ndarray(self._shape[0], dtype=np.int32, buffer=self._lengths_shm.buf)

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, i):
        L = int(self._lengths[i])
        return self._polygons[i, :L].copy()


def load_polygons_from_shared_memory(shared_memory_dict: dict[str, str]) -> PolygonsShmView:
    return PolygonsShmView(shared_memory_dict)


class SharedBenchData:
    """POSIX SHM for one validation instance row (polygons block + per-polygon lengths)."""

    def __init__(self, polygons_block: np.ndarray, lengths_row: np.ndarray):
        self._polygons = np.asarray(polygons_block, dtype=np.float32)
        self._lengths = np.asarray(lengths_row, dtype=np.int32)
        self.shared_polygons_memory_name: str | None = None
        self.shared_lengths_memory_name: str | None = None

    def create_shared_memories(self) -> dict[str, str | tuple]:
        shm_poly = shared_memory.SharedMemory(create=True, size=self._polygons.nbytes)
        shm_len = shared_memory.SharedMemory(create=True, size=self._lengths.nbytes)
        arr_poly = np.ndarray(self._polygons.shape, dtype=np.float32, buffer=shm_poly.buf)
        arr_len = np.ndarray(self._lengths.shape, dtype=np.int32, buffer=shm_len.buf)
        arr_poly[:] = self._polygons
        arr_len[:] = self._lengths
        self.shared_polygons_memory_name = shm_poly.name
        self.shared_lengths_memory_name = shm_len.name
        return {
            "polygons_name": self.shared_polygons_memory_name,
            "lengths_name": self.shared_lengths_memory_name,
            "shape": self._polygons.shape,
        }

    def close(self) -> None:
        if self.shared_polygons_memory_name is None:
            return
        shm_poly = shared_memory.SharedMemory(name=self.shared_polygons_memory_name)
        shm_len = shared_memory.SharedMemory(name=self.shared_lengths_memory_name)
        shm_poly.close()
        shm_len.close()
        shm_poly.unlink()
        shm_len.unlink()
