import gc
import numpy as np


def npz_load_polygons(db_npy_path: str, max_polygons: int | None):
    # print(f"Loading polygons from {db_npy_path}...")
    data = np.load(db_npy_path)
    max_polygons = 5000 if max_polygons is None else max_polygons
    loaded_polygons = min(len(data["lengths"]), max_polygons)
    data_dict = {
        "polygons": data["polygons"][:loaded_polygons],
        "lengths": data["lengths"][:loaded_polygons],
    }
    del data
    gc.collect()
    return data_dict


def load_list_from_dict(data_dict: dict):
    data_loaded_list = [
        data_dict["polygons"][i, : data_dict["lengths"][i]]
        for i in range(len(data_dict["lengths"]))
    ]
    del data_dict
    gc.collect()
    return data_loaded_list


# np_array is of shape (N_point, 2)
def rotate_np_array_around_point(point, np_array, angle):
    orig_x, orig_y = point
    arr_x, arr_y = np_array[:, 0], np_array[:, 1]

    rotated_x = (
        orig_x + np.cos(angle) * (arr_x - orig_x) - np.sin(angle) * (arr_y - orig_y)
    )
    rotated_y = (
        orig_y + np.sin(angle) * (arr_x - orig_x) + np.cos(angle) * (arr_y - orig_y)
    )
    return np.stack([rotated_x, rotated_y]).T


def _polygon_centroid(verts):
    # Ensure verts is a numpy array
    verts = np.asarray(verts)

    # Get x and y coordinates
    x = verts[:, 0]
    y = verts[:, 1]

    # Get "rolled" coordinates (x_{i+1}, y_{i+1})
    x_plus_1 = np.roll(x, -1)
    y_plus_1 = np.roll(y, -1)

    # Calculate the common term (x_i * y_{i+1} - x_{i+1} * y_i)
    # This is 2x the signed area of the triangle (0, P_i, P_{i+1})
    common_term = (x * y_plus_1) - (x_plus_1 * y)

    # Calculate 2x the total signed area (A)
    # A = 0.5 * np.sum(common_term)
    # We need 6A for the centroid formula
    six_A = 3 * np.sum(common_term)

    # Calculate centroid coordinates
    # C_x = (1 / 6A) * sum((x_i + x_{i+1}) * common_term)
    # C_y = (1 / 6A) * sum((y_i + y_{i+1}) * common_term)
    Cx = np.sum((x + x_plus_1) * common_term) / six_A
    Cy = np.sum((y + y_plus_1) * common_term) / six_A

    return np.array([Cx, Cy])


def rotate_np_array(array: np.ndarray, rotation: float) -> np.ndarray:
    point = _polygon_centroid(array)
    return rotate_np_array_around_point(point, array, rotation)
