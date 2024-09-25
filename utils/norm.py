import numpy as np

def normalize_points(points):
    """
    Normalize a point cloud to fit within a unit cube [0, 1] in all dimensions.
    """
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Translate points to the origin
    points = points - min_vals

    # Scale points to fit within the unit cube
    scale = max_vals - min_vals
    normalized_points = points / scale

    return normalized_points