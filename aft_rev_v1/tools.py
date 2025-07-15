import skimage
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def base_beam( short_width,width,short_end_center, resolution):
    length = resolution

    left_end_bottom = (0,((resolution)/2)-width/2)
    left_end_top = (0,((resolution)/2)+width/2)

    right_end_bottom = (length,(short_end_center)-short_width/2)
    right_end_top = (length,(short_end_center)+short_width/2)

    rr,cc=skimage.draw.polygon([left_end_bottom[1], left_end_top[1], right_end_top[1], right_end_bottom[1]],

    [left_end_bottom[0], left_end_top[0], right_end_top[0], right_end_bottom[0]] )

    beam = np.zeros((resolution+1, resolution+1), dtype=int)
    # print(img.shape)
    # print(rr,cc)
    beam[rr, cc] = 1
    beam = beam[1:,:-1]
    return beam


def remove_polygon_material(n_sides, radius, center, angle_offset,base_beam):
    """
    Create a binary image of base beam by by removing a polygonal material.
    Parameters:
    n_sides (int): Number of sides of the polygon.
    radius (int): Radius of the circumscribed circle around the polygon.
    center (tuple): Center coordinates of the polygon (x, y).
    angle_offset (float): Rotation angle in degrees.

    Returns:
    numpy.ndarray: The generated binary image with the rotated polygonal removed material.
    """
    size=(64, 64)
    remove_material = np.ones(size)  # Start with a white canvas
    angle_offset_radians = np.radians(angle_offset)
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle_offset_radians
    vertices = np.array([(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles])

    # Rounding vertices to integer values to use them as image coordinates
    vertices = np.round(vertices).astype(int)

    # Creating the polygon on the image
    rr, cc = skimage.draw.polygon(vertices[:, 1], vertices[:, 0], remove_material.shape)
    remove_material[rr, cc] = 0  # Setting the polygon area to 0 (cutout)
    new_beam = base_beam * remove_material

    return new_beam

def add_polygon_material(n_sides, radius, center, angle_offset,base_beam):
    """
    Create a binary image of base beam by adding a polygonal material.

    Parameters:
    n_sides (int): Number of sides of the polygon.
    radius (int): Radius of the circumscribed circle around the polygon.
    center (tuple): Center coordinates of the polygon (x, y).
    angle_offset (float): Rotation angle in degrees.

    Returns:
    numpy.ndarray: The generated binary image with the rotated polygonal added material.
    """
    size=(64, 64)
    add_material = np.zeros(size)  # Start with a white canvas
    angle_offset_radians = np.radians(angle_offset)
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle_offset_radians
    vertices = np.array([(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles])

    # Rounding vertices to integer values to use them as image coordinates
    vertices = np.round(vertices).astype(int)

    # Creating the polygon on the image
    rr, cc = skimage.draw.polygon(vertices[:, 1], vertices[:, 0], add_material.shape)
    add_material[rr, cc] = 1  # Setting the polygon area to 1 (add material to base_beam)

    plt.imshow(add_material)
    plt.show()
    new_beam = np.logical_or(base_beam, add_material.astype(bool)).astype(int)

    return new_beam

def check_geometry(geo):
    """
    Check the validity of a geometric structure represented by an array.

    This function examines a geometric structure represented as a binary array (geo)
    to determine if it adheres to certain validity criteria. Specifically, it checks
    if there are two or more disjoint regions of 1s, and whether the geometry touches
    the left and right boundaries of the array.

    Parameters:
    - geo (array-like): A 2D array representing the geometric structure, where 1s represent
                        the structure and 0s represent empty space.

    Returns:
    - str: A message indicating whether the geometry is valid, does not touch the left or
           right boundary, or contains two or more disjoint regions of 1s.

    The function first uses the `label` function to identify distinct regions within the
    geometry. It then checks the sum of the values in the first and last columns of `geo`
    to determine if the geometry touches the left and right boundaries, respectively.
    """
    labeled_array, num_features = label(geo)
    out = "Geometry is valid."
    # Determine if there are two disjoint regions of 1s
    two_disjoint_ones = num_features >= 2
    if two_disjoint_ones:
        out = ("The geometry is not valid. Try changing the cutout parameters.")

    print(geo[:,0].sum())
    print(geo[:,-1].sum())

    if not geo[:,0].sum() > 0:
        out=("The geometry Does not touch the left boundary, change cutout parameters.")

    if not geo[:,-1].sum() > 0:
        out= ("The geometry Does not touch the right boundary, change parameters.")
    
    return out
