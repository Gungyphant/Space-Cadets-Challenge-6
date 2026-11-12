from math import cos, sin, radians
from typing import Generator

from PIL import Image as ImageModule
from PIL.Image import Image

from matrix import Matrix


def get_points_on_circle(centre_x: int, centre_y: int, radius: int) -> Generator[tuple[int, int]]:
    for theta in range(360):  # TODO: determine the theta difference
        x = centre_x + radius * cos(radians(theta))
        y = centre_y + radius * sin(radians(theta))
        yield int(x), int(y)


SOBEL_X_KERNEL = Matrix([
    [-1,  0, +1],
    [-2,  0, +2],
    [-1,  0, +1]
])

SOBEL_Y_KERNEL = Matrix([
    [-1, -2, -1],
    [ 0,  0,  0],
    [+1, +2, +1]
])

def _matrix_sobel(matrix: Matrix) -> Matrix:
    # Find horizontal edges
    g_x = SOBEL_X_KERNEL * matrix

    # Find vertical edges
    g_y = SOBEL_Y_KERNEL * matrix

    # Find all edges (via Pythagoras)
    g = (g_x ** 2 + g_y ** 2).sqrt()

    return g

def sobel(image: Image) -> Image:
    """Use the Sobel filter to find edges in image

    :param image: The image to find edges in. Must be monochrome
    :return: The image with edges detected
    """
    # Convert it to a matrix
    image_as_matrix = Matrix(image)

    # Run the Sobel filter over the matrix
    sobel_image_as_matrix = _matrix_sobel(image_as_matrix)

    # Normalise the matrix to valid pixel values
    sobel_image_as_matrix = sobel_image_as_matrix.normalise(255)

    # Convert it back to an Image
    sobel_image = sobel_image_as_matrix.to_image()

    return sobel_image


def _matrix_hough_parameter_space(matrix: Matrix, radius: int) -> Matrix:
    result = Matrix(matrix.width, matrix.height)

    for y, line in enumerate(matrix.values):
        for x, val in enumerate(line):
            if val:
                for (circle_x, circle_y) in get_points_on_circle(x, y, radius):
                    if (
                        0 <= circle_y < result.height
                        and
                        0 <= circle_x < result.width
                    ):
                        result[circle_y][circle_x] += val

    return result


def hough_parameter_space(image: Image, radius: int) -> Image:
    """Find the Hough parameter space of image, for a given radius

    :param image: The image to find the Hough parameter space of. Must be an
                  edge-detected image, like those output by sobel()
    :param radius: The radius of circle to search for
    :return: An image where the brighter a pixel, the more likely it is to be
             the centre of a circle
    """
    # Convert it to a matrix
    image_as_matrix = Matrix(image)

    # Run the Hough transform over the matrix
    hough_image_as_matrix = _matrix_hough_parameter_space(image_as_matrix, radius)

    # Normalise the matrix
    hough_image_as_matrix = hough_image_as_matrix.normalise(255)

    # Convert it back to an Image
    hough_image = hough_image_as_matrix.to_image()

    return hough_image


def _matrix_hough(matrix: Matrix, *, debug: bool = False) -> tuple[int, int, int]:
    best = -1
    result = -1, -1, -1

    # For every possible radius, find the most likely centre and how likely it
    #  is. Find the radius that maximises that likelihood and output it along
    #  with the coordinates of the most likely centre for that radius
    for radius in range(1, min(matrix.width, matrix.height) // 2):
        if debug:
            print(f"Radius {radius} ", end="")
        brightness, best_x, best_y = _matrix_hough_parameter_space(
            matrix,
            radius
        ).max_and_indices()
        if debug:
            print(f"centre at ({best_x}, {best_y}), score {brightness}", end="")
        if brightness > best:
            best = brightness
            result = best_x, best_y, radius
            if debug:
                print(" new best")
        elif debug:
            print()

    return result


def hough(image: Image, *, debug: bool = False) -> tuple[int, int, int]:
    """Finds the most likely circle in an image

    :param image: The image to find the circle in. Must be an
                  edge-detected image, like those output by sobel()
    :param debug: Determines if debug messages should be printed. Default False
    :return: The x and y of the centre of the circle, and its radius
    """
    # Convert it to a matrix
    image_as_matrix = Matrix(image)

    # Run the Hough Transform over the matrix
    x, y, radius = _matrix_hough(image_as_matrix, debug=debug)

    return x, y, radius


if __name__ == "__main__":
    with ImageModule.open("Test images/White circle with black border.webp") as im:
        # Convert the image to 100x100:
        im = im.resize((100, 100))

        # Convert to greyscale:
        im = im.convert("L")

        im.show()

        sobel_im = sobel(im)
        sobel_im.show()

        hough_im = hough_parameter_space(sobel(im), 41)
        hough_im.show()

        print(hough(sobel(im), debug=True))
