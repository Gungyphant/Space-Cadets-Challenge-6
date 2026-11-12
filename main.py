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

    # Convert it to a matrix
    image_as_matrix = Matrix(image)

    # Run the Sobel filter over the matrix
    sobel_image_as_matrix = _matrix_sobel(image_as_matrix)

    # Normalise the matrix to valid pixel values
    sobel_image_as_matrix = sobel_image_as_matrix.normalise(255)

    # Convert it back to an Image
    sobel_image = sobel_image_as_matrix.to_image()

    return sobel_image


if __name__ == "__main__":
    with ImageModule.open("Test images/White circle with black border.webp") as im:
        # Convert the image to 100x100:
        im = im.resize((100, 100))

        # Convert to greyscale:
        im = im.convert("L")

        im.show()

        sobel_im = sobel(im)
        sobel_im.show()
