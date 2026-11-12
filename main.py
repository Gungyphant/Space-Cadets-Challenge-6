import numpy as np
from PIL import Image as ImageModule
from PIL.Image import Image

from matrix import Matrix

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

    # Normalise g:
    normalised_g = g.normalise(255)

    return normalised_g

def sobel(image: Image) -> Image:
    # Convert to greyscale:
    image = image.convert("L")

    # Convert it to a matrix
    image_as_matrix = Matrix(image)

    # Run the sobel filter over the matrix
    sobel_image_as_matrix = _matrix_sobel(image_as_matrix)

    # Convert it back to an Image
    sobel_image = sobel_image_as_matrix.to_image()

    return sobel_image


if __name__ == "__main__":
    with ImageModule.open("Test images/Utah teapot.jpg") as im:
        # Convert the image to 100x100:
        # im = im.resize((100, 100))

        im.show()
        sobel_im = sobel(im)
        sobel_im.show()
