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

def sobel(image: Image) -> Image:
    # Convert to greyscale:
    image = image.convert("L")

    image_as_matrix = Matrix(image)

    g_x = SOBEL_X_KERNEL * image_as_matrix
    g_y = SOBEL_Y_KERNEL * image_as_matrix

    g = (g_x ** 2 + g_y ** 2).sqrt()

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
