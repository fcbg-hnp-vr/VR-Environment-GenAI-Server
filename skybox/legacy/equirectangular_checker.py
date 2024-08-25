"""
Evaluates how much an image variates from an equirectangular projection.
"""

import sys
from PIL import Image


def check_ratio(img):
    """Check if the image's aspect ratio is 2:1"""
    width, height = img.size
    return width / height >= 2


def define_boxes(img_size, subdivisions, pixels):
    """Define the boxes of an image divided by a given number of subdivisions."""
    width, height = img_size
    for i in range(subdivisions):
        box_range = i * width // subdivisions, (i + 1) * width // subdivisions
        box = (box_range[0], 0, box_range[1], pixels)
        opp = (box_range[0], height - pixels, box_range[1], height)
        yield box, opp
    for i in range(subdivisions):
        box_range = i * height // subdivisions, (i + 1) * height // subdivisions
        box = (0, box_range[0], pixels, box_range[1])
        opp = (width - pixels, box_range[0], width, box_range[1])
        yield box, opp


def check_boundaries(img, subdivisions=10, pixels=5):
    """Check if the boundaries of an image can be matched."""
    diff = 0
    for box, partner in define_boxes(img.size, subdivisions, pixels):
        regions = img.crop(box), img.crop(partner)
        means = [sum(regions[i].getdata()) / 255 / pixels / pixels for i in range(2)]
        diff += (means[1] - means[0]) ** 2
        # Method 2: diff = sum((regions[1].mirror - regions[0]) ** 2)
    return diff / subdivisions


def check_frontiers(img):
    """
    Return how different are the pixels on the opposite borders of an image.

    :param PIL.Image.Image img: Input image
    :return: Frontiers difference. 0 for identical, 1 if they are totally different.
    :rtype: float
    """
    diff = 0
    width, height = img.size
    for y in range(width):
        pixels = img.getpixel((0, y)), img.getpixel((width - 1, y))
        diff += ((pixels[1] - pixels[0]) / 255) ** 2
    for x in range(height):
        pixels = img.getpixel((x, 0)), img.getpixel((x, height - 1))
        diff += ((pixels[1] - pixels[0]) / 255) ** 2
    return diff / (width + height)


def score_image(img):
    """
    Return the likelyhood of the input image to be equirectangular.
    """
    variation = check_frontiers(img.convert("L"))
    return variation


def score_file(image_path):
    """Assigns a variation score to the image."""
    img = Image.open(image_path)
    variation = score_image(img)
    print("Boundary", check_boundaries(img), "frontier", check_frontiers(img))
    return variation


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide an image.")
        sys.exit(126)
    print("Value :", score_file(sys.argv[1]))
