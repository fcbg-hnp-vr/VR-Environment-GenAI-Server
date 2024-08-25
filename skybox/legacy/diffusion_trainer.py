"""
Training pipeline for a diffusion network.
"""

import random

from skybox.diffusion import generate_images
from skybox.legacy.equirectangular_checker import score_image


def random_sentence():
    """Generate random sentences."""
    # Sets of words
    adjectives = ("quick", "lazy", "smart", "cute", "red")
    nouns = ("dog", "cat", "bird", "apple", "car")
    verbs = ("runs", "eats", "hops", "jumps", "drives")
    adverbs = ("quickly", "slowly", "carefully", "loudly", "eagerly")

    sentence = " ".join(map(random.choice, (adjectives, nouns, verbs, adverbs)))
    return sentence


def generate():
    """Generate a new image."""
    prompt = random_sentence() + " monoscopic 360 equirectangular"
    print(prompt)
    image = generate_images(prompt)[0]
    image.show()
    return image


def evaluate(img):
    """Evaluates a given image quality."""
    return score_image(img)


if __name__ == "__main__":
    for _ in range(5):
        score = evaluate(generate())
        print(f"Borders variation: {score}")
