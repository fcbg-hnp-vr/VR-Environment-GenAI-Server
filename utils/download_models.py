"""
Simple utility script that forces the download of all models.

Just load the script, and the models should get installed.
"""
import warnings

import asr.speech_to_text
import environment.depth_generation
import environment.depth_inpainting
import skybox.diffusion
import skybox.inpainting


def download_alpha_pipelines():
    """Models not used in production yet."""
    print("Starting loading models")
    print("Loading ControlNet inpainting...")
    environment.depth_inpainting.get_inpaint_depth_pipeline()
    print("Loading depth generation...")
    environment.depth_generation.get_depth_pipeline()

    print("Finished loading models in alpha with success!")


def download_production_pipelines():
    """Load all pipelines used in the server in order to download the associated models."""
    print("Starting loading models")
    print("Loading speech recognition...")
    asr.speech_to_text.get_asr_model()
    print("Loading image generation...")
    skybox.diffusion.get_image_generation_pipeline()
    print("Loading image refinement...")
    skybox.diffusion.get_image_refinement_pipeline()
    print("Loading inpainting...")
    skybox.inpainting.get_inpainting_pipeline()

    print("Finished loading models with success!")


def load_production_pipelines():
    """Load all pipelines used in the server in order to download the associated models."""
    warnings.warn(
        "load_production_pipelines is deprecated,"
        " use download_production_pipelines instead."
    )
    return download_production_pipelines()


if __name__ == "__main__":
    download_production_pipelines()
