"""
Simple utility script that forces the download of all models.

Just load the script, and the models should get installed.
"""
import asr.speech_to_text
import skybox.diffusion
import skybox.inpainting


def load_production_pipelines():
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


if __name__ == "__main__":
    load_production_pipelines()
