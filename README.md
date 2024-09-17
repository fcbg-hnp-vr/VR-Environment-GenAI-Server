# VR Environment Creation with Generative AI, Python Server

A Python project to create VR environments using Generative AI.
You can run it as a TCP server to interface it with a [Unity client](https://github.com/fcbg-hnp-vr/VR-Environment-GenAI-Unity),
to get the fully-fledged AI/VR application.

> This is a public archive, development continues at [HugoFara/speech-to-world-server](https://github.com/HugoFara/speech-to-world-server)!

This is a use case of generative AI to build a complete VR scenery.
It was developed at the [Fondation Campus Biotech Geneva](https://fcbg.ch/),
in collaboration with the [Laboratory of Cognitive Science](https://www.epfl.ch/labs/lnco/),
by Hugo FARAJALLAH.

## Requirements

- Python 3.10.12+
- A CUDA-compatible graphic card and at least 12 GB of VRAM.
- Up to 15 GB of storage for the models.

## Installation

You need to get Python 3.10 and CUDA 12.1 (other versions are untested).
Once the requirements are installed, the project should work.

Here is a detailed installation procedure:

1. Install [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive), it enables computation on the GPU.
2. Install **Python 3.10**, for Windows you can download it using the [official installer](https://www.python.org/downloads/windows/). 
3. Clone or copy this Git repository: <https://github.com/fcbg-hnp-vr/VR-Environment-GenAI-Server/>.
4. Create a Python virtual environment. While not strictly necessary, it is highly recommended as the project has 
many dependencies.
For instance using [venv](https://docs.python.org/3/library/venv.html):
   * On Linux:
   ```bash
   cd VR-Environment-GenAI-Server
   # From https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments
   python -m venv .venv       # Creates the virtual environment under .venv
   source .venv/bin/activate  # Activates it
   ```
   * On Windows:
   ```shell
   cd VR-Environment-GenAI-Server
   # From https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments
   py -m venv .venv        # Creates the virtual environment under .venv
   .venv\Scripts\activate  # Activates it
   ```
6. Install the Python requirements.
   ```bash
   pip install -r requirements.txt
   ```
   **Important**: at the time of writing (2024-07-29) the default version of PyTorch
   is compatible with CUDA 12.1, and you may not need any extra steps. If you receive
   error message telling you that your version of PyTorch is not compatible with CUDA,
   uninstall PyTorch completely and reinstall it by running 
   ``pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121``.
   Please have a look at <https://pytorch.org/get-started/locally/> for details.

From here on, the project should be functional. The next section is optional, but it can save you a lot of time.

> (optional) You can speed up image generation using [accelerate](https://huggingface.co/docs/accelerate/index). 
Download it with ``pip install accelerate``.


## Installation details

* The first time a model is launched is needs to be downloaded, 
this operation can take some time, and you need an internet connection. 
The [Usage](#usage) section explains how to download all models at once.
* For users of PyCharm, an `.idea` folder is included to add the folder as a project.
* Optional, demo only: to capture the audio from the microphone in Python (ASR), 
you need ffmpeg, portaudio and pyaudio:
  ```bash
  sudo apt install ffmpeg portaudio19-dev python3-pyaudio
  pip install -r requirements-optional.txt # Installs PyAudio
  ```

## Usage

Each file can be executed independently, so they are as many entry points as files.

The most common use cases are the following:

* Generate a new image with ``python -m skybox.diffusion``.
* Download all models with ``python -m utils.download_models``. 
If you don't do it the models will be downloaded at run time which may be very slow.
* Start the server with ``python -m server.run``.

Next is the detail for special files.

### Image generation

Go to the ``skybox`` folder.

1. diffusion.py - base module to create an image from a diffusion model.
2. inpainting.py - implements an inpainting model.
3. image_processing.py - defines image processing features
4. mask_editor.py - code logics to generate a mask adapted to the image. 
The result is usually passed to inpainting functions.
5. panorama_creator.py - code logics to generate a panorama.
6. The code in ``skybox/legacy`` may not be useful. I keep it there for personal intents.

### 3D features

3D features are in the ``environment`` folder. It is still in active development at the time of writing (June 2024), 
hence the following is subject to change.

1. depth_generation.py - provides a model to come from a standard RGB image and create a depth map.
2. point_cloud_pipeline.py - uses the RGBD to create a point cloud, and converts it to a mesh.
3. mesh_pipeline.py - uses the RGBD image and representation features to create a terrain mesh. 
4. mask_former.py - semantic segmentation of an RGB image. 
5. image_segmentation.py - uses an RGBD+semantic image to isolate the main elements.
6. depth_inpainting.py - combines inpainting controlled by depth data to recreate parts of a terrain.
Yet not integrated in the main code base.
7. rendered.py - create a 3D view for the terrain, not finished yet.

### Speech-to-text (ASR)

For speech to text features, go to ``asr`` (automatic speech recognition) 

* speech_to_text.py - implements an Automatic Speech Recognition (ASR) model.
* asr_demo.py - simply a demo, you can either use your microphone or load the dataset

### ComfyUI graphical interface

If you want to use a graphical interface instead of Python code,
you can use the provided [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows
in the `ComfyUI` folder.

The explanation for each workflow is detailed in [ComfyUI/README.md](ComfyUI/README.md).

### Server

The server features are in `server`. See [Start as a TCP server](#start-as-a-tcp-server) for the details on usage.

* run.py - starts a TCP server, able to serve requests to the previously defined models.
* task_tracker.py - Just a class adding syntactic suger to track a task easily
* utils.py - Utility functions for the server.


### Other Features

* As a test, the ``sound`` folder has some experiments with sound generation.
* The ``utils`` folder contains useful functions for the user:
  * download_models.py - downloads useful models for the server. It does not download all models.

## Configuration

The main server configuration is in ``api.json``.
The most significant configuration data are "serverIp" and "serverPort" as they set the address of the server.

## Start as a TCP server

A TCP server can be started in order to offload the AI part from the application thread. 
Just launch `python -m server.run`. The server [configuration](#configuration) is defined in `api.json`.
The communication is handled in JSON format, with a strong HTTP style.

To connect to the server from another computer on the same network, you need to open a port. 
On Windows, you simply need to go to the control panel add a new rule for the port `9000` (with the default configuration).
This [How-To Geek tutorial](https://www.howtogeek.com/394735/how-do-i-open-a-port-on-windows-firewall/) seems guiding enough.
On Linux, opening ports is a bit more fun, I personally recommend using nginx with a port redirection.

## Roadmap

Current status of the project, from a very far perspective.

- [x] Skybox generation : v0.4 done, go to ``skybox/panorama_creator.py``
- [ ] Terrain generation : Early 3D terrain generation in ``environment/renderer.py`` not suitable for production now. 
- [ ] Props generation : use billboards only as current technology do not allow to dream bigger.

## Models' list

This project includes several artificial neural network models. 
If you want to substitute a model by another one, you should have a good knowledge of what you are doing,
otherwise the quality of the end product may be decreased.

- Image creation : [Stable Diffusion XL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and 
[Stable Diffusion XL refiner 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0).
- Inpainting and outpainting : [Stable Diffusion XL 1.0 Inpainting 0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1).
- Speech-to-text and translation : [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3).

Please have a look at ``utils/download_models.py`` to see where those models are loaded from.

## Useful Links

You can download the official Unity client from [VR-Environment-GenAI-Unity (GitHub)](https://github.com/fcbg-hnp-vr/VR-Environment-GenAI-Unity).
If you are looking for the active public repository of this project, go to [HugoFara/speech-to-world-server](https://github.com/HugoFara/speech-to-world-server).
