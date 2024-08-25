"""
A custom implementation of eigenpunk/ComfyUI-audio, with the main difference of accepting several prompts as an entry.

Source : https://github.com/eigenpunk/ComfyUI-audio/blob/main/musicgen_nodes.py
"""
import ast
import gc
from contextlib import contextmanager
from torch.nn.functional import pad
from typing import Optional, Union

import torch
from audiocraft.models import AudioGen, MusicGen


MODEL_NAMES = [
    "musicgen-small",
    "musicgen-medium",
    "musicgen-melody",
    "musicgen-large",
    "musicgen-melody-large",
    # TODO: stereo models seem not to be working out of the box
    # "musicgen-stereo-small",
    # "musicgen-stereo-medium",
    # "musicgen-stereo-melody",
    # "musicgen-stereo-large",
    # "musicgen-stereo-melody-large",
    "audiogen-medium",
]


def do_cleanup(cuda_cache=True):
    gc.collect()
    if cuda_cache:
        torch.cuda.empty_cache()


def object_to(obj, device=None, exclude=None, empty_cuda_cache=True, verbose=False):
    """
    recurse through an object and move any pytorch tensors/parameters/modules to the given device.
    if device is None, cpu is used by default. if the device is a CUDA device and empty_cuda_cache is
    enabled, this will also free unused CUDA memory cached by pytorch.
    """

    if not hasattr(obj, "__dict__"):
        return obj

    classname = type(obj).__name__
    exclude = exclude or set()
    device = device or "cpu"

    def _move_and_recurse(o, name=""):
        child_moved = False
        for k, v in vars(o).items():
            moved = False
            cur_name = f"{name}.{k}" if name != "" else k
            if cur_name in exclude:
                continue
            if isinstance(v, (torch.nn.Module, torch.nn.Parameter, torch.Tensor)):
                setattr(o, k, v.to(device))
                moved = True
            elif hasattr(v, "__dict__"):
                v, moved = _move_and_recurse(v, name=cur_name)
                if moved: setattr(o, k, v)
            if verbose and moved:
                print(f"moved {classname}.{cur_name} to {device}")
            child_moved |= moved
        return o, child_moved

    if isinstance(obj, torch.nn.Module):
        obj = obj.to(device)

    obj, _ = _move_and_recurse(obj)
    if "cuda" in device and empty_cuda_cache:
        torch.cuda.empty_cache()
    return obj


def tensors_to(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    if hasattr(tensors, "__dict__"):
        return object_to(tensors, device, empty_cuda_cache=False)
    if isinstance(tensors, (list, tuple)):
        return [tensors_to(x, device) for x in tensors]
    if isinstance(tensors, dict):
        return {k: tensors_to(v, device) for k, v in tensors.items()}
    if isinstance(tensors, set):
        return {tensors_to(x, device) for x in tensors}
    return tensors


def tensors_to_cpu(tensors):
    return tensors_to(tensors, "cpu")


@contextmanager
def obj_on_device(model, src="cpu", dst="cuda", exclude=None, empty_cuda_cache=True, verbose_move=False):
    model = object_to(model, dst, exclude=exclude, empty_cuda_cache=empty_cuda_cache, verbose=verbose_move)
    yield model
    model = object_to(model, src, exclude=exclude, empty_cuda_cache=empty_cuda_cache, verbose=verbose_move)


def stack_audio_tensors(tensors, mode="pad"):
    sizes = [x.shape[-1] for x in tensors]

    if mode in {"pad_l", "pad_r", "pad"}:
        # pad input tensors to be equal length
        dst_size = max(sizes)
        stack_tensors = (
            [pad(x, pad=(0, dst_size - x.shape[-1])) for x in tensors]
            if mode == "pad_r"
            else [pad(x, pad=(dst_size - x.shape[-1], 0)) for x in tensors]
        )
    elif mode in {"trunc_l", "trunc_r", "trunc"}:
        # truncate input tensors to be equal length
        dst_size = min(sizes)
        stack_tensors = (
            [x[:, x.shape[-1] - dst_size:] for x in tensors]
            if mode == "trunc_r"
            else [x[:, :dst_size] for x in tensors]
        )
    else:
        assert False, 'unknown mode "{pad}"'

    return torch.stack(stack_tensors)


class MusicgenGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MUSICGEN_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 300.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "top_k": ("INT", {"default": 250, "min": 0, "max": 10000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {"audio": ("AUDIO_TENSOR",)},
        }

    RETURN_NAMES = ("RAW_AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "generate"
    CATEGORY = "audio"

    def generate(
        self,
        model: Union[AudioGen, MusicGen],
        text: str = "",
        batch_size: int = 1,
        duration: float = 10.0,
        cfg: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        seed: int = 0,
        audio: Optional[torch.Tensor] = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # empty string = unconditional generation
        if text == "":
            text = None

        model.set_generation_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            duration=duration,
            cfg_coef=cfg,
        )
        with torch.random.fork_rng(), obj_on_device(model, dst=device, verbose_move=True) as m:
            torch.manual_seed(seed)
            text_input = ast.literal_eval(text)
            print(text_input)
            if audio is not None:
                # do continuation with input audio and (optional) text prompting
                if isinstance(audio, list):
                    # left-padded stacking into batch tensor
                    audio = stack_audio_tensors(audio)

                if audio.shape[0] < batch_size:
                    # (try to) expand batch if smaller than requested
                    audio = audio.expand(batch_size, -1, -1)
                elif audio.shape[0] > batch_size:
                    # truncate batch if larger than requested
                    audio = audio[:batch_size]

                audio_input = tensors_to(audio, device)
                audio_out = m.generate_continuation(audio_input, model.sample_rate, text_input, progress=True)
            elif text is not None:
                # do text-to-music
                audio_out = m.generate(text_input, progress=True)
            else:
                # do unconditional music generation
                audio_out = m.generate_unconditional(batch_size, progress=True)

            audio_out = tensors_to_cpu(audio_out)

        audio_out = torch.unbind(audio_out)
        do_cleanup()
        return list(audio_out),


NODE_CLASS_MAPPINGS = {
    "MusicgenGenerateCustom": MusicgenGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicgenGenerateCustom": "Musicgen Generator Custom",
}