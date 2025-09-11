from pathlib import Path
import os

from torch import nn
from typing_extensions import Annotated
import numpy as np
import onnx
import onnxruntime as ort
import torch
import typer

from NeuralNetworks.Model_NoCheckerboard import Model_NoCheckerboard

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # for opencv
import cv2 as cv

app = typer.Typer()


def tonemap_reinhard(hdr_tens: torch.Tensor) -> torch.Tensor:
    return (hdr_tens / (1.0 + hdr_tens)).clip(min=1e-4)


def detonemap_reinhard(ldr_tens: torch.Tensor) -> torch.Tensor:
    return (ldr_tens / (1.0 - ldr_tens)).clip(min=1e-4)


def gamma_correction(tensor: torch.Tensor, gamma_coefficent: float = 2.2) -> torch.Tensor:
    return torch.pow(tensor, gamma_coefficent)


def pre_process(
    hdr_tens: torch.Tensor, exposure = 0.5, gamma_coefficent = 2.2
) -> torch.Tensor:
    # 1/exposure -> tonemap -> 1/gamma
    return gamma_correction(tonemap_reinhard(hdr_tens/exposure),  1/gamma_coefficent)


def post_process(
    ldr_tens: torch.Tensor, exposure = 0.5, gamma_coefficent = 2.2
) -> torch.Tensor:
    # gamma -> de-tonemap -> exposure
    return detonemap_reinhard(gamma_correction(ldr_tens.clip(min=1e-4, max=0.999994))) * exposure


def get_model(model_path: Path) -> torch.nn.Module:
    trained_model = Model_NoCheckerboard(in_channels=3, out_channels=3)

    state_dict = torch.load(model_path)
    trained_model.load_state_dict(state_dict["model_state_dict"])
    return trained_model


DEFAULT_LOAD_PATH: str = "../Results/2023-09-22/Trainings/Model_NoCheckerboard/Epochs_600_L1_VGG/model_float32_best.pth"

class SISR_Model(nn.Module):
    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.model = get_model(Path(model_path if model_path else DEFAULT_LOAD_PATH))
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        pre = pre_process(tensor)
        out = self.model(pre)
        return post_process(out)


def _export_to_onnx(model: nn.Module, out_path, input_size=(1, 3, 1080, 1920), channels_last: bool = True,):
    input_tensor = torch.randn(input_size)
    if channels_last:
        model = model.to(memory_format=torch.channels_last) # nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
    onnx_program = torch.onnx.export(model, input_tensor, dynamo=True, input_names=["lr"], output_names=["hr"])
    onnx_program.save(out_path)


@app.command()
def export_to_onnx(
        model_path: Annotated[str, typer.Argument()] = DEFAULT_LOAD_PATH,
        out_path: Annotated[str, typer.Argument] = "sisr.onnx",
        input_size:tuple[int, int, int, int]=(1, 3, 1080, 1920),
        channels_last: bool = False,
    ):
    height, width = input_size[-2:]
    if (width // 8) * 8 != width:
        print(f"Width ({width}) is not a pefect multiple of 8")
    if (height // 8) * 8 != height:
        print(f"Height ({height}) is not a pefect multiple of 8")
    return _export_to_onnx(SISR_Model(model_path), out_path, input_size, channels_last)

@app.command()
def import_onnx(onnx_path: Annotated[str, typer.Argument()] = "sisr.onnx"):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def setup_ort(model):
    priority_list = ['CUDAExecutionProvider', 'MIGraphXExecutionProvider', 'ROCMExecutionProvider', 'CPUExecutionProvider']
    options = ort.SessionOptions()
    options.enable_profiling=False
    ort_session = ort.InferenceSession(model, sess_options=options, providers=priority_list)
    return ort_session


def run_onnx(ort_session, input_tensors: dict[str, torch.Tensor], run_options: ort.RunOptions = ort.RunOptions()):
    output_names = None
    outputs = ort_session.run(output_names, input_tensors)
    return outputs


@app.command()
def upscale(image: str, onnx_path: Annotated[str, typer.Argument()] = "sisr.onnx"):
    input_image = cv.imread(image, cv.IMREAD_COLOR)
    image_tensor = cv.resize(input_image, (1920, 1080), interpolation=cv.INTER_LANCZOS4)
    if image_tensor.dtype != np.float32:
        image_tensor = image_tensor.astype(np.float32) / 255.
    image_tensor = image_tensor.transpose(2, 0, 1)
    # model = import_onnx(onnx_path)
    ort_session = setup_ort(onnx_path)
    out = run_onnx(ort_session, {"lr": image_tensor[np.newaxis, ...]})
    out_image = (out[0] * 255).astype(np.uint8)
    cv_image = out_image[0, ...].transpose(1, 2, 0)
    cv.imwrite("hr.png", cv_image)
    return out[0]
    
    return out

if __name__ == "__main__":
    app()