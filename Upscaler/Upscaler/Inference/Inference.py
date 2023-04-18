from typing import Dict, Union
from pathlib import Path
from Config.Config import PathType, CurrentDevice
from NeuralNetworks.NN_Base import Model_Base
from Dataset.Dataset_Base import Dataset_Base
from Dataset.Dataset_UE import save_exr
from Colorspace.PreProcessing import preprocessing_pipeline, depreprocessing_pipeline
from torchmetrics import StructuralSimilarityIndexMeasure

import torch


def Inference_pipeline(
    outputs_save_path: PathType = None,
    model: Model_Base = None,
    dataset: Dataset_Base = None,
    device=CurrentDevice,
):

    assert (
        model is not None and dataset is not None
    ), "Model and Dataset has to be provided as an argument"

    # turn on eval mode of model and turn off requires grad!
    model.eval()
    model.requires_grad_(False)
    model.to(device=device)  # move model to specified device
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    # load test inputs
    frame_idx = 20
    lr, hr = dataset[frame_idx]

    # model is trained with float32 precision for now
    lr, hr = lr.to(dtype=torch.float32), hr.to(dtype=torch.float32)

    # clone lr and hr color, maybe for future purposes
    lr_clone, hr_clone = lr.clone(), hr.clone()

    # preprocess input and GT to have the same colorspace
    lr = preprocessing_pipeline(lr).unsqueeze(0).to(device=device)
    hr = preprocessing_pipeline(hr).unsqueeze(0).to(device=device)

    assert not torch.isinf(lr).any(), "lr contains NaNs/Infs"
    assert not torch.isinf(hr).any(), "hr contains NaNs/Infs"

    # create specified folder, even if exists (overwrite)
    if not outputs_save_path.exists():
        outputs_save_path.mkdir(exist_ok=False)

    with torch.no_grad():

        # Model
        pred_hr = model(lr)  # get prediction
        pred_ldr = pred_hr.clone()
        pred_hdr = depreprocessing_pipeline(pred_hr)

        assert not torch.isinf(pred_ldr).any(), "pred_ldr contains NaNs/Infs"
        assert not torch.isinf(pred_hdr).any(), "pred_hdr contains NaNs/Infs"

        # save buffers
        save_exr(str(outputs_save_path / "hdr_GT.exr"), hr_clone.cpu().half())
        save_exr(str(outputs_save_path / "ldr_GT.exr"), hr.squeeze(0).cpu().half())
        save_exr(
            str(outputs_save_path / "ldr_pred.exr"), pred_ldr.squeeze(0).cpu().half()
        )
        save_exr(
            str(outputs_save_path / "hdr_pred.exr"), pred_hdr.squeeze(0).cpu().half()
        )

        # Print info about similaritiness and range of prediction, target
        print(outputs_save_path.stem)
        print("Pred min: ", torch.min(pred_ldr), "| Pred max: ", torch.max(pred_ldr))
        print("Target min: ", torch.min(hr), "| Target max: ", torch.max(hr))

        print(
            "SSIM between pred hdr and GT hdr: ",
            ssim(pred_hdr.cpu(), hr_clone.unsqueeze(0).cpu()),
        )
        print("SSIM between pred ldr and GT ldr: ", ssim(pred_ldr.cpu(), hr.cpu()))

    # Naive upscaling
    upscale_nearest = torch.nn.Upsample(scale_factor=2, mode="nearest")
    pred_hr_nearest = upscale_nearest(lr)
    pred_nearest_ldr = pred_hr_nearest.clone()
    pred_nearest_hdr = depreprocessing_pipeline(pred_hr_nearest)

    # save buffers
    save_exr(str(outputs_save_path / "hdr_GT.exr"), hr_clone.cpu().half())
    save_exr(str(outputs_save_path / "ldr_GT.exr"), hr.squeeze(0).cpu().half())
    save_exr(
        str(outputs_save_path / "ldr_pred_naive.exr"),
        pred_nearest_ldr.squeeze(0).cpu().half(),
    )
    save_exr(
        str(outputs_save_path / "hdr_pred_naive.exr"),
        pred_nearest_hdr.squeeze(0).cpu().half(),
    )

    # Print info about similaritiness and range of prediction, target
    print(
        "Pred naive min: ",
        torch.min(pred_nearest_ldr),
        "| Pred naive max: ",
        torch.max(pred_nearest_ldr),
    )
    print("Target min: ", torch.min(hr), "| Target max: ", torch.max(hr))
    print(
        "SSIM between pred naive hdr and GT hdr: ",
        ssim(pred_nearest_hdr.cpu(), hr_clone.unsqueeze(0).cpu()),
    )
    print(
        "SSIM between pred naive ldr and GT ldr: ",
        ssim(pred_nearest_ldr.cpu(), hr.cpu()),
    )
