

from typing                     import Dict, Union
from pathlib                    import Path
from Config.Config              import PathType, CurrentDevice
from NeuralNetworks.NN_Base     import Model_Base
from Dataset.Dataset_Base       import Dataset_Base
from Dataset.Dataset_UE         import save_exr
from Colorspace.PreProcessing   import preprocessing_pipeline, depreprocessing_pipeline

import torch

def Inference_pipeline(outputs_save_path:PathType=None, model:Model_Base=None, dataset:Dataset_Base=None, device=CurrentDevice):
    
    assert model is not None and dataset is not None, "Model and Dataset has to be provided as an argument"

    # turn on eval mode of model and turn off requires grad!
    model.eval()
    model.requires_grad_(False)
    model.to(device=device) #move model to specified device
    frame_idx = 20
    with torch.no_grad():

        #load test inputs
        lr, hr = dataset[frame_idx]

        # model is trained with float32 precision for now
        lr, hr = lr.to(dtype=torch.float32), hr.to(dtype=torch.float32)

        #clone lr and hr color, maybe for future purposes
        lr_clone, hr_clone = lr.clone(), hr.clone()

        # preprocess input and GT to have the same colorspace
        lr = preprocessing_pipeline(lr).unsqueeze(0).to(device=device)
        hr = preprocessing_pipeline(hr).unsqueeze(0).to(device=device)

        assert not torch.isnan(lr).any(), "lr contains NaNs"
        assert not torch.isnan(hr).any(), "hr contains NaNs"

        pred_hr = model(lr) # get prediction

        # create specified folder, even if exists (overwrite)
        if not outputs_save_path.exists():
            outputs_save_path.mkdir(exist_ok=True)

        # save buffers
        save_exr(str(outputs_save_path/"GT_hdr.exr"), hr_clone.cpu().half())
        save_exr(str(outputs_save_path/"GT_ldr.exr"), hr.squeeze(0).cpu().half())

        print("Pred min: ", torch.min(pred_hr), "| Pred max: ",torch.max(pred_hr))

        save_exr(str(outputs_save_path/"pred_ldr.exr"), pred_hr.squeeze(0).cpu().half())
        save_exr(str(outputs_save_path/"pred_hdr.exr"), depreprocessing_pipeline(pred_hr).squeeze(0).cpu().half())