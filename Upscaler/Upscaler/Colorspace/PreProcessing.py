
from typing             import Dict, Union
from pathlib            import Path
from Config.Config      import PathType, TensorType

import torch

Exposure_Coefficient = 0.5
Gamma_Coefficient = 2.2

# Pre Processing input to the network stage

# Tonemapper HDR -> LDR
def tonemap_reinhard(hdr_tens:TensorType) -> TensorType:
    hdr_tens[:] = hdr_tens / (1. + hdr_tens)
    return hdr_tens.clip(min=1e-4)

# Detonemapper LDR -> HDR
def detonemap_reinhard(ldr_tens:TensorType) -> TensorType:
    ldr_tens[:] = ldr_tens / (1. - ldr_tens)
    return ldr_tens.clip(min=1e-4)

# Gamma curve adjustments to get sRGB colorspace
def gamma_correction(ldr_tens:TensorType, gamma_coefficent:Union[TensorType, float]=Gamma_Coefficient) -> TensorType:
    ldr_tens[:] = torch.pow(ldr_tens, gamma_coefficent)
    return ldr_tens


# PreProcessing of inputs, 1/Exposure -> Tonemap -> Inverse Gamma
def preprocessing_pipeline(hdr_tens:TensorType, exposure:Union[TensorType, float]=Exposure_Coefficient) -> TensorType:
    # 1./Exposure -> tonemap -> inverse gamma correction
    hdr_tens *= (1. / exposure)
    hdr_tens[:] = tonemap_reinhard(hdr_tens)
    return gamma_correction(hdr_tens, 1.0 / Gamma_Coefficient) #inverse gamma correction


# Depreprocessing of inputs, Gamma -> Detonemap -> Exposure multiplication
def depreprocessing_pipeline(ldr_tens:TensorType, exposure:Union[TensorType, float]=Exposure_Coefficient) -> TensorType:
    # gamma correction -> detonemap -> Exposure
    ldr_tens[:] = gamma_correction(ldr_tens.clip(min=1e-4, max=0.99999994))
    ldr_tens[:] = detonemap_reinhard(ldr_tens)
    ldr_tens *= exposure
    return ldr_tens