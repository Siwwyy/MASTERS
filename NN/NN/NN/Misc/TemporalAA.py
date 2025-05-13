import math
from typing import Union
import torch
import torch.nn.functional as F

import cupy as cp

from pathlib import Path
from NN.Config.BaseTypes import CurrentDevice, TensorType
from NN.Dataloader.DataloaderUtils import (
    loadUnrealJitterXYCSV,
    loadEXR,
    loadUnrealCSV,
    readColorBuffer,
    readDepthBuffer,
    readVelocityBuffer,
    saveEXR,
)
from NN.ImageProcessing.ImageTransform import getNDCGrid, reproject


def DIV_UP(nominator: int, denominator: int) -> int | float:
    return math.floor((nominator + denominator - 1) / denominator)


def TAAReprojectionResolve(
    previousFrame: TensorType, currentVelocity: TensorType, currentDepth: TensorType
) -> TensorType:
    reprojectedFrame = reproject(previousFrame, currentVelocity, currentDepth)
    return reprojectedFrame


def TAAColorClampingPytorch(
    currentColorBuffer: TensorType, historyColorBuffer: TensorType
) -> TensorType:
    # Tensor padding (for AABB a.k.a sliding window)
    p3d = (1, 1, 1, 1)  # respectively padding is added to: left, right, top bottom
    padded_input_tensor = F.pad(
        input=currentColorBuffer.float(), pad=p3d, mode="replicate"
    )

    # Max pool for getting sliding window 3x3
    min_val = -torch.nn.functional.max_pool2d(
        -padded_input_tensor, kernel_size=(3, 3), stride=(1, 1)
    )
    max_val = torch.nn.functional.max_pool2d(
        padded_input_tensor, kernel_size=(3, 3), stride=(1, 1)
    )

    # Clamping tensor in NN Clamping manner
    return torch.clamp(historyColorBuffer, min=min_val, max=max_val)


def TAAColorClampingCupy(
    currentColorBuffer: TensorType, historyColorBuffer: TensorType
) -> TensorType:
    # Tensor padding (for AABB a.k.a sliding window)
    p3d = (1, 1, 1, 1)  # respectively padding is added to: left, right, top bottom
    padded_input_tensor = F.pad(
        input=currentColorBuffer.float(), pad=p3d, mode="replicate"
    )

    NNClampingKernel = cp.RawKernel(
        r"""

    inline float fminf(float a, float b)
    {
        return a < b ? a : b;
    }

    inline __host__ __device__ float3 fminf(float3 a, float3 b)
    {
        return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
    }

    inline float fmaxf(float a, float b)
    {
        return a > b ? a : b;
    }

    inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
    {
        return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
    }

    inline __device__ __host__ float clamp(float f, float a, float b)
    {
        return fmaxf(a, fminf(f, b));
    }

    inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
    {
        return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
    }

    extern "C" __global__
    void NNClampingKernel(const float* x1, 
                          float* y1, 
                          size_t size_x, 
                          size_t size_y) 
    {
        const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
        if(tid_x < size_x && tid_y < size_y)
        {
            const int index = tid_y * size_x + tid_x; //times size_x, because we have to "move/offset" to current row by row amount
            const int channelR_offset = (size_x * size_y * 0); //R channel is at 0th position
            const int channelG_offset = (size_x * size_y * 1); //G channel is at 1st position
            const int channelB_offset = (size_x * size_y * 2); //B channel is at 2nd position

            // Arbitrary out of range numbers
            float3 minColorRGB = make_float3(999999.0f, 999999.0f, 999999.0f);
            float3 maxColorRGB = make_float3(-999999.0f, -999999.0f, -999999.0f);
         
            // Sample a 3x3 neighborhood to create a box in color space
            for(int idy = -1; idy <= 1; ++idy)
            {
                for(int idx = -1; idx <= -1; ++idx)
                {
                    // Calculate indices
                    const int aabb_offset_x = 1 + idx;
                    const int aabb_offset_y = 1 + idy;
                    const int padded_size_x = (size_x + 2); // because padding left(1) and right(1) of row
                    const int padded_size_y = (size_y + 2); // because padding top(1) and bottom(1) of column
                    const int padded_channelR_offset = padded_size_y * (padded_size_x * 0); //R channel is at 0th position
                    const int padded_channelG_offset = padded_size_y * (padded_size_x * 1); //G channel is at 1st position
                    const int padded_channelB_offset = padded_size_y * (padded_size_x * 2); //B channel is at 2nd position
                    const int aabb_indicesR = ((tid_y + aabb_offset_y) * padded_size_x) + (tid_x + aabb_offset_x) + padded_channelR_offset;
                    const int aabb_indicesG = ((tid_y + aabb_offset_y) * padded_size_x) + (tid_x + aabb_offset_x) + padded_channelG_offset;
                    const int aabb_indicesB = ((tid_y + aabb_offset_y) * padded_size_x) + (tid_x + aabb_offset_x) + padded_channelB_offset;
                    // Sample neighbor      
                    float3 color = make_float3(0.0f, 0.0f, 0.0f);
                    color.x = x1[aabb_indicesR];
                    color.y = x1[aabb_indicesG];
                    color.z = x1[aabb_indicesB];              
                    // Take min and max
                    minColorRGB = fminf(minColorRGB, color); 
                    maxColorRGB = fmaxf(maxColorRGB, color);
                }
            }

            // Clamp previous color to min/max bounding box
            y1[index + channelR_offset] = clamp(y1[index + channelR_offset], minColorRGB.x, maxColorRGB.x);
            y1[index + channelG_offset] = clamp(y1[index + channelG_offset], minColorRGB.y, maxColorRGB.y);
            y1[index + channelB_offset] = clamp(y1[index + channelB_offset], minColorRGB.z, maxColorRGB.z);
        }
    }
    """,
        "NNClampingKernel",
    )
    # prepare inputs to cupy kernel (cuda)
    cp_padded_input_tensor = cp.asarray(padded_input_tensor)
    cp_output_tensor = cp.asarray(historyColorBuffer.float())

    # prepare dispatch/invocation
    THREADS_PER_BLOCK = 32
    BLOCKS_NUM_X = DIV_UP(currentColorBuffer.shape[-1], THREADS_PER_BLOCK)
    BLOCKS_NUM_Y = DIV_UP(currentColorBuffer.shape[-2], THREADS_PER_BLOCK)

    # dispatch/invocation
    print(
        "Dispatch Blocks(%d, %d), Threads(%d)"
        % (BLOCKS_NUM_X, BLOCKS_NUM_Y, THREADS_PER_BLOCK)
    )
    NNClampingKernel(
        (
            BLOCKS_NUM_X,
            BLOCKS_NUM_Y,
        ),  # blocks per kernel
        (THREADS_PER_BLOCK, THREADS_PER_BLOCK),  # threads per block
        (
            cp_padded_input_tensor,  # x1
            cp_output_tensor,  # y1
            cp_output_tensor.shape[-1],  # size_x
            cp_output_tensor.shape[-2],
        ),
    )  # size_y

    return torch.from_numpy(cp.asnumpy(cp_output_tensor)).to(currentColorBuffer)


if __name__ == "__main__":

    print("TemporalAA")
    DataPath = Path(
        r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\1920x1080-native"
    )

    HistoryColorBuffer = torch.zeros(1, 3, 1080, 1920, dtype=torch.float32).to(
        device=CurrentDevice
    )
    for idx in range(120):
        print("Frame {}".format(idx))
        CurrentColorBuffer = readColorBuffer(
            str(DataPath / f"SceneColorTexture/{idx}.exr")
        ).unsqueeze(0)
        CurrentVelocityBuffer = readVelocityBuffer(
            str(DataPath / f"SceneVelocityTexture/{idx}.exr")
        ).unsqueeze(0)
        CurrentDepthBuffer = readDepthBuffer(
            str(DataPath / f"SceneDepthTexture/{idx}.exr")
        ).unsqueeze(0)

        # 1. Resolve pass

        # Get grid
        NDC_GRID_X, NDC_GRID_Y = getNDCGrid(1920, 1080)
        NDC_GRID_XY = torch.stack([NDC_GRID_X, NDC_GRID_Y], dim=-1).unsqueeze(
            0
        )  # get batch dim with unsqueeze

        # Invert Y axis, because of other convention used in Pytorch i.e., top-left is -1,-1, whereas
        # in DX12 -1,-1 its bottom-left
        CurrentVelocityBuffer[..., 1:2] = CurrentVelocityBuffer[..., 1:2] * -1.0

        # Permute MV (Velocity) buffer channels to NHWC (otherwise, we are unable to subtract by grid simply)
        nhwc_mv = CurrentVelocityBuffer.permute(0, 2, 3, 1)
        # "Projects" previous grid values to current by addition
        NEW_NDC_COORDS = NDC_GRID_XY - nhwc_mv

        HistoryColorBuffer = F.grid_sample(
            HistoryColorBuffer,
            NEW_NDC_COORDS,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # HistoryColorBuffer = TAAReprojectionResolve(HistoryColorBuffer, CurrentVelocityBuffer, CurrentDepthBuffer)

        # 2. Color Clamping pass
        # HistoryColorBuffer = TAAColorClampingPytorch(CurrentColorBuffer, HistoryColorBuffer)
        HistoryColorBuffer = TAAColorClampingCupy(
            CurrentColorBuffer, HistoryColorBuffer
        )

        # 3. History update with resolve output and dejitter Current color
        csvPath = Path(
            r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\info_Native.csv"
        )
        jitter_XY = loadUnrealJitterXYCSV(
            csvPath, startsWithFilter="Jitter_", frameIdx=idx
        )

        # Current Color Dejittering
        dejittered_NDC_GRID_XY = NDC_GRID_XY.clone()
        dejittered_NDC_GRID_XY[:, :, :, 0] = (
            dejittered_NDC_GRID_XY[:, :, :, 0] - jitter_XY[0]
        )
        dejittered_NDC_GRID_XY[:, :, :, 1] = (
            dejittered_NDC_GRID_XY[:, :, :, 1] - jitter_XY[1]
        )
        dejitteredColorBuffer = F.grid_sample(
            CurrentColorBuffer.float(),
            NDC_GRID_XY,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Accumulate samples to history
        HistoryColorBuffer = dejitteredColorBuffer * 0.1 + HistoryColorBuffer * 0.9

        saveEXR(
            str(DataPath / f"SceneColorTextureAfterAA/{idx}.exr"),
            HistoryColorBuffer.squeeze(0),
        )
