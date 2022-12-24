from Config.Config_Utils import read_cfg
from Config.Config import *

# pth = Path("Config/config_yamls/config.yaml")
# print(read_cfg(pth))


#from NeuralNetworks.UNet import test

## test()

#from Dataset.Dataset_UE import test_ds_ue

#test_ds_ue()

#####################################################################
#####################################################################
#####################################################################
#####################################################################

# Own imports
from Config.Config import CurrentDevice, TrainingsPath, InferenceResultsPath
from Dataset.Dataset_UE import Dataset_UE, save_exr, load_exr_file
from NeuralNetworks.UNet import Model_UNET
from Colorspace.PreProcessing import preprocessing_pipeline, depreprocessing_pipeline
from Utils.Utils import save_model, load_model

# Training time
from Training_Pipeline import training_pipeline

trained_model = training_pipeline(training=True, model_load=True)



# Inference time
from Inference.Inference import Inference_pipeline
from datetime import date

test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers"),
        csv_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"),)



loaded_training_state_dict = load_model(Path(TrainingsPath/"model_float32_final.pth"))
trained_model.load_state_dict(loaded_training_state_dict['model_state_dict'])

Inference_pipeline(InferenceResultsPath, trained_model, test_ds)



if __name__ == "__main__":
    print("UPSCALER!")