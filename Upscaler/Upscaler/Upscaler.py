from Config.Config_Utils import read_cfg
from Config.Config       import *

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
from Config.Config              import CurrentDevice, TrainingsPath, InferenceResultsPath, GetInferencePath
from Dataset.Dataset_UE         import Dataset_UE, save_exr, load_exr_file
from NeuralNetworks.UNet        import Model_UNET
from Colorspace.PreProcessing   import preprocessing_pipeline, depreprocessing_pipeline
from Utils.Utils                import save_model, load_model
from Training_Pipeline          import training_pipeline
from Config.TrainingConfig      import DefaultTrainingDict, BaselineTrainingCfg


################# 
# Training Time #
#################
config = BaselineTrainingCfg
trained_model = training_pipeline(config, training=True, model_load=False)



##################
# Inference Time #
##################
from Inference.Inference import Inference_pipeline

test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers"),
                     csv_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"))

loaded_training_state_dict = load_model(config['model_save_path']/"model_float32_final.pth")
trained_model.load_state_dict(loaded_training_state_dict['model_state_dict'])

Inference_pipeline(config['model_inference_path'], trained_model, test_ds)



if __name__ == "__main__":
    print("UPSCALER!")