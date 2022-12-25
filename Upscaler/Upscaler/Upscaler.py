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
from Config.Config              import CurrentDevice, TrainingsPath, InferenceResultsPath, GetInferenceResultsPath
from Dataset.Dataset_UE         import Dataset_UE, save_exr, load_exr_file
from NeuralNetworks.UNet        import Model_UNET
from Colorspace.PreProcessing   import preprocessing_pipeline, depreprocessing_pipeline
from Utils.Utils                import save_model, load_model
from Training_Pipeline          import new_training_pipeline
from Config.TrainingConfig      import DefaultTrainingDict, TrainingConfig


# Training time
#trained_model = training_pipeline(training=True, model_load=True)
trainingConfig = TrainingConfig()
trained_model = new_training_pipeline(trainingConfig, 
                                      training=False, 
                                      model_load=False)



# Inference time
from Inference.Inference import Inference_pipeline
from datetime            import date, time, datetime

test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers"),
        csv_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"),)



loaded_training_state_dict = load_model(Path(TrainingsPath/"custom/model_float32_final.pth"))
trained_model.load_state_dict(loaded_training_state_dict['model_state_dict'])

time_now = datetime.now().strftime("%H_%M_%S")
#stem_pth = Path(f"baseline_{time_now}")
stem_pth = Path(f"baseline_custom")
inference_results_path = GetInferenceResultsPath(stem=stem_pth)
Inference_pipeline(inference_results_path, trained_model, test_ds)



if __name__ == "__main__":
    print("UPSCALER!")