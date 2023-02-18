
# Own imports
from Config.Config              import Path
from Config.DefaultConfigs      import config
from Dataset.Dataset_UE         import Dataset_UE
from Utils.Utils                import save_model, load_model
from Training_Pipeline          import training_pipeline

################# 
# Training Time #
#################

from datetime import datetime 

start_time = datetime.now() 

trained_model = training_pipeline(config, training=config['run_training'], model_load=config['load_model'])

time_elapsed = datetime.now() - start_time 

print('\nTraining Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


##################
# Inference Time #
##################
from Inference.Inference        import Inference_pipeline

#test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/DATASET/SubwaySequencer_4_26_2/DumpedBuffers"),
#                     csv_root_path=Path("E:/MASTERS/UE4/DATASET/SubwaySequencer_4_26_2/DumpedBuffers/info_Native.csv"))

test_ds = Dataset_UE(ds_root_path=Path("F:/MASTERS/UE4/DATASET/InfiltratorDemo_4_26_2/DumpedBuffers"),
                     csv_root_path=Path("F:/MASTERS/UE4/DATASET/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"))

#loaded_training_state_dict = load_model(Path('E:/MASTERS/Upscaler/Results/2022-12-29/Trainings/Model_Custom/epoch600')/"model_float32_final.pth")
loaded_final_model_state_dict = load_model(config['model_save_path']/"model_float32_final.pth")
trained_model.load_state_dict(loaded_final_model_state_dict['model_state_dict'])
Inference_pipeline(config['model_inference_path']/"model_final", trained_model, test_ds, device=config['device'])

loaded_final_best_state_dict = load_model(config['model_save_path']/"model_float32_best.pth")
trained_model.load_state_dict(loaded_final_best_state_dict['model_state_dict'])
Inference_pipeline(config['model_inference_path']/"model_best", trained_model, test_ds, device=config['device'])


if __name__ == "__main__":
    print("UPSCALER!")