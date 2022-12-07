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
from Config.Config import CurrentDevice
from Dataset.Dataset_UE import Dataset_UE, save_exr, load_exr_file
from NeuralNetworks.UNet import Model_UNET
from Colorspace.PreProcessing import preprocessing_pipeline, depreprocessing_pipeline
from Utils.Utils import save_model, load_model

# Libs imports
import torch.nn as nn
import torchvision.transforms.functional as tvf
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar when training the data!
from datetime import date

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True




# Hyperparameters
in_channels = 3
out_channels = 3
learning_rate = 0.001
batch_size = 32
num_epochs = 60


# Load Data
train_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
        csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),
        #crop width x height == 128x128 (for now)
        crop_coords=(900, 1028, 500, 628))
        #crop_coords=(900, 964, 500, 564))

# Create dataloader
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, drop_last=True, pin_memory=True)


# Initialize network
model = Model_UNET(in_channels=3, out_channels=3).to(device=CurrentDevice, dtype=torch.float32)

## Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


print("INFO: Used device: ", CurrentDevice)

# Train Network
avg_loss_per_epoch = []
model.train() # prep model for training
for epoch in range(num_epochs):

    #Log pass
    print('Epoch: %03d' % (epoch + 1), end="\n")
    avg_train_loss = 0.0

    if epoch % 5 == 0 or epoch == 1:
        # TODO make better saving pipeline
        model_save_path = Path("E:/MASTERS/Upscaler/Models/{}/model_float32_epoch{}.pth".format(date.today(), epoch))
        save_model(model_save_path, 
            {
            'epoch': epoch,
            'batch_size': batch_size,
            'lr': learning_rate,
            'Dataset': 'Dataset_UE',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion.state_dict(),
            })

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

        #Zero gradients
        optimizer.zero_grad()

        # Get data to cuda if possible
        data = data.to(device=CurrentDevice, dtype=torch.float32)
        target = target.to(device=CurrentDevice, dtype=torch.float32)

        # PreProcess the data
        data = preprocessing_pipeline(data)
        target = preprocessing_pipeline(target)

        # forward
        pred = model(data)
        loss = criterion(pred, target)
        # accumulate loss, loss * amount N batch size
        avg_train_loss += loss.item() * data.size(0)

        # loss backward and optimizer
        loss.backward()
        optimizer.step()

    # divide avg train loss by length of data loader sampler
    # it will give a correct avg loss
    # if divided by batch_size, then sometimes it may be not correct,
    # because batch_size is sometimes not dividable by num of samples
    avg_train_loss = avg_train_loss / len(train_loader.sampler)
    avg_loss_per_epoch.append(avg_train_loss)

    #Log pass
    print(' Avg loss: %.3f' % avg_train_loss, end="\n")


import matplotlib.pyplot as plt
# summarize history for loss
fig, axs = plt.subplots(figsize = (20,6))
plt.plot(avg_loss_per_epoch)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model_save_path = Path("E:/MASTERS/Upscaler/Models/{}/model_float32_final.pth".format(date.today()))
save_model(model_save_path, 
{
'epoch': epoch,
'batch_size': batch_size,
'lr': learning_rate,
'Dataset': 'Dataset_UE',
'model_state_dict': model.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
'loss': criterion.state_dict(),
})




# Inference time
from Inference.Inference import Inference_pipeline
#test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
#        csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),)
test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers"),
        csv_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"),)



loaded_training_state_dict = load_model(Path("E:/MASTERS/Upscaler/Models/{}/model_float32_final.pth".format(date.today())))
model.load_state_dict(loaded_training_state_dict['model_state_dict'])

Inference_pipeline(Path("E:/MASTERS/Upscaler/Results/{}".format(date.today())), model, test_ds)























##model.load_state_dict(torch.load(Path("E:/MASTERS/Upscaler/Models/06-12-2022/model_float32_epoch29.pth"))['model_state_dict'])

#model.eval()
#import matplotlib.pyplot as plt
#with torch.no_grad():
#    # Plotting part
#    figure = plt.figure(figsize=(20, 20))
#    lr, hr = test_ds[500]
#    lr = lr.to(dtype=torch.float32)
#    hr = hr.to(dtype=torch.float32)
#    clone_lr = lr.clone()
#    #lr = postprocessing_pipeline(lr)
#    lr = lr.unsqueeze(0).to(device=CurrentDevice)


#    print(torch.isnan(clone_lr).any())
#    print(torch.isnan(lr).any())
#    #pred = model(torch.nan_to_num(tens))
#    pred = model(lr)
#    #pred = lr
#    #plt.imshow(pred.squeeze(0).permute(1,2,0).to(dtype=torch.float32).cpu().detach().numpy())
#    #plt.show()

#    save_exr("E:/MASTERS/Upscaler/Results/Temp/lr_hdr.exr", clone_lr.cpu().half(),
#    channels=["R", "G", "B"])

#    save_exr("E:/MASTERS/Upscaler/Results/Temp/lr_ldr.exr", lr.cpu().squeeze(0).half(),
#    channels=["R", "G", "B"])
#    save_exr("E:/MASTERS/Upscaler/Results/Temp/pred_ldr.exr",
#    pred.cpu().squeeze(0).half(), channels=["R","G","B"])

#    save_exr("E:/MASTERS/Upscaler/Results/Temp/pred_hdr.exr",
#    depostprocessing_pipeline(pred).cpu().squeeze(0).half(), channels=["R", "G", "B"])




if __name__ == "__main__":
    print("UPSCALER!")
