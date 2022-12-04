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
from Config.Config import CurrentDevice
from Dataset.Dataset_UE import Dataset_UE, save_exr, load_exr_file
from NeuralNetworks.UNet import Model_UNET

import torch.nn as nn
import torchvision.transforms.functional as tvf
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar when training the data!


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True






# Hyperparameters
in_channels = 3
out_channels = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 20


# Load Data
train_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
        csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),
        #crop width x height == 128x128 (for now)
        crop_coords=(900, 1028, 500, 628))
        #crop_coords=(900, 964, 500, 564))

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, drop_last=True, pin_memory=True)


# Initialize network
model = Model_UNET(in_channels=3, out_channels=3).to(device=CurrentDevice, dtype=torch.float32)

## Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Post Processing stage
def tonemap_reinhard(hdr_tens:TensorType) -> TensorType:
    return hdr_tens / (1. + hdr_tens)


def detonemap_reinhard(ldr_tens:TensorType) -> TensorType:
    return ldr_tens / (1. - ldr_tens).clip(min=1e-4)


def gamma_correction(ldr_tens:TensorType, gamma_coefficent:Union[TensorType, float]=2.2) -> TensorType:
    #assert torch.logical_and(ldr_tens.min() > 0., ldr_tens.max() < 1.  +
    #1e-05)
    return torch.pow(ldr_tens, gamma_coefficent)


def postprocessing_pipeline(hdr_tens:TensorType, exposure:Union[TensorType, float]=0.5) -> TensorType:
    # 1./Exposure -> tonemap -> inverse gamma correction
    hdr_tens *= (1. / exposure)
    hdr_tens = tonemap_reinhard(hdr_tens)
    return gamma_correction(hdr_tens, 1.0 / 2.2) #inverse gamma correction


def depostprocessing_pipeline(ldr_tens:TensorType, exposure:Union[TensorType, float]=0.5) -> TensorType:
    # gamma correction -> detonemap -> Exposure
    ldr_tens = gamma_correction(ldr_tens)
    ldr_tens = detonemap_reinhard(ldr_tens)
    ldr_tens *= exposure
    return ldr_tens




print("INFO: Used device: ", CurrentDevice)

## Train Network
#avg_loss_per_epoch = []
#model.train() # prep model for training
#for epoch in range(num_epochs):

#    #Log pass
#    print('Epoch: %03d' % (epoch + 1), end="\n")
#    avg_train_loss = 0.0

#    if epoch % 5 == 0:
#        model_save_path = Path("E:/MASTERS/Upscaler/Models/model_float32.pth")
#        torch.save({
#            'epoch': epoch,
#            'batch_size': batch_size,
#            'lr': learning_rate,
#            'Dataset': 'Dataset_UE',
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': criterion.state_dict(),
#            }, 
#           model_save_path)

#    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

#        #Zero gradients
#        optimizer.zero_grad()

#        # Get data to cuda if possible
#        data = data.to(device=CurrentDevice, dtype=torch.float32)
#        target = target.to(device=CurrentDevice, dtype=torch.float32)

#        # 1./Exposure -> tonemap -> inverse gamma correction
#        data = postprocessing_pipeline(data)
#        #save_exr("E:/MASTERS/Upscaler/data_after_postprocess_1080p_new.exr", data.cpu()[0].half(), channels=["R", "G", "B"])
#        #break
#        target = postprocessing_pipeline(target)


#        # forward
#        pred = model(data)
#        loss = criterion(pred, target)
#        # accumulate loss, loss * amount N batch size
#        avg_train_loss += loss.item() * data.size(0)

#        # loss backward and optimizer
#        loss.backward()
#        optimizer.step()

#    # divide avg train loss by length of data loader sampler
#    # it will give a correct avg loss
#    # if divided by batch_size, then sometimes it may be not correct,
#    # because batch_size is sometimes not dividable by num of samples
#    avg_train_loss = avg_train_loss / len(train_loader.sampler)
#    avg_loss_per_epoch.append(avg_train_loss)

#    #Log pass
#    print(' Avg loss: %.3f' % avg_train_loss, end="\n")


#import matplotlib.pyplot as plt
## summarize history for loss
#fig, axs = plt.subplots(figsize = (20,6))
#plt.plot(avg_loss_per_epoch)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()


#model_save_path = Path("E:/MASTERS/Upscaler/Models/model_float32.pth")
#torch.save({
#            'epoch': epoch,
#            'batch_size': batch_size,
#            'lr': learning_rate,
#            'Dataset': 'Dataset_UE',
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': loss,
#            }, 
#           model_save_path)




# Inference time
test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
        csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),)
#test_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers"),
#        csv_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"),)

model.load_state_dict(torch.load(Path("E:/MASTERS/Upscaler/Models/model_float32.pth"))['model_state_dict'])

model.eval()
import matplotlib.pyplot as plt
with torch.no_grad():
    # Plotting part
    figure = plt.figure(figsize=(20, 20))
    lr, hr = test_ds[50]
    lr = postprocessing_pipeline(lr).unsqueeze(0).to(device=CurrentDevice,
    dtype=torch.float32)



    tens = load_exr_file("E:/MASTERS/UE4/00338.exr").unsqueeze(0).to(device=CurrentDevice,
    dtype=torch.float32)

    tens = postprocessing_pipeline(tens)
    print(torch.isnan(tens).any())
    pred = model(torch.nan_to_num(tens))
    #pred = lr
    #plt.imshow(pred.squeeze(0).permute(1,2,0).to(dtype=torch.float32).cpu().detach().numpy())
    #plt.show()

    save_exr("E:/MASTERS/Upscaler/Results/lr1.exr", depostprocessing_pipeline(tens).cpu().squeeze(0),
    channels=["R", "G", "B"])
    save_exr("E:/MASTERS/Upscaler/Results/pred_hdr1.exr",
    pred.cpu().squeeze(0), channels=["R",
    "G", "B"])

    save_exr("E:/MASTERS/Upscaler/Results/pred_ldr1.exr",
    depostprocessing_pipeline(pred).cpu().squeeze(0).half(), channels=["R",
    "G", "B"])




if __name__ == "__main__":
    print("UPSCALER!")
