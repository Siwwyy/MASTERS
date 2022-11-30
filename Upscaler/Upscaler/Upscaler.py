from Config.Config_Utils import read_cfg
from Config.Config import *

# pth = Path("Config/config_yamls/config.yaml")
# print(read_cfg(pth))


#from NeuralNetworks.UNet import test

## test()

from Dataset.Dataset_UE import test_ds_ue

test_ds_ue()

#####################################################################
#from Config.Config import CurrentDevice
#from Dataset.Dataset_UE import Dataset_UE
#from NeuralNetworks.UNet import Model_UNET

#import torch.nn as nn
#import torchvision.transforms.functional as tvf
#from torch import optim  # For optimizers like SGD, Adam, etc.
#from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
#from tqdm import tqdm  # For nice progress bar when training the data!


#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = True






## Hyperparameters
#in_channels = 3
#out_channels = 3
#learning_rate = 0.001
#batch_size = 2
#num_epochs = 1


## Load Data
#train_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
#        csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),)
#train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, drop_last=True, pin_memory=False)


## Initialize network
#model = Model_UNET(in_channels=3, out_channels=3).to(device=CurrentDevice, dtype=torch.float16)

## Loss and optimizer
#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)



#def tonemap_reinhard(hdr_tens:TensorType) -> TensorType:
#    return hdr_tens / (1. + hdr_tens)


#def detonemap_reinhard(ldr_tens:TensorType) -> TensorType:
#    return ldr_tens / (1. - ldr_tens).clip(min=1e-4)




## Train Network
##for epoch in range(num_epochs):
##    print("Epoch {}".format(epoch))
##    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
##        # Get data to cuda if possible
##        data = data.to(device=CurrentDevice)
##        target = target.to(device=CurrentDevice)

##        data = tvf.resize(data, size=(200, 200))
##        target = tvf.resize(target, size=(200, 200))


##        data = tonemap_reinhard(data)
##        target = tonemap_reinhard(target)
##        ## Enables autocasting for the forward pass (model + loss)
##        #with torch.autocast(device_type="cuda"):
##        # forward
##        pred = model(data)
##        loss = criterion(pred, target)

##        # backward
##        optimizer.zero_grad()
##        loss.backward()


##        # gradient descent or adam step
##        optimizer.step()

##        #del pred, loss, data, target
##        #torch.cuda.empty_cache()


##del optimizer, train_loader
##torch.cuda.empty_cache()

##import matplotlib.pyplot as plt
##with torch.no_grad():
##    # Plotting part
##    figure = plt.figure(figsize=(10, 8))
##    cols, rows = 5, 5
##    dd, _ = train_ds[0]
##    #dd = tvf.resize(dd, size=(200, 200))
##    dd = dd.unsqueeze(0).to(device=CurrentDevice)
##    #aa = model(tonemap_reinhard(dd)).squeeze(0).permute(2,1,0).to(dtype=torch.float32)
##    aa = dd.squeeze(0).permute(2,1,0).to(dtype=torch.float32)
##    plt.imshow(aa.cpu().detach().numpy())
##    plt.show()






if __name__ == "__main__":
    print("UPSCALER!")
