from Config.Config_Utils import read_cfg
from Config.Config import *

# pth = Path("Config/config_yamls/config.yaml")
# print(read_cfg(pth))


from NeuralNetworks.UNet import test

# test()

from Dataset.Dataset_UE import test_ds_ue

test_ds_ue()


if __name__ == "__main__":
    print("UPSCALER!")
