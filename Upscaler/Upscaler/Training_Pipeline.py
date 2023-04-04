from typing import Dict, Union, Any
from pathlib import Path
from tqdm import tqdm  # For nice progress bar when training the data!
from datetime import date

# Own imports
from Config.Config import PathType, CurrentDevice
from Config.DefaultConfigs import (
    ModelHyperparameters,
    ConfigMapping,
    initObjectFromConfig,
)
from NeuralNetworks.NN_Base import Model_Base
from Dataset.Dataset_UE import save_exr, Dataset_UE, FullDataset_UE
from Colorspace.PreProcessing import preprocessing_pipeline, depreprocessing_pipeline
from Utils.Utils import save_model, load_model

# Libs imports
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np

# Turn on cudnn backend and benchmark for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def save_checkpoint(
    model_save_path: PathType = None,
    model_name: str = None,
    epoch: int = 0,
    model: Model_Base = None,
    hyperparams: ModelHyperparameters = None,
    optimizer: optim = None
):

    model_save_path = model_save_path / (model_name + ".pth")
    save_model(
        model_save_path,
        {
            "epoch": epoch,
            "batch_size": hyperparams.batch_size,
            "lr": hyperparams.learning_rate,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )


# summarize history for loss
import matplotlib.pyplot as plt


def plot_loss_valid(train_loss: list = None, valid_loss: list = None, epochs: int = 10):
    assert train_loss is not None and valid_loss is not None
    epochs_list = range(0, epochs)
    max_plot_value = (
        max(train_loss) if max(train_loss) > max(valid_loss) else max(valid_loss)
    )

    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title("Training and validation loss")
    plt.ylim(0.0, max_plot_value + 0.1)

    plt.plot(epochs_list, train_loss, "-", label="Training loss")
    plt.plot(epochs_list, valid_loss, "-", label="Validation loss")

    plt.ylabel("loss")
    plt.xlabel("epoch")

    # plt.xticks(range(0, epochs, int(epochs/10)))
    # plt.xticks(range(0, epochs, 1))
    plt.yticks(np.linspace(0.0, max_plot_value + 0.1, num=10))

    fig.tight_layout()
    plt.legend(["train", "valid"], loc="upper right")
    plt.show()


def training_pipeline(
    config: ConfigMapping = None, training: bool = True, model_load: bool = False
) -> Model_Base:
    """
    Training pipeline
    """
    # Init of Hyperparams, dataset, dataloader, model etc.
    hyperparams = config["hyperparameters"]["className"](
        **config["hyperparameters"]["args"]
    )
    train_ds = config["trainDS"]["className"](**config["trainDS"]["args"])
    valid_ds = config["validDS"]["className"](**config["validDS"]["args"])
    train_loader = initObjectFromConfig(
        config["trainDL"]["className"], train_ds, **config["trainDL"]["args"]
    )
    valid_loader = initObjectFromConfig(
        config["validDL"]["className"], valid_ds, **config["validDL"]["args"]
    )
    model = config["model"]["className"](**config["model"]["args"])
    criterion = config["criterion"]["className"](**config["criterion"]["args"])
    optimizer = initObjectFromConfig(
        config["optimizer"]["className"],
        model.parameters(),
        **config["optimizer"]["args"]
    )
    device = config["device"]
    dtype = config["dtype"]

    model.to(device=device, dtype=dtype)

    # If training is False, then just return model | TODO, rethink that
    if not training:
        return model

    # Load checkpoint
    if model_load:
        loaded_training_state_dict = load_model(config["model_load_path"])
        model.load_state_dict(loaded_training_state_dict["model_state_dict"])
        optimizer.load_state_dict(loaded_training_state_dict["optimizer_state_dict"])

    # Train Network
    avg_train_loss_per_epoch = []
    avg_valid_loss_per_epoch = []
    min_valid_loss = 9999.9  # kind of "max" value of valid loss to find minimal valid loss of specified training
    for epoch in range(hyperparams.num_epochs):

        # Log pass
        print("Epoch: %03d" % (epoch + 1), end="\n")

        #####################
        # Training Pipeline #
        #####################
        total_train_loss = 0.0
        model.train()  # prepare model for training
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

            # Zeroing gradients
            optimizer.zero_grad()

            # Get data to cuda if possible
            data = data.to(device=device, dtype=dtype)
            target = target.to(device=device, dtype=dtype)

            # save_exr(str("F:/MASTERS/TEST/DATA/HDR/data_hdr_iter{}.exr".format(batch_idx * epoch + batch_idx)), data.squeeze(0).cpu().half())
            # save_exr(str("F:/MASTERS/TEST/TARGET/HDR/target_hdr_iter{}.exr".format(batch_idx * epoch + batch_idx)), target.squeeze(0).cpu().half())

            # PreProcess the data
            data = preprocessing_pipeline(data)
            target = preprocessing_pipeline(target)

            #save_exr(str("F:/MASTERS/TEST/DATA/LDR/aadata_ldr_iter{}.exr".format(batch_idx * epoch + batch_idx)), data.squeeze(0).cpu().half())
            #save_exr(str("F:/MASTERS/TEST/TARGET/LDR/aatarget_ldr_iter{}.exr".format(batch_idx * epoch + batch_idx)), target.squeeze(0).cpu().half())

            # forward
            pred = model(data)
            loss = criterion(pred, target)

            # accumulate loss, loss * amount N batch size
            total_train_loss += loss.item() * data.size(0)

            # loss backward and optimizer
            loss.backward()
            optimizer.step()

        #######################
        # Validating Pipeline #
        #######################
        total_valid_loss = 0.0
        model.eval()  # prepare model for validation
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):

                # Get data to cuda if possible
                data = data.to(device=device, dtype=dtype)
                target = target.to(device=device, dtype=dtype)

                # PreProcess the data
                data = preprocessing_pipeline(data)
                target = preprocessing_pipeline(target)

                # save_exr(str("F:/MASTERS/TEST/DATA/LDR/data_ldr_iter{}.exr".format(batch_idx * epoch + batch_idx)), data.squeeze(0).cpu().half())
                # save_exr(str("F:/MASTERS/TEST/TARGET/LDR/target_ldr_iter{}.exr".format(batch_idx * epoch + batch_idx)), target.squeeze(0).cpu().half())

                # forward
                pred = model(data)
                loss = criterion(pred, target)

                # accumulate loss, loss * amount N batch size
                total_valid_loss += loss.item() * data.size(0)

        # divide avg train/valid loss by length of data loader
        # it will give a correct avg loss
        # if divided by batch_size, then sometimes it may be not correct,
        # because batch_size is sometimes not dividable by num of samples
        total_train_loss = total_train_loss / len(train_loader)
        total_valid_loss = total_valid_loss / len(valid_loader)
        avg_train_loss_per_epoch.append(total_train_loss)
        avg_valid_loss_per_epoch.append(total_valid_loss)

        # Model's Checkpoint saving
        if min_valid_loss > total_valid_loss:
            min_valid_loss = total_valid_loss
            save_checkpoint(
                config["model_save_path"],
                "model_float32_best",
                #"model_float32_best_epoch{}".format(str(epoch)),
                epoch,
                model,
                hyperparams,
                optimizer,
            )
            print(
                "Checkpoint saved at {} epoch with {:.4f} total valid loss".format(
                    epoch, total_valid_loss
                )
            )

        # Log pass
        print()
        print(" Total train loss: %.4f" % total_train_loss, end="\n")
        print(" Total valid loss: %.4f" % total_valid_loss, end="\n")
        print()

    # Plot train and valid loss at n epochs
    plot_loss_valid(
        avg_train_loss_per_epoch, avg_valid_loss_per_epoch, hyperparams.num_epochs
    )

    # Save model's checkpoint
    save_checkpoint(
        config["model_save_path"],
        "model_float32_final",
        hyperparams.num_epochs,
        model,
        hyperparams,
        optimizer,
    )

    return model
