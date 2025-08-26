# Neural Network Engine

import hydra

from functools import partial
from omegaconf import DictConfig, OmegaConf

from NN.Config.ConfigUtils.Utils import CreateObjectfromConfig
from NN.Dataset import create_dataset
from NN.Config.ConfigUtils.TrainingConfig import DispatchParams, ModelHyperparameters
from NN.TrainingPipeline import dispatchTraining
from NN.Dataset import DatasetUE
from NN.Loss import MSELoss


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "NN/Config/ConfigFiles",
    "config_name": "config.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def config_main(cfg: DictConfig) -> None:
    print(cfg)
    print(OmegaConf.to_yaml(cfg))
    convertedCfg = OmegaConf.to_yaml(cfg)

    # Create objects from hydra config
    dataset = create_dataset(cfg.dataset)
    dataloader = CreateObjectfromConfig(cfg.dataloader, dataset=dataset)
    loss = CreateObjectfromConfig(cfg.loss)
    model = CreateObjectfromConfig(cfg.model)
    optimizer = CreateObjectfromConfig(cfg.optimizer, params=model.parameters())
    model_hyperaparemeters = CreateObjectfromConfig(cfg.hyperparams)

    # Create dispatch params
    dispatchParams = DispatchParams(
        model_hyperaparemeters, dataset, dataloader, loss, optimizer, model
    )

    # Training dispatch
    dispatchTraining(dispatchParams)


if __name__ == "__main__":
    config_main()


# # the metaclass will automatically get passed the same argument
# # that you usually pass to `type`
# def upper_attr(future_class_name, future_class_parents, future_class_attrs):
#     """
#       Return a class object, with the list of its attribute turned
#       into uppercase.
#     """
#     # pick up any attribute that doesn't start with '__' and uppercase it
#     uppercase_attrs = {
#         attr if attr.startswith("__") else attr.upper(): v
#         for attr, v in future_class_attrs.items()
#     }

#     # let `type` do the class creation
#     return type(future_class_name, future_class_parents, uppercase_attrs)

# __metaclass__ = upper_attr # this will affect all classes in the module

# class Foo(metaclass=upper_attr): # global __metaclass__ won't work with "object" though
#     # but we can define __metaclass__ here instead to affect only this class
#     # and this will work with "object" children
#     bar = 'bip'


# print(hasattr(Foo, 'bar'))
# print(Foo.__class__)

# print(Foo.__dict__)


# # remember that `type` is actually a class like `str` and `int`
# # so you can inherit from it
# class UpperAttrMetaclass(type):
#     # __new__ is the method called before __init__
#     # it's the method that creates the object and returns it
#     # while __init__ just initializes the object passed as parameter
#     # you rarely use __new__, except when you want to control how the object
#     # is created.
#     # here the created object is the class, and we want to customize it
#     # so we override __new__
#     # you can do some stuff in __init__ too if you wish
#     # some advanced use involves overriding __call__ as well, but we won't
#     # see this
#     def __new__(
#         upperattr_metaclass,
#         future_class_name,
#         future_class_parents,
#         future_class_attrs
#     ):
#         uppercase_attrs = {
#             attr if attr.startswith("__") else attr.upper(): v
#             for attr, v in future_class_attrs.items()
#         }
#         return type(future_class_name, future_class_parents, uppercase_attrs)


# class UpperAttrMetaclass(type):
#     def __new__(cls, clsname, bases, attrs):
#         uppercase_attrs = {
#             attr if attr.startswith("__") else attr.upper(): v
#             for attr, v in attrs.items()
#         }

#         # # Python 2 requires passing arguments to super:
#         # return super(UpperAttrMetaclass, cls).__new__(
#         #     cls, clsname, bases, uppercase_attrs)

#         # Python 3 can use no-arg super() which infers them:
#         return super().__new__(cls, clsname, bases, uppercase_attrs)
