from NN.Config.ConfigUtils.TrainingConfig import DispatchParams

__all__ = ["dispatchTraining"]


def dispatchTraining(dispatchParams: DispatchParams = None) -> None:
    print(dispatchParams)
