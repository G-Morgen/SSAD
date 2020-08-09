from typing import TypeVar, Any, List

from nptyping import NDArray

Tensor = TypeVar("torch.tensor")
Loss = TypeVar("torch.nn.modules.loss._Loss")
Optimizer = TypeVar("torch.optim.Optimizer")
DataLoader = TypeVar("torch.utils.data.DataLoader")
Module = TypeVar("torch.nn.Module")
DictConfig = TypeVar("omegaconf.DictConfig")
Compose = TypeVar("stad.albu.Compose")
Dataset = TypeVar("torch.utils.data.Dataset")
DataFrame = TypeVar("pandas.DataFrame")
Path = TypeVar("pathlib.Path")
