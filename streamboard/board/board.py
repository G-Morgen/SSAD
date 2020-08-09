from pathlib import Path

from streamboard.board.config import BoardConfig
from streamboard.board.loss import BoardLoss
from streamboard.board.confusion_matrix import BoardConfusionMatrix
from streamboard.board.semseg import BoardSemseg
from streamboard.board.widget import BoardWidget
from streamboard.board.augs import BoardAugs
from streamboard.board.dataset import BoardDataset


class Board(
    BoardConfig, BoardLoss, BoardConfusionMatrix, BoardSemseg, BoardWidget, BoardAugs, BoardDataset
):
    def __init__(self, base: Path):

        self.base = base
        self.cfg = self.init_config()
        self.S_loss, self.C_loss = self.init_loss()
        self.classifier_result = self.init_confusion_matrix()

        self.augs = {}
        self.dataset = {}
        for data_type in ["S", "C", "test"]:
            self.augs[data_type] = self.init_augs(data_type)
            self.dataset[data_type] = self.init_dataset(data_type)

        self.fn = self.init_fn()
