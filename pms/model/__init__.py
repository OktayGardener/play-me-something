from pms.model.deep_model import ContextModel
from pms.model.rnn_model import ContextRNNModel
from pms.model.deeper_model import DeeperContextModel

MODEL_CLASSES = {
    'ContextModel': ContextModel, 'ContextRNNModel': ContextRNNModel,
    'DeeperContextModel': DeeperContextModel
}


def get(model_name):
    return MODEL_CLASSES.get(model_name)
