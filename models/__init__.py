from .modular_ac import ModularACModel, ModularACConvModel
from .simple_nn import NNModel, NNConvModel, ConditionModel
from .relational import RelationalModel


def load(config):
    cls_name = config.model.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
