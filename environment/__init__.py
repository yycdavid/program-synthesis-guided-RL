from .craft import CraftWorldHard
from .box import BoxWorld


def load(config, seed):
    cls_name = config.world.name
    try:
        cls = globals()[cls_name]
        return cls(config, seed)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))
