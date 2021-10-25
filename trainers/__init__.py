from .partial import Trainer


def load(config, world, manager):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config, world, manager)
    except KeyError:
        raise Exception("No such trainer: {}".format(cls_name))
