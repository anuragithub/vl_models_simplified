import os
import yaml

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(ROOT_DIR, 'src', 'config', 'config.yaml')


def get_config():
    with open(config_path, 'r') as stream:
        config_loaded = yaml.safe_load(stream)
    return config_loaded


class AverageMeter(object):
    # taken from imagenet example
    # https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L287
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
