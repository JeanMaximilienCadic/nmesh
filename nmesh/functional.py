import yaml
from argparse import Namespace


def load_config():
    conf = yaml.load(open("nmesh.yml", "r"), Loader=yaml.FullLoader)
    ns = Namespace(**conf)
    return ns







