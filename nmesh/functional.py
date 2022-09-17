from gnutools.fs import load_config as _load_config, parent
import os
from gnutools.utils import RecNamespace


def load_config():
    # Priority is on global variable
    try:
        filename = os.environ["NMESH_CONFIG"]
        cfg = _load_config(filename)
    # Then default path
    except:
        filename = f"{parent(__file__)}/config.yml"
        cfg = _load_config(filename)
    return cfg
