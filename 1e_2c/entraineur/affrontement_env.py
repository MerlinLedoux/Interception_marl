import os 
import sys

sys.path.append(os.path.abspath(r"C:\Users\FX643778\Documents\Git\Interception_marl\1e_2c\environement"))

from petting_zoo import AffrontementMultiZoo


def env_creator(config):
    return AffrontementMultiZoo(
        render_mode=config.get("render_mode", None)
    )
