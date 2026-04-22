import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'AgileFormer'
# Model name
_C.MODEL.NAME = 'AgileFormer-Tiny'
_C.MODEL.PRETRAIN_CKPT = 'pretrained_ckpt/dat_pp_tiny_in1k_224.pth'
_C.MODEL.Params = CN(new_allowed=True)

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)



def get_config(cfg_file):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    _update_config_from_file(config, cfg_file)

    return config