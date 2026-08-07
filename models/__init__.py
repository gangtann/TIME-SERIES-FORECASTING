from models import (DLinear, DUET, PatchMLP, PatchTST, PathFormer, PRISM, SimpleTM,
                    TimeFilter, TimeMixer, TimeMosaic, iTransformer, xPatch)

# 新模型：在 models/ 下新建 X.py（暴露 class Model），然后在这里注册
MODEL_REGISTRY = {
    'PatchTST': PatchTST,
    'PRISM': PRISM,
    'DLinear': DLinear,
    'DUET': DUET,
    'PatchMLP': PatchMLP,
    'PathFormer': PathFormer,
    'SimpleTM': SimpleTM,
    'TimeFilter': TimeFilter,
    'TimeMixer': TimeMixer,
    'TimeMosaic': TimeMosaic,
    'iTransformer': iTransformer,
    'xPatch': xPatch,
}
