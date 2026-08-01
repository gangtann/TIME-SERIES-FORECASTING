from models import PatchTST

# 新模型：在 models/ 下新建 X.py（暴露 class Model），然后在这里注册
MODEL_REGISTRY = {
    'PatchTST': PatchTST,
}
