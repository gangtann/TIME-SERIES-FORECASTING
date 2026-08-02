import os

import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.config import load_config
from utils.tools import seed_everything, tee_output


def main():
    configs = load_config()
    seed_everything(configs.seed, configs.deterministic)

    for ii in range(configs.itr):
        # 短目录名；完整参数看同目录下的 config.yaml 快照。
        # 注意：用 --set 改超参对比实验时记得同时改 des 或 model_id，否则同名覆盖
        setting = '{}_{}_{}'.format(configs.model_id, configs.des, ii)

        exp_dir = os.path.join(configs.checkpoints, setting)
        os.makedirs(exp_dir, exist_ok=True)
        configs.save(os.path.join(exp_dir, 'config.yaml'))

        with tee_output(os.path.join(exp_dir, 'train.log')):
            print(configs)
            exp = Exp_Long_Term_Forecast(configs)
            if configs.is_training:
                print('>>>>>>> start training : {} >>>>>>>'.format(setting))
                exp.train(setting)
                print('>>>>>>> testing : {} <<<<<<<'.format(setting))
                exp.test(setting)
            else:
                print('>>>>>>> testing : {} <<<<<<<'.format(setting))
                exp.test(setting, test=1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
