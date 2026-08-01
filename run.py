import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.config import load_config
from utils.tools import seed_everything


def main():
    configs = load_config()
    seed_everything(configs.seed, configs.deterministic)
    print(configs)

    for ii in range(configs.itr):
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_pt{}_st{}_{}_{}'.format(
            configs.model_id,
            configs.model,
            configs.data,
            configs.features,
            configs.seq_len,
            configs.label_len,
            configs.pred_len,
            configs.d_model,
            configs.n_heads,
            configs.e_layers,
            configs.d_ff,
            configs.patch_len,
            configs.stride,
            configs.des,
            ii)

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
