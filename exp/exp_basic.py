import torch

from models import MODEL_REGISTRY


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = MODEL_REGISTRY
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            if self.args.use_gpu:
                print('Warning: GPU requested but not available, falling back to CPU')
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        raise NotImplementedError

    def train(self, setting):
        raise NotImplementedError

    def test(self, setting):
        raise NotImplementedError
