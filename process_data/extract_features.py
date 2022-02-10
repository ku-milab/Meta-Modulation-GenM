import torch, os
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models import *
from process_data.data_loader import *
from util import *
from solver import Baselearner

class ExtractFeature(Baselearner):
    def __init__(self, flags):
        super(ExtractFeature, self).__init__(flags)

    def set_path(self, flags):
        super(ExtractFeature, self).set_path(flags)
        self.feature_path = os.path.join(flags.feat_path, flags.exp, str(flags.fold))
        if not os.path.isdir(self.feature_path):
            os.makedirs(self.feature_path)

    def setup_data(self, flags):
        self.dataloader = MultiDomain()
        self.trainset, self.valset, self.testset, self.trunc = self.dataloader.load_onedomain(flags.data_path, flags.target, flags.fold, flags.batch_size, flags.sw)

    def build_models(self, flags):
        super(ExtractFeature, self).build_models(flags)

    def save_feature(self, flags):
        checkpoint = torch.load(os.path.join(self.model_path, flags.checkpoint))
        self.model_feature.load_state_dict(checkpoint['model_feature'])
        self.model_task.load_state_dict(checkpoint['model_task'])
        self.optim_feat.load_state_dict(checkpoint['optim_feat'])
        self.optim_task.load_state_dict(checkpoint['optim_task'])

        self.model_feature.eval(), self.model_task.eval()

        x, y = self.testset
        x, y = x.type(torch.FloatTensor).to(flags.device), y.type(torch.LongTensor).to(flags.device)
        features = self.model_feature(x)
        logits = self.model_task(features)

        print(features.size())

        np.save(os.path.join(self.feature_path, '{}_feat_f{}'.format(flags.target, flags.fold)), features.detach().cpu().numpy().astype(np.float))