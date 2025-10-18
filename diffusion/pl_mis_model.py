import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_sparse import SparseTensor
from diffusion.co_datasets.mis_dataset import MISDataset

from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.pl_meta_model import COMetaModel
from diffusion.utils.mis_utils import mis_decode_np
from diffusion.consistency import MISConsistency


class MISModel(COMetaModel):
    def __init__(self, param_args=None, target_model=None, teacher_model=None, load_dataset=True):
        super(MISModel, self).__init__(param_args=param_args, node_feature_only=True)

        if load_dataset:
            # data_label_dir = None
            # if self.args.training_split_label_dir is not None:
            #     training_data_label_dir = os.path.join(self.args.training_split_label_dir)

            if self.args.training_split:
                self.train_dataset = MISDataset(
                    data_file=os.path.join(self.args.training_split),
                    data_label_dir=self.args.training_split_label_dir,
                )
            if self.args.test_split:
                self.test_dataset = MISDataset(
                    data_file=os.path.join(self.args.test_split),
                    data_label_dir=self.args.test_split_label_dir,
                )
            if self.args.validation_split:
                self.validation_dataset = MISDataset(
                    data_file=os.path.join(self.args.validation_split),
                    data_label_dir=self.args.validation_split_label_dir,
                )

        if self.args.consistency:
            self.consistency_trainer = MISConsistency(self.args, sigma_max=self.diffusion.T, boundary_func=self.args.boundary_func)

    def forward(self, x, t, edge_index):
        return self.model(x, t, edge_index=edge_index)

    def consistency_training_step(self, batch, batch_idx):
        loss = self.consistency_trainer.consistency_losses(self, batch)
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.consistency_training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, split='test'):
        return self.consistency_trainer.consistency_test_step(self, batch, batch_idx, split)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_state_dict'] = self.model.state_dict()
