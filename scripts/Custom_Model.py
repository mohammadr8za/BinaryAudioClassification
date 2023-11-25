import os
from ast_models import ASTModel
import torch.nn as nn

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'


class model:
    def __init__(self, input_tdim, input_fdim, label_dim, model_size='small224', drop_out=0):
        self.input_tdim = input_tdim
        self.input_fdim = input_fdim
        self.label_dim = label_dim
        self.drop_out = drop_out
        self.model_size = model_size

    def get_model(self):
        ast_mdl = ASTModel(label_dim=self.label_dim, input_tdim=self.input_tdim, input_fdim=self.input_fdim,
                           model_size=self.model_size,
                           imagenet_pretrain=False, audioset_pretrain=False, drop_out_rate=self.drop_out).to('cuda')

        embedding_dim = ast_mdl.original_embedding_dim

        ast_mdl.mlp_head = nn.Sequential(
            nn.LayerNorm((embedding_dim,), eps=1e-05, elementwise_affine=True),
            nn.Linear(in_features=embedding_dim, out_features=2, bias=True),
            nn.Sigmoid()
        ).to('cuda')

        for param in ast_mdl.parameters():
            param.requires_grad = True
        for param in ast_mdl.mlp_head.parameters():
            param.requires_grad = True

        return ast_mdl
