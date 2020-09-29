import torch
import torch.nn as nn
import numpy as np

from dataset import SimpleDataLoader
from utils_model import (
    DNN,
    get_varlen_pooling_list
)


class MLP(nn.Module):
    def __init__(
        self,
        dnn_input, dnn_hidden_units, dnn_dropout,
        activation='relu', use_bn=True, l2_reg=1e-4, init_std=1e-4,
        device='cpu',
        feature_index={},
        embedding_dict={},
        dense_features=[],
        sparse_features=[],
        varlen_sparse_features=[],
        varlen_mode_list=[],
        embedding_size=8,
        batch_size=256,
    ):
        super().__init__()
        self.device = device
        self.feature_index = feature_index
        self.embedding_dict = embedding_dict
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.varlen_sparse_features = varlen_sparse_features
        self.varlen_mode_list = varlen_mode_list
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.reg_loss = torch.zeros((1,), device=device)

        self.dnn = DNN(
            dnn_input, dnn_hidden_units,
            activation='relu', l2_reg=l2_reg, dropout_rate=dnn_dropout, use_bn=use_bn,
            init_std=init_std, device=device
        )
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        # add regularization
        self.add_regularization_loss(self.embedding_dict.parameters(), l2_reg)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg)
        self.add_regularization_loss(self.dnn_linear.weight, l2_reg)

        self.out = nn.Sigmoid()

        self.to(device)

    def forward(self, X):

        dense_value_list = [
            X[:, self.feature_index[feat]: self.feature_index[feat] + 1] for feat in self.dense_features
        ]
        sparse_embedding_list = [
            self.embedding_dict[feat](
                X[:, self.feature_index[feat]: self.feature_index[feat] + 1].long()
            ) for feat in self.sparse_features
        ]
        varlen_sparse_embedding_list = get_varlen_pooling_list(
            self.embedding_dict, X, self.feature_index, self.varlen_sparse_features, self.varlen_mode_list, self.device
        )
        sparse_embedding_list = sparse_embedding_list + varlen_sparse_embedding_list

        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)

        dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)
        return y_pred

    def predict(self, x, batch_size=256):

        model = self.eval()
        test_loader = SimpleDataLoader(
            [torch.from_numpy(x.values)],
            batch_size=batch_size,
            shuffle=False
        )

        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x).squeeze()
                pred_ans.append(y_pred.cpu().detach().numpy())

        return np.concatenate(pred_ans)

    def add_regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = torch.zeros((1,), device=self.device)
        for w in weight_list:
            if isinstance(w, tuple):
                l2_reg = torch.norm(w[1], p=p, )
            else:
                l2_reg = torch.norm(w, p=p, )
            reg_loss = reg_loss + l2_reg
        reg_loss = reg_loss * weight_decay
        self.reg_loss = self.reg_loss + reg_loss.item()
