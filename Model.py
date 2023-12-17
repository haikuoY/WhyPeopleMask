import mindspore as ms
ms.set_context(device_target='GPU')
from mindspore import nn, ops

class MyLoss(nn.LossBase):
    def __init__(self):
        super().__init__()
        
    def construct(self, x, y):        
        loss = nn.BCELoss('sum')(x, y)
        return loss

class NodeClassifer(nn.Cell):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc_0 = nn.Dense(c_in, 1000)
        self.fc_1 = nn.Dense(1000, 1000)
        self.fc_2 = nn.Dense(1000, 1000)
        self.fc_3 = nn.Dense(1000, 1)
        self.fc1 = nn.Dense(c_in, 1000)
        self.fc2 = nn.Dense(c_in, 1000)
        self.fc3 = nn.Dense(2000, 2000)
        self.fc4 = nn.Dense(2000, 1)
        self.LR1 = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def construct(self, node_feats, adj_matrix):
        x_0 = ops.bmm(adj_matrix, node_feats[...,:-8])
        x_0 = ops.cat([x_0,node_feats[...,-8:]], axis = -1)
        x = self.fc_0(x_0)
        x = self.LR1(x)
        x = self.fc_1(x)
        x = self.LR1(x)
        x = self.fc_2(x)
        x = self.LR1(x)
        x_1 = self.fc_3(x)
        # print("x_1", x_1)
        x1 = self.fc1(node_feats)
        x1 = self.LR1(x1)
        x2 = self.fc2(node_feats)
        x2 = self.LR1(x2)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x1 = x1.tile((1, x2.shape[1], 1, 1))
        x2 = x2.tile((1, 1, x1.shape[2], 1))
        x3 = ops.cat([x1,x2], axis = -1)
        adj1 = self.fc3(x3)
        adj1 = self.LR1(adj1)
        adj1 = self.fc4(adj1)
        adj1 = adj1.squeeze(-1)
        # print("adj1", adj1)
        x = ops.bmm(adj1, x_1)
        # print("x", x)
        x_sig = self.sig(x).squeeze(-1)
        # print("x_sig", x_sig)
        return x_sig
        # return x_sig, x, adj1, x_1


class NodeClassifer2(nn.Cell):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc_0 = nn.Dense(c_in, 1000)
        self.fc_1 = nn.Dense(1000, 1000)
        self.fc_2 = nn.Dense(1000, 1000)
        self.fc_3 = nn.Dense(1000, 1)
        self.fc1 = nn.Dense(c_in, 1000)
        self.fc2 = nn.Dense(c_in, 1000)
        self.fc3 = nn.Dense(2000, 2000)
        self.fc4 = nn.Dense(2000, 1)
        self.LR1 = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def construct(self, node_feats):
        # node_feats = data[0]
        # adj_matrix = data[1]
        # x_0 = ops.bmm(adj_matrix, node_feats[...,:-8])
        # x_0 = ops.cat([x_0,node_feats[...,-8:]], axis = -1)
        x = self.fc_0(node_feats)
        x = self.LR1(x)
        x = self.fc_1(x)
        x = self.LR1(x)
        x = self.fc_2(x)
        x = self.LR1(x)
        x_1 = self.fc_3(x)
        x_sig = self.sig(x_1).squeeze(-1)
        return x_sig
        # print("x_1", x_1)
        # x1 = self.fc1(node_feats)
        # x1 = self.LR1(x1)
        # x2 = self.fc2(node_feats)
        # x2 = self.LR1(x2)
        # x1 = x1.unsqueeze(1)
        # x2 = x2.unsqueeze(2)
        # # print("x1, x2 ", x1.shape, x2.shape)
        # x1 = x1.tile((1, x2.shape[1], 1, 1))
        # x2 = x2.tile((1, 1, x1.shape[2], 1))
        # # print("x1, x2 ", x1.shape, x2.shape)
        # x3 = ops.cat([x1,x2], -1)
        # adj1 = self.fc3(x3)
        # adj1 = self.LR1(adj1)
        # adj1 = self.fc4(adj1)
        # adj1 = adj1.squeeze(-1)
        # # print("adj1", adj1.shape, adj1)
        # x = ops.bmm(adj1, x_1)
        # # print("x", x)
        # x_sig = self.sig(x).squeeze(-1)
        # # print("x_sig", x_sig)
        # return x_sig