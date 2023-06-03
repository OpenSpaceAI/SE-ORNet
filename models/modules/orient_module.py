import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.modules.dgcnn import get_graph_feature


class OrientNet(nn.Module):
    """This is the OrientNet module for point clouds

    Args:
        xyz_s (torch.Tensor): A tensor of shape (B, N, 3) representing source point clouds, where each point is
            represented as a 3D coordinate.
        xyz_t (torch.Tensor): A tensor of shape (B, N, 3) representing target point clouds, where each point is
            represented as a 3D coordinate.

    Returns:
        outputs (torch.Tensor): angle_x, angle_y and global_d_pred
    """        
    def __init__(
        self,
        input_dims=[3, 64, 128, 256],
        output_dim=None,
        latent_dim=None,
        mlps: list = [],
        num_neighs: int = 24,
        input_neighs: int = 27,
        num_class: int = 8,
    ):
        super(OrientNet, self).__init__()
        self.num_neighs = num_neighs
        self.input_modules = []
        
        # input modules
        self.input_neighs = input_neighs
        input_dim = input_dims[0]
        for i in range(len(input_dims) - 1):
            output_dim = input_dims[i + 1]
            self.input_modules.append(
                EdgeConvModule(self.input_neighs, input_dim, output_dim)
            )
            input_dim = output_dim

        self.input_modules = nn.ModuleList(self.input_modules)
        
        self.orient_module = OrientModule(self.num_neighs, input_dim, latent_dim)
        self.edgeconv = EdgeConvModule(self.num_neighs, latent_dim, output_dim)
        
        # output mlps
        self.mlps = []
        input_dim = 2 * (latent_dim + output_dim)
        for dim in mlps:
            self.mlps.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
            input_dim = dim
        self.mlps = nn.ModuleList(self.mlps)

        self.num_class = num_class
        self.classifier = nn.Conv1d(input_dim, self.num_class, 1, bias=False)

        # global domain prediction
        self.global_netD1 = nn.Sequential(
            nn.Conv1d(2 * (latent_dim + output_dim),256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.global_netD2 = nn.Linear(128, 2)

    def forward(self, xyz_s, xyz_t):
        batch_size = xyz_s.shape[0]
        idx_s = knn(xyz_s, xyz_s, k=self.input_neighs)
        idx_t = knn(xyz_t, xyz_t, k=self.input_neighs)
        feature_s = xyz_s.transpose(1, 2)
        feature_t = xyz_t.transpose(1, 2)
        for input_module in self.input_modules:
            feature_s = input_module(feature_s,idx = idx_s)
            feature_t = input_module(feature_t,idx = idx_t)
        latent_s_0 = self.orient_module(xyz_s, xyz_t, feature_s, feature_t)
        latent_t_0 = self.orient_module(xyz_t, xyz_s, feature_t, feature_s)
        latent_s_1 = self.edgeconv(latent_s_0)
        latent_t_1 = self.edgeconv(latent_t_0)
        x = torch.cat((latent_s_0, latent_s_1), dim=1)
        y = torch.cat((latent_t_0, latent_t_1), dim=1)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [B, C_out]
        y1 = F.adaptive_max_pool1d(y, 1).view(batch_size, -1)  # [B, C_out]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [B, C_out]
        y2 = F.adaptive_avg_pool1d(y, 1).view(batch_size, -1)  # [B, C_out]
        x = torch.cat((x1, x2), 1).unsqueeze(-1)  # [B, 2C_out, 1]
        y = torch.cat((y1, y2), 1).unsqueeze(-1)  # [B, 2C_out, 1]

        D_input = torch.cat((x, y), dim=-1)
        global_d_pred = self.global_netD1(grad_reverse(D_input)) # Bx128x2
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx128
        global_d_pred = self.global_netD2(global_d_pred) # Bx2

        for m in self.mlps:
            x = m(x)  # B C 1
            y = m(y)  # B C 1
        angle_x = self.classifier(x).squeeze(-1)  # B 8
        angle_y = self.classifier(y).squeeze(-1)  # B 8

        outputs = dict(
            angle_x = angle_x,
            angle_y = angle_y,
            global_d_pred = global_d_pred,
        )
        return outputs


class OrientModule(nn.Module):
    def __init__(
        self,
        k: int,
        input_size: int,
        output_size: int,
    ):
        """init OrientModule

        Args:
            k (int): the number of neighbor
            input_size (int): feature size of input
            output_size (int): feature size of output
        """
        super(OrientModule, self).__init__()
        self.k = k
        self.linear = nn.Conv2d(
            (input_size + 3) * 2, output_size, kernel_size=1, bias=False
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.bn = nn.BatchNorm2d(output_size)

    def forward(
        self,
        xyz_s: torch.tensor,
        xyz_t: torch.tensor,
        feature_s: torch.tensor,
        feature_t: torch.tensor,
    ):
        """OrientModule forward

        Args:
            xyz_s (torch.tensor): [B, N, 3]
            xyz_t (torch.tensor): [B, N, 3]
            feature_s (torch.tensor): [B, C_in, N]
            feature_t (torch.tensor): [B, C_in, N]

        Returns:
            torch.tensor: output [B, C_out, N]
        """
        feature, xyz = get_graph_feature(
            feature_s.transpose(2, 1),
            feature_t.transpose(2, 1),
            k=self.k,
            ref_xyz=xyz_s,
        )  # B N K C_in, B N K 3
        xyz_t = xyz_t.unsqueeze(2).repeat(1, 1, self.k, 1)  # B N K 3
        feature_t = (
            feature_t.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k, 1)
        )  # B N K 2*C_in
        feature = torch.cat(
            (feature - feature_t, feature, xyz - xyz_t, xyz_t), dim=-1
        )  # B N K 2*(C_in + 3)
        feature = feature.permute(0, 3, 1, 2).contiguous()  # B 2*(C_in + 3) N K
        feature = self.relu(self.bn(self.linear(feature)))  # B C_out N K
        output, _ = feature.max(dim=-1, keepdim=False)  # B C_out N
        return output


class EdgeConvModule(nn.Module):
    def __init__(
        self,
        k: int,
        input_size: int,
        output_size: int,
    ):
        """init EdgeConvModule

        Args:
            k (int): the number of neighbor
            input_size (int): feature size of input
            output_size (int): feature size of output
        """
        super(EdgeConvModule, self).__init__()
        self.k = k
        self.linear = nn.Conv2d(input_size * 2, output_size, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.bn = nn.BatchNorm2d(output_size)

    def forward(
        self,
        input: torch.tensor,
        idx = None,
    ):
        """EdgeConvModule forward

        Args:
            feature (torch.tensor): [B, C_in, N]

        Returns:
            torch.tensor: output [B, C_out, N]
        """
        feature = get_graph_feature(
            input.transpose(2, 1), input.transpose(2, 1), k=self.k, idx=idx
        )  # B N K C_in
        input = (
            input.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k, 1)
        )  # B N K 2*C_in
        feature = torch.cat((feature - input, input), dim=-1)  # B N K 2*(C_in + 3)
        feature = feature.permute(0, 3, 1, 2).contiguous()  # B 2*(C_in + 3) N K
        feature = self.relu(self.bn(self.linear(feature)))  # B C_out N K
        output, _ = feature.max(dim=-1, keepdim=False)  # B C_out N
        return output


def knn(ref: torch.tensor, query: torch.tensor, k: int):
    """KNN Retrieve nearest neighbor indices

    Args:
        ref (torch.tensor): ref feature
        query (torch.tensor): query feature
        k (int): the number of neighbor

    Returns:
        idx: tensor, [B, N, K]
    """
    if torch.cuda.is_available():
        from knn_cuda import KNN

        ref = ref  # (batch_size, num_points, feature_dim)
        query = query
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)

    else:
        inner = -2 * torch.matmul(ref, query.transpose(2, 1))
        xx = torch.sum(ref.transpose(2, 1) ** 2, dim=1, keepdim=True)
        yy = torch.sum(query.transpose(2, 1) ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - yy.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def get_graph_feature(
    ref: torch.tensor, query: torch.tensor, k: int = 20, idx=None, ref_xyz=None
):
    """extract graph feature

    Args:
        ref (torch.tensor): ref feature [B, N, C]
        query (torch.tensor): query feature [B, N, C]
        k (int, optional): KNN. Defaults to 20.
        idx (_type_, optional): neighbor index. Defaults to None.

    Returns:
        output: tensor, [B, N, K, C]
    """
    batch_size, num_points, num_dims = ref.size()

    # get neighbor index
    if idx is None:
        idx = knn(ref, query, k=k)  # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size).to(idx).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # extract feature of neighbor
    feature = ref.reshape((batch_size * num_points, -1))[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    if ref_xyz is not None:
        xyz = ref_xyz.reshape((batch_size * num_points, -1))[idx, :]
        xyz = xyz.view(batch_size, num_points, k, 3)
        return feature, xyz
    return feature


class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1.0)

def grad_reverse(x):
    return GradReverse.apply(x)