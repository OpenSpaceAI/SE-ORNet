#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch_cluster import knn
from models.modules.dgcnn import get_graph_feature
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.orient_module import OrientNet


class DGCNN_MODULAR(nn.Module):
    def __init__(
        self,
        hparams,
        output_dim=None,
        use_inv_features=False,
        latent_dim=None,
        use_only_classification_head=False,
    ):  # bb - Building block
        super(DGCNN_MODULAR, self).__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = (
            latent_dim if latent_dim is not None else hparams.DGCNN_latent_dim
        )
        self.input_features = self.hparams.in_features_dim * 2
        self.use_inv_features = use_inv_features
        if self.use_inv_features:
            self.input_features = 4

            if self.hparams.use_sprin:
                self.input_features = 8
            if self.hparams.concat_xyz_to_inv:
                self.input_features += 3
        self.depth = self.hparams.nn_depth
        bb_size = self.hparams.bb_size
        output_dim = output_dim if output_dim is not None else self.hparams.latent_dim
        if not use_only_classification_head:
            self.convs = []
            for i in range(self.depth):
                in_features = (
                    self.input_features if i == 0 else bb_size * (2 ** (i + 1)) * 2
                )  # + 2*out_features
                out_features = bb_size * 4 if i == 0 else in_features

                if i == 3:
                    in_features = in_features

                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_features, out_features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_features),
                        nn.LeakyReLU(negative_slope=0.2),
                    )
                )
            last_in_dim = (
                bb_size * 2 * sum([2**i for i in range(1, self.depth + 1, 1)])
            )
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.latent_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
            self.convs = nn.ModuleList(self.convs)

        input_latent_dim = (
            self.latent_dim
            if not use_only_classification_head
            else self.hparams.DGCNN_latent_dim
        )

        hparams_mini = copy.deepcopy(hparams)
        hparams_mini.nn_depth = 2
        self.ANGLE = self.hparams.ANGLE
        self.linear1 = nn.Linear(input_latent_dim * 2, bb_size * 64, bias=False)
        self.bn6 = nn.BatchNorm1d(bb_size * 64)
        self.dp1 = nn.Dropout(p=self.hparams.dropout)

        self.linear2 = nn.Linear(bb_size * 64, bb_size * 32)
        self.bn7 = nn.BatchNorm1d(bb_size * 32)
        self.dp2 = nn.Dropout(p=self.hparams.dropout)

        self.linear3 = nn.Linear(bb_size * 32, output_dim)

        self.orientnet = OrientNet(
            output_dim=256, latent_dim=256, mlps=[256, 128, 128]
        )

    def forward_per_point(
        self, x_source, x_target, start_neighs_souce=None, start_neighs_target=None, norm=True,
    ):
        if norm == True:
            x_source = self.normalize_data(x_source)
            x_target = self.normalize_data(x_target)
        outputs = self.orientnet(
            x_source,
            x_target,
        )
        orient, orient_T = outputs["angle_x"], outputs["angle_y"]
        orient_list = [orient, orient_T]
        _, angle_index = orient.max(dim=-1)
        x_target = self.rotate_point_cloud_by_angle(x_target, angle_index)

        x_source = x_source.transpose(1, 2)  # DGCNN assumes BxFxN
        x_target = x_target.transpose(1, 2)
        
        if start_neighs_souce is None:
            start_neighs_souce = knn(x_source, k=self.num_neighs)
            start_neighs_target = knn(x_target, k=self.num_neighs)
        x_source = get_graph_feature(
            x_source,
            k=self.num_neighs,
            idx=start_neighs_souce,
            only_intrinsic=False if not self.use_inv_features else "concat",
        )  # only_intrinsic=self.hparams.only_intrinsic)
        other_source = x_source[:, :3, :, :]
        x_target = get_graph_feature(
            x_target,
            k=self.num_neighs,
            idx=start_neighs_target,
            only_intrinsic=False if not self.use_inv_features else "concat",
        )  # only_intrinsic=self.hparams.only_intrinsic)
        other_target = x_target[:, :3, :, :]
        if self.hparams.concat_xyz_to_inv:
            x_source = torch.cat([x_source, other_source], dim=1)
            x_target = torch.cat([x_target, other_target], dim=1)
        outs_source = [x_source]
        outs_target = [x_target]
        for conv in self.convs[:-1]:
            if len(outs_source) > 1:
                x_source = get_graph_feature(
                    outs_source[-1],
                    k=self.num_neighs,
                    idx=None
                    if not self.hparams.only_true_neighs
                    else start_neighs_souce,
                )
                x_target = get_graph_feature(
                    outs_target[-1],
                    k=self.num_neighs,
                    idx=None
                    if not self.hparams.only_true_neighs
                    else start_neighs_target,
                )
            x_source = conv(x_source)
            x_target = conv(x_target)
            x_source = x_source.max(dim=-1, keepdim=False)[0]
            x_target = x_target.max(dim=-1, keepdim=False)[0]
            outs_source.append(x_source)
            outs_target.append(x_target)
        x_source = torch.cat(outs_source[1:], dim=1)
        x_target = torch.cat(outs_target[1:], dim=1)
        features_source = self.convs[-1](x_source)
        features_target = self.convs[-1](x_target)
        orient_list = torch.stack(orient_list)
        return (
            features_source.transpose(1, 2),
            features_target.transpose(1, 2),
            orient_list,
            outputs["global_d_pred"]
        )
        # It is advised

    def get_theta(self, orient):

        orient = orient.softmax(-1)
        angle_negpi_to_pi = (self.ANGLE * orient).sum(-1)
        # angle_negpi_to_pi = torch.angle(orient[...,1] + 1j*orient[...,0]) + torch.pi/2
        return angle_negpi_to_pi

    def aggregate_all_points(self, features_per_point):
        if features_per_point.shape[1] == self.hparams.num_points:
            features_per_point = features_per_point.transpose(1, 2)
        batch_size = features_per_point.size(0)
        x1 = features_per_point.max(-1)[0].view(batch_size, -1)
        x2 = features_per_point.mean(-1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

    def rotate_point_cloud_by_angle(self, xyz, angles):
        """Rotate the point cloud along up direction with certain angle.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = torch.zeros(xyz.shape, dtype=torch.float32).cuda()
        for k in range(xyz.shape[0]):
            toss = angles[k]
            rotation_angle = -self.ANGLE[toss]
            cosval = torch.cos(rotation_angle)
            sinval = torch.sin(rotation_angle)
            rotation_matrix = torch.tensor(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            ).cuda()
            shape_pc = xyz[k, :, 0:3]
            rotated_data[k, :, 0:3] = torch.mm(
                shape_pc.reshape((-1, 3)), rotation_matrix
            )

        return rotated_data
    
    def normalize_data(self, batch_data):
        """ Normalize the batch data, use coordinates of the block centered at origin,
            Input:
                BxNxC array
            Output:
                BxNxC array
        """
        B, N, C = batch_data.shape
        normal_data = torch.zeros((B, N, C)).cuda()
        for b in range(B):
            pc = batch_data[b]
            centroid = torch.mean(pc, axis=0)
            pc = pc - centroid
            m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
            pc = pc / m
            normal_data[b] = pc
        return normal_data

    def forward(
        self,
        x_source,
        x_target,
        start_neighs_source,
        start_neighs_target,
        sigmoid_for_classification=False,
    ):

        features_source, features_target, orient_list, global_d_pred = self.forward_per_point(
            x_source,
            x_target,
            start_neighs_souce=start_neighs_source,
            start_neighs_target=start_neighs_target,
        )
        # features_aggregated = self.aggregate_all_points(features_per_point)
        # if sigmoid_for_classification:
        #     features_aggregated = torch.sigmoid(features_aggregated)

        return (
            features_source.transpose(1, 2),
            features_target.transpose(1, 2),
            orient_list,
            global_d_pred,
        )  # conv assumes B F N
