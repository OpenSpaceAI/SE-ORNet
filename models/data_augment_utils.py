import torch
import numpy as np
import torch.nn as nn


class DataAugment(nn.Module):
    def __init__(
        self,
        operations: list = ["flip", "scale", "rotate", "noise"],
        flip_probability: float = 0.0,
        scale_range: list = [0.95, 1.05],
        rotate_nbins: int = 8,
        noise_variance: float = 0.0001,
    ):
        """Data Augmentation for point cloud

        Args:
            flip_probability (float, optional): the probability of flip. Defaults to 0.0.
            scale_range (list, optional): the scale range. Defaults to [0.95, 1.05].
            rotate_nbins (int, optional): the number of the rotation bins. Defaults to 8.
            noise_variance (float, optional): the variance of noise. Defaults to 0.0001.
        """
        super().__init__()
        self.operations = operations
        self.flip_probability = flip_probability
        self.scale_range = scale_range
        self.rotate_nbins = rotate_nbins
        self.noise_variance = noise_variance

    def forward(self, batch_data: torch.Tensor):
        """Data Augmentation for point cloud

        Args:
            batch_data (torch.Tensor): A tensor of shape (B, N, 3) representing B point clouds, where each point is
            represented as a 3D coordinate.

        Returns:
            torch.Tensor: A tensor of the same shape as `batch_data` representing the point cloud after the augmentation.
        """
        B = batch_data.shape[0]
        if "flip" in self.operations:
            flip_axes = np.random.choice(
                a=[0, 1, 2, 3],
                size=B,
                p=[
                    self.flip_probability / 3,
                    self.flip_probability / 3,
                    self.flip_probability / 3,
                    1.0 - self.flip_probability,
                ],
            )
            batch_data = flip_pointcloud(batch_data, flip_axes)
        if "scale" in self.operations:
            scales = np.random.uniform(self.scale_range[0], self.scale_range[1], B)
            batch_data = scale_pointcloud(batch_data, scales)
        if "rotate" in self.operations:
            batch_data, rotated_gt = rotate_by_z_axis(batch_data)
        if "noise" in self.operations:
            batch_data = add_noise_to_pointcloud(batch_data, self.noise_variance)
        if "rotate" in self.operations:
            return batch_data, rotated_gt
        else:
            return batch_data


def save_point_with_RGB(save_path: str, points: np.array, colors: np.array = None):
    """save point cloud to .obj file

    Args:
        save_path (str): the path of file
        points (np.array): (N 3)
        colors (np.array, optional): (N 3). Defaults to None.
    """
    fs = open(save_path, "w")
    for vid in range(points.shape[0]):
        if colors != None:
            fs.write(
                "v "
                + str(points[vid][0])
                + " "
                + str(points[vid][1])
                + " "
                + str(points[vid][2])
                + " "
                + str(colors[vid][0])
                + " "
                + str(colors[vid][1])
                + " "
                + str(colors[vid][2])
                + "\n"
            )
        else:
            fs.write(
                "v "
                + str(points[vid][0])
                + " "
                + str(points[vid][1])
                + " "
                + str(points[vid][2])
                + " "
                + str(0.0)
                + " "
                + str(0.5)
                + " "
                + str(0.0)
                + "\n"
            )
    fs.close()


def load_point_with_RGB(file_path: str, points_only: bool = True):
    """load point cloud from .obj file

    Args:
        file_path (str): the path of file
        points_only (bool, optional): load points without colors. Defaults to True.

    Returns:
        np.array: points: (N 3) colors: (N 3)
    """
    points = []
    colors = []
    fs = open(file_path, "r")
    for line in fs:
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == "v":
            p = tuple(map(float, values[1:4]))
            points.append(p)
            if points_only == False:
                c = tuple(map(float, values[4:7]))
                colors.append(c)
    fs.close()

    points = np.array(points, dtype=np.float32)
    if points_only == True:
        return points
    else:
        colors = np.array(colors, dtype=np.float32)
        return points, colors


def rotate_by_z_axis(batch_data: torch.Tensor, nbins: int = 8):
    """Rotate the point cloud along up direction with certain angle.

    Args:
        batch_data (torch.Tensor): A tensor of shape (B, N, 3) representing B point clouds, where each point is
            represented as a 3D coordinate.

    Returns:
        torch.Tensor: rotated_data: BxNx3; rotated_gt: B
    """
    B = batch_data.shape[0]
    Device = batch_data.device
    ANGLE = torch.arange(nbins) * 2 * torch.pi / nbins - torch.pi / 4
    rotated_gt = torch.randint(nbins, (B,)).to(Device)
    rotation_angle = ANGLE[rotated_gt]
    rotated_data = torch.zeros(batch_data.shape, dtype=torch.float32).to(Device)
    for k in range(B):
        cosval = torch.cos(rotation_angle[k])
        sinval = torch.sin(rotation_angle[k])
        rotation_matrix = torch.tensor(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        ).to(Device)
        pointcloud = batch_data[k, :, 0:3]
        centroids = torch.mean(pointcloud, dim=0, keepdim=True)[0]
        pointcloud = pointcloud - centroids
        rotated_data[k, :, 0:3] = (
            torch.mm(pointcloud.reshape((-1, 3)), rotation_matrix) + centroids
        )
    return rotated_data, rotated_gt


def add_noise_to_pointcloud(batch_data: torch.Tensor, noise_variance: float = 0.1):
    """
    Add random noise with specified variance to a point cloud.

    Args:
        batch_data (torch.Tensor): original batch of point clouds BxNx3
        noise_variance (float): The variance of the noise.

    Returns:
        torch.Tensor: A tensor of the same shape as `batch_data` representing the point cloud with added noise.
    """
    Device = batch_data.device
    noise = (
        torch.from_numpy(
            np.random.normal(0, np.sqrt(noise_variance), size=batch_data.shape)
        ).float()
    ).to(Device)
    return batch_data + noise


def scale_pointcloud(batch_data: torch.Tensor, scales: torch.Tensor):
    """
    Scale a point cloud by a given scaling factor.

    Args:
        batch_data (torch.Tensor): A tensor of shape (B, N, 3) representing B point clouds, where each point is
            represented as a 3D coordinate.
        scales (torch.Tensor): A tensor of shape (B,) representing the scaling factor for each point cloud.

    Returns:
        torch.Tensor: A tensor of the same shape as `batch_data` representing the scaled point cloud.
    """
    scales = torch.tensor(scales).to(batch_data.device).to(torch.float32)
    scales = scales.unsqueeze(-1).unsqueeze(-1)
    return batch_data * scales


def flip_pointcloud(batch_data: torch.Tensor, flip_axes: torch.Tensor):
    """
    Flips a point cloud based on the given flip axes.

    Args:
        batch_data (torch.Tensor): A tensor of shape (B, N, 3) representing B point clouds, where each point is
            represented as a 3D coordinate.
        flip_axes (torch.Tensor): A tensor of shape (B,) representing the flip axis for each point cloud.

    Returns:
        torch.Tensor: A tensor of shape (B, N, 3) representing the flipped point cloud.
    """
    Device = batch_data.device
    flip_axes = torch.tensor(flip_axes).to(Device)
    flip_matrices = []
    for flip_axis in flip_axes:
        if flip_axis < 3:
            flip_matrices.append(
                torch.eye(3)
                .to(Device)
                .index_put([flip_axis, flip_axis], torch.tensor(-1.0))
            )
        else:
            flip_matrices.append(torch.eye(3).to(Device))
    flip_matrices = torch.stack(flip_matrices)
    return torch.matmul(batch_data, flip_matrices)


if __name__ == "__main__":
    file_path = "output/vis/test.obj"
    points_0 = load_point_with_RGB(file_path).reshape((1, -1, 3))
    points_1 = load_point_with_RGB(file_path).reshape((1, -1, 3))

    batch_data = np.concatenate((points_0, points_1), axis=0)
    batch_data = torch.tensor(batch_data, device="cuda")
    # rotated_data, rotated_gt = rotate_by_z_axis(batch_data)
    # rotated_data = add_noise_to_pointcloud(rotated_data, 0.0001)
    # rotated_data = scale_pointcloud(rotated_data, [0.1, 10])
    # rotated_data = flip_pointcloud(rotated_data, [0, 2])

    # save_point_with_RGB("output/vis/1.obj", rotated_data[0])
    # save_point_with_RGB("output/vis/2.obj", rotated_data[1])

    student_aug = DataAugment(
        operations=["flip", "scale", "rotate", "noise"],
        flip_probability=0.5,
        scale_range=[0.9, 1.1],
        rotate_nbins=8,
        noise_variance=0.0001,
    )
    student_data, _ = student_aug(batch_data)

    teacher_aug = DataAugment(
        operations=["flip", "scale", "noise"],
        flip_probability=0.5,
        scale_range=[0.95, 1.05],
        noise_variance=0.00001,
    )
    teacher_data = teacher_aug(batch_data)

    rotated_data = student_data.cpu().numpy()
    save_point_with_RGB("output/vis/1.obj", rotated_data[0])
    save_point_with_RGB("output/vis/2.obj", rotated_data[1])
    rotated_data = teacher_data.cpu().numpy()
    save_point_with_RGB("output/vis/3.obj", rotated_data[0])
    save_point_with_RGB("output/vis/4.obj", rotated_data[1])
