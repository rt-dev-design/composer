import torch
import numpy as np
from kmeans_pytorch import kmeans

# data
data_size, dims, num_clusters = 13, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)

# kmeans
cluster_ids_x, cluster_centers, cluster_iterations = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', tqdm_flag=True, device=torch.device('cuda:0')
)