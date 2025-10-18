import os
import numpy as np
import torch
import torch.utils.data

from diffusion.co_datasets.tsp_graph_dataset import TSPGraphDataset
from diffusion.pl_meta_model import COMetaModel
from diffusion.consistency import TSPConsistency


class TSPModel(COMetaModel):
    def __init__(self, param_args=None, load_dataset=True):
        super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

        if load_dataset:
            if self.args.training_split:
                self.train_dataset = TSPGraphDataset(
                    data_file=os.path.join(self.args.training_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )
            if self.args.test_split:
                self.test_dataset = TSPGraphDataset(
                    data_file=os.path.join(self.args.test_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )
            if self.args.validation_split:
                self.validation_dataset = TSPGraphDataset(
                    data_file=os.path.join(self.args.validation_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )

        if self.args.consistency:
            self.consistency_trainer = TSPConsistency(
                self.args,
                sigma_max=self.diffusion.T,
                boundary_func=self.args.boundary_func,
            )

    def forward(self, x, adj, t, edge_index):
        return self.model(x, t, adj, edge_index)

    def consistency_training_step(self, batch, batch_idx):
        loss = self.consistency_trainer.consistency_losses(self, batch)
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.consistency_training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, split="test"):
        return self.consistency_trainer.consistency_test_step(
            self, batch, batch_idx, split
        )

    def run_save_numpy_heatmap(
        self, adj_mat, np_points, real_batch_idx, split, cost=None, path=None
    ):
        if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
            raise NotImplementedError("Save numpy heatmap only support single sampling")
        exp_save_dir = os.path.join(
            self.logger.save_dir, self.logger.name, self.logger.version
        )
        heatmap_path = (
            os.path.join(exp_save_dir, "numpy_heatmap") if path is None else path
        )
        # rank_zero_info(f"Saving heatmap to {heatmap_path}")
        os.makedirs(heatmap_path, exist_ok=True)
        real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
        np.save(
            os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat
        )
        np.save(
            os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"),
            np_points,
        )
        if cost is not None:
            np.save(
                os.path.join(heatmap_path, f"{split}-cost-{real_batch_idx}.npy"), cost
            )

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split="val")

    def tour2adj(self, tour, points, sparse, sparse_factor, edge_index):
        if not sparse:
            adj_matrix = torch.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
                adj_matrix[tour[i+1], tour[i]] = 1
        else:
            adj_matrix = np.zeros(points.shape[0], dtype=np.int64)
            adj_matrix[tour[:-1]] = tour[1:]
            adj_matrix = torch.from_numpy(adj_matrix)
            adj_matrix = (
                adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            )
            adj_matrix = torch.eq(edge_index[1].cpu(), adj_matrix).to(torch.int)
        return adj_matrix

    def points2adj(self, points):
        """
        return distance matrix
        Args:
          points: b, n, 2
        Returns: b, n, n
        """
        assert points.dim() == 3
        return (
            torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1) ** 0.5
        )
