import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
# from diffusion.utils.cython_merge.cython_merge import merge_cython as merge_cython
from diffusion.utils.cython_merge.cython_merge_threshold import merge_cython_threshold as merge_cython_threshold

def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
    iterator = 0
    tour = tour.copy()
    with torch.inference_mode():
        cuda_points = torch.from_numpy(points).to(device)
        cuda_tour = torch.from_numpy(tour).to(device)
        batch_size = cuda_tour.shape[0]
        min_change = -1.0
        while min_change < 0.0:
            points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape(
                (batch_size, -1, 1, 2)
            )
            points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape(
                (batch_size, 1, -1, 2)
            )
            points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape(
                (batch_size, -1, 1, 2)
            )
            points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape(
                (batch_size, 1, -1, 2)
            )

            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(
                torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1)
            )
            A_i_i_plus_1 = torch.sqrt(
                torch.sum((points_i - points_i_plus_1) ** 2, axis=-1)
            )
            A_j_j_plus_1 = torch.sqrt(
                torch.sum((points_j - points_j_plus_1) ** 2, axis=-1)
            )

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(
                valid_change.reshape(batch_size, -1), dim=-1
            )
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode="floor")
            min_j = torch.remainder(flatten_argmin_index, len(points))

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1 : min_j[i] + 1] = torch.flip(
                        cuda_tour[i, min_i[i] + 1 : min_j[i] + 1], dims=(0,)
                    )
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break
        tour = cuda_tour.cpu().numpy()
    return tour, iterator

def cython_sub_merge(adj_mat, alpha, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_adj_mat, merge_iterations = merge_cython_threshold(
            adj_mat, alpha, seed
        )
        real_adj_mat = np.asarray(real_adj_mat)
    return real_adj_mat, merge_iterations

def merge_sub_tours(
    adj_mat,
    np_points,
    edge_index_np,
    sparse_graph=False,
    parallel_sampling=1,
    alpha=0.0,
    seed=None,
):

    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

    if not sparse_graph:

        splitted_adj_mat = [m[0] + m[0].T for m in splitted_adj_mat]
    else:
        splitted_adj_mat = [
            scipy.sparse.coo_matrix(
                (m, (edge_index_np[0], edge_index_np[1]))
            ).toarray()
            + scipy.sparse.coo_matrix(
                (m, (edge_index_np[1], edge_index_np[0]))
            ).toarray()
            for m in splitted_adj_mat
        ]

    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        arglist = []
        base_seed = 0 if seed is None else int(seed)
        for i, m in enumerate(splitted_adj_mat):
            arglist.append((m, float(alpha), base_seed + i))
        with Pool(parallel_sampling) as p:
            results = p.starmap(merge_cython_threshold, arglist)
    else:
        results = [
            cython_sub_merge(m.astype("double"), float(alpha), 0 if seed is None else int(seed) + i)
            for i, m in enumerate(splitted_adj_mat)
        ]

    splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

    tours = []
    for i in range(parallel_sampling):
        A = splitted_real_adj_mat[i]
        N = A.shape[0]
        deg = A.sum(axis=1)
        deg = np.asarray(deg, dtype=int)
        start_candidates = np.where(deg == 1)[0]
        start = int(start_candidates[0]) if len(start_candidates) > 0 else 0

        tour = [start]
        prev = -1
        while True:
            cur = tour[-1]
            nbs = np.nonzero(A[cur])[0]
            if prev != -1:
                nbs = nbs[nbs != prev]
            if len(nbs) == 0:
                break
            nxt = int(nbs[0]) if len(nbs) == 1 else int(nbs.max())
            tour.append(nxt)
            prev = cur
            if len(tour) > N:
                break
            if len(tour) >= 2 and tour[-1] == tour[0] and len(set(tour[:-1])) == N:
                break

        if len(set(tour)) == N and tour[-1] != tour[0]:
            tour.append(tour[0])

        tours.append(tour)

    merge_iterations = float(np.mean(splitted_merge_iterations))
    return tours, merge_iterations


class TSPEvaluator(object):
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost
