import numpy as np


def mis_decode_np(predictions, adj_matrix, ep):
    """Decode the labels to the MIS."""
    solution = np.zeros_like(predictions.astype(int))
    # sorted_predict_labels = np.argsort(- predictions)
    csr_adj_matrix = adj_matrix.tocsr()

    # Compute denominator: sum of neighbors' predicted probabilities
    neighbor_sum = csr_adj_matrix @ predictions  # shape: (N,)
    scores = predictions / (ep + neighbor_sum)  # guided score
    sorted_predict_labels = np.argsort(- scores)

    for i in sorted_predict_labels:
        next_node = i

        if solution[next_node] == -1:
            continue

        solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
        solution[next_node] = 1

    return (solution == 1).astype(int)
