from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import Tuple


def run_svd(input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=rank, n_iter=n_iter, random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd(input_matrix: np.ndarray, rank: int, n_iter: int,
                       random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd(input_matrix, rank, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_
