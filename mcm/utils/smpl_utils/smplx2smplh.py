import numpy as np


def smplx2smplh(positions_smplx: np.ndarray):
    smplh_positions = np.concatenate([positions_smplx[:, :22], positions_smplx[:, 25:]], axis=1)
    assert smplh_positions.shape[1] == 52
    return smplh_positions
