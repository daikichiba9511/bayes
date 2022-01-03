"""
実装しかけ。メトロポリス法の詳細忘れた。元の自分のコードも間違ってる気がする
"""
from typing import Callable, Tuple, Union

import numpy as np


class MetroPolis:
    def __init__(
        self,
        loglikelihood_func: Callable[[np.ndarray], np.ndarray],
        shape: Union[int, Tuple[int, int]] = 1,
        dtype=np.float64,
    ) -> None:
        self._dtype = dtype
        self._shape = shape
        self._loglikelihood_func = loglikelihood_func

    def sample(
        self, n_iter: int = 100, seed: int = 42, step_size: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # set global random seed
        np.random.seed(seed)

        # initialize
        x = np.zeros(shape=self._shape, dtype=self._dtype)
        n_accept_cnt = 0

        for _ in range(n_iter):
            backup_x = x

            # logliklihood in Statistics, but also Hamiltonian in Physics
            action_init = self._loglikelihood_func(x)

            # uniform random
            dx = np.random.random(size=x.shape)
            dx = (dx - 0.5) * step_size * 2
            x = x + dx

            # final state
            action_fin = self._loglikelihood_func(x)

            # testing reject or accept from here
            # range : 0 ~ 1
            metropolis = np.random.random(size=x.shape)
            # accept
            # I mitht get wrong. so, should review this process after
            if all(np.exp(action_init - action_fin) > metropolis):
                n_accept_cnt += 1

            # reject
            else:
                x = backup_x

        return x, n_accept_cnt, n_accept_cnt / n_iter


def test_metropolis():
    def loglikelihood(x: np.ndarray) -> np.ndarray:
        return 0.5 * x ** 2

    metropolis = MetroPolis(loglikelihood)
    print(metropolis.sample())

    metropolis_multiple = MetroPolis(loglikelihood, shape=(10,))
    print(metropolis_multiple.sample())

if __name__ == "__main__":
    test_metropolis()
