
import numpy as np

from dataclasses import dataclass
from typing import Union

import redueced_rank_reg

np.random.seed(42)

class CreateData:
    """ create data used experiment　

    True distribution.

    Parameters
    ----------
    sample_num 
        the number of samples. In other words it's the unit num as Group is samples from sampled at once

    sample_size
        sample size is the number of the point contained samples sampled at once.
    
    """
    def __init__(self,sample_num : int = 1000,  sample_size : int = 6):
        self._the_num_samples = sample_num
        self._sample_size = sample_size

    
    def sample(self, A0, B0) -> Union[np.ndarray, float]:
        return np.random.random_sample(self._sample_size)

    def samples(self) -> np.ndarray:
        return np.array([ self.sample() for _ in range(self._the_num_samples)])


@dataclass(frozen=True)
class Samples:
    """ immutable classes contains data created CreateData class I defined and immutable
    """
    x1 : np.ndarray
    x2 : np.ndarray


@dataclass(frozen=True)
class Parameters:
    A : np.ndarray
    B : np.ndarray


def check_rank(A : np.ndarray , B : np.ndarray, H0 : int = 3) -> int:
    """ check the rank of BA where B, A is matrix
    
    """
    rankH = np.linalg.matrix_rank(np.dot(B, A))
    return rankH == H0

def waic(samples : np.ndarray, loglikeli: np.ndarray ,beta=1) -> np.float64:
    """ calculate waic

    E[ Gn ] = E[ Wn ] + o_p(1/n)

    where. Gn is generalization error. true distribution is defined as q(x) and prediction distribution is defined as p*(x).

    D(q||p) is Kullback Leibler divergence. It measures diferrence q(x) and p(x) .

    $$ D(q||p) = \int q(x) \text{log} \frac{q(x)}{p(x)} dx$$

    where $ - \int q(x) \text{log} p*(x) dx $ is defined as Gn , and is called as generalization error.

    o_p(1/n) is the reduce parts at the speed earlier than 1/n

    Parameters
    ----------
    samples
        mcmc samples
    
    loglikli
        log of likelihood calculated by MCMC
        
        where
        if you will calculate mean in hidimension,
        
        * $$ \int f(w)p(w)dw \simeq \frac{1}{K} \sum _ {k=1} ^ {K} f(w _ k) $$
        * p(w) is posterior dist.

    beta
        inverse temperture

    Returns
    -------
    Wn
        Wn is WAIC estimates generalization error
    """
    n = samples.shape[0]
    K = loglikeli.shape[0]
    # Empirical Risk (defined p9 in reference[1])
    Tn = - (1 / n) * np.sum(np.log(samples))
    # Functional Dispersion (defined p117 in reference[1])
    Vn = np.sum( (1/K) * np.sum(np.power(loglikeli, 2)) - np.power((1/K) * (np.sum(loglikeli)), 2))
    # waic (defined p118 in reference[1])
    Wn = Tn + ( (beta * Vn) / n)
    return Wn


def prepare():
    H = 1
    mat_is_ok = True
    cs = CreateData()
    samples = Samples(x1=cs.samples(), x2=cs.samples())
    A0 = 0
    B0 = 0
    while mat_is_ok :
        true_prams = Parameters(
            A=np.random.random(size=(len(samples.x1), H)),
            B=np.random.random(size=(H, len(samples.x2)))
        )

        if check_rank(A=true_prams.A, B=true_prams.B):
            mat_is_ok = False
            A0 = true_prams.A
            B0 = true_prams.B

    print(np.linalg.matrix_rank(np.power(B0, A0)))
    return A0, B0


if __name__ == "__main__":
    prepare()