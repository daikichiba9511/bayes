# 「ベイズ統計の理論と方法」 p119 例17
import numpy as np


class ReducedRankReg:
    """ for experiment using Reduced Rank Regression
    
    Pameters
    --------
    x1
        input data
    x2
        output data corresponding with x1
    q
        probability distribution don't contains parameters , and is not estimated
    """

    def __init__(self, x1 : np.ndarray, x2 : np.ndarray, q):
        self._x1 = x1
        self._x2 = x2
        self._N2 = x2.shape[0]
        self._q = q

    def __call__(self,A, B, sigma=0.1):
        return ( self._q(self._x1) / ( (2 * np.pi * sigma ** 2) ** (self._N2/2) ) ) \
                 * np.exp( - (1 / ( 2 * sigma ** 2)) ** np.norm(self._x2 - np.dot( np.dot(B, A), self._x1) ) )

    def _pior(self, A, B):
        """prior distribution of Reduced Rank Regression
        
        \varphi \varpropto \text{exp} \left( - 2.0 \cdot 10^{-5} ( \| A \| + \| B \|)  )

        """
        return np.exp(
            - 2.0 * 10**(-5) * ( np.power(np.norm(A), 2) + np.power(np.norm(B), 2) )
            )


