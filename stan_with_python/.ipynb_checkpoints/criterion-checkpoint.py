import numpy as np



class Criterion(object):
    """
    calcurate infomation criterion (e.g WAIC, WBIC et.al)
    
    params:
        - samples : log likelihood 
    """
    def __init__(self, samples):
        self.__samples = samples
        self.__waic = None
        self.__wbic = None

    @property
    def samples(self):
        return self.samples

    @property
    def waic(self):
        if self.__waic is None:
            self.__waic = self._calc_waic(self.samples)
        return self.__waic

    @property
    def wbic(self):
        if self.__wbic is None:
            self.__wbic = self._calc_wbic(self.samples)
        return self.__wbic
    
    def _calc_waic(self, samples, beta=1):
        """
        ==params==
        samples: 対数尤度　log p(X_i|w)
        beta : 逆温度
        ==return==
        waic : 
        """
        
        Tn = -np.mean(np.log(np.mean(np.exp(samples), axis=0)))
        # print("Tn:",Tn)
        Vn = np.sum(np.mean(samples**2, axis=0) - np.mean(samples, axis=0)**2)
        # print("Vn",Vn)
        waic = Tn + beta * (Vn/samples.shape[0])
        # print("waic", waic)
        return waic

    def _calc_wbic(self, samples):
        """
        ==params==
        samples: 対数尤度　log p(X_i|w)
        beta : 逆温度
        ==return==
        wbic :
        """
        beta = 1/np.log(samples.shape[0])
        wbic = - np.mean(beta * np.sum(samples, axis=1))
        # print("wbic",wbic)
        return wbic

    def init_waic(self):
        self._waic = None
    
    def init_wbic(self):
        self._wbic = None