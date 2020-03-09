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
    def waic(self):
        if self.__waic is None:
            self.__waic = self.calc_waic(self.__samples)
        return self.__waic

    @property
    def wbic(self):
        if self.__wbic is None:
            self.__wbic = self.calc_wbic(self.__samples)
        return self.__wbic
    
    def calc_waic(self, samples):
        """
        Tn : 経験損失
        Vn : 汎関数分散
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
        waic = Tn +  (Vn/samples.shape[0])
        # print("waic", waic)
        return waic

    def calc_wbic(self, samples):
        """
        WBIC : beta = 1/log n の時の事後分布で対数尤度を重み付けしたもの
        ==params==
        samples: 対数尤度　log p(X_i|w)
        beta : 逆温度
        ==return==
        wbic :
        """
        
        wbic = - np.mean(np.sum(samples, axis=1))
        # print("wbic",wbic)
        return wbic


