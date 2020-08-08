import numpy as np

#TODO : 今は1変数でしかできないので多変数でもできるように拡張する
class MetroPolis:

    def __init__(self, loglikli):
        self.loglikli = loglikli

    def sample(self, n_iter=100, seed=42):
        step_size = 0.5e0

        np.random.seed(seed)

        # initialize
        x = 0e0
        n_accept_cnt = 0e0

        for iter in range(1, n_iter+1):
            backup_x = x

            # logliklihood in Statistics, but also Hamiltonian in Physics
            action_init = self.loglikli(x)
    
            # uniform random
            dx = np.random.random()
            dx = (dx - 0.5e0) * step_size * 2e0
            x = x + dx

            # final state
            action_fin = self.loglikli(x)

            # testing reject or accept from here
            metropolis = np.random.random()

            # accept
            if np.exp(action_init - action_fin) > metropolis:
                n_accept_cnt += 1

            # reject
            else:
                x = backup_x

        return x, n_accept_cnt, n_accept_cnt/n_iter

def metropolis_test():

    def loglikli(x):
        return 0.5 * x * x

    metropolis = MetroPolis(loglikli=loglikli)
    print(metropolis.sample())



if __name__ == "__main__":
    metropolis_test()