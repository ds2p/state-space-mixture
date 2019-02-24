import numpy as np
from numba import jit
from utils.distributions_jit import samp_normal, samp_uniform, normal_logpdf, uniform_logpdf
from binomial_ssm_csmc import bssm_log_likelihood
from dirichlet_process import infer_dp

SEED = 1234
np.random.seed(SEED)

delta = 1 / 1000
obs_all = []

for _ in range(5):
    l1 = np.random.uniform(10, 15)
    l2 = l1 * np.exp(1)
    p1 = l1 * delta
    p2 = l2 * delta
    obs = np.hstack([np.random.binomial(225, p1, size=(100,)), np.random.binomial(225, p2, size=(300,))])
    obs_all.append(obs)
for _ in range(5):
    l1 = np.random.uniform(10, 15)
    l2 = l1 * np.exp(-1)
    p1 = l1 * delta
    p2 = l2 * delta
    obs = np.hstack([np.random.binomial(225, p1, size=(100,)), np.random.binomial(225, p2, size=(300,))])
    obs_all.append(obs)
for _ in range(5):
    l1 = np.random.uniform(10, 15)
    p1 = l1 * delta
    obs = np.random.binomial(225, p1, size=(400,))
    obs_all.append(obs)
for _ in range(5):
    l1 = np.random.uniform(10, 15)
    l2 = l1 * np.exp(1)
    p1 = l1 * delta
    p2 = l2 * delta
    obs = np.hstack([np.random.binomial(225, p1, size=(100,)), 
                   np.random.binomial(225, p2, size=(50,)), 
                   np.random.binomial(225, p1, size=(250,))])
    obs_all.append(obs)
for _ in range(5):
    l1 = np.random.uniform(10, 15)
    l2 = l1 * np.exp(-1)
    p1 = l1 * delta
    p2 = l2 * delta
    obs = np.hstack([np.random.binomial(225, p1, size=(100,)), 
                     np.random.binomial(225, p2, size=(50,)), 
                     np.random.binomial(225, p1, size=(250,))])
    obs_all.append(obs)

# base distribution (G)
@jit(nopython=True, nogil=True)
def samp_base(n):
    jumps = samp_normal(np.zeros(n), 2 * np.ones(n))
    variances = samp_uniform(np.full(n, -15), np.full(n, 0))
    all_samps = np.vstack((jumps, variances)).T
    return all_samps

@jit(nopython=True, nogil=True)
def base_logpdf(params):
    n = params.shape[0]
    jump_logpdf = normal_logpdf(params[:, 0], np.zeros(n), 2 * np.ones(n))
    variances_logpdf = uniform_logpdf(params[:, 1], np.full(n, -15), np.full(n, 0))
    total_logpdf = jump_logpdf + variances_logpdf
    return total_logpdf

for shift in [40, 32, 28]:
    
    # Log likelihood function
    @jit(nopython=True, nogil=True)
    def calc_bssm_log_like_sim(ob, param, n):
        jump, log_var = param
        p_init = ob[:100+shift].sum() / (45 * 5 * (100+shift)) 
        mean_init = np.log(p_init / (1 - p_init))
        var = np.exp(log_var)
        return bssm_log_likelihood(ob[100+shift:], var=var, n_particles=64, n_trials=45*5, 
                                   max_iters=3, mean_init=mean_init+jump, var_init=1e-10)

    output = infer_dp(obs_all, calc_bssm_log_like_sim, 1.0, samp_base, base_logpdf, n_gibbs_iters=10000, 
                      sds_mh_proposal=1/2 * np.array([0.5, 0.5]), n_aux=5, dump_file='../pickle/sim_shift{}.p'.format(shift), 
                      seed=SEED)
