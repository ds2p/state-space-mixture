import numpy as np
from numba import jit
from utils.data import load_network
from utils.distributions_jit import samp_normal, samp_uniform, normal_logpdf, uniform_logpdf
from binomial_ssm_csmc import bssm_log_likelihood
from dirichlet_process import infer_dp

SEED = 1235
np.random.seed(SEED)

network_df = load_network()
obs_1ms_df = network_df.apply(lambda x: x.sum(0))
n_trials_1ms_df = network_df.apply(lambda x: x.shape[0])

n_ms = 5
obs_all = obs_1ms_df.apply(lambda x: x.reshape((-1, n_ms)).sum(1)).values.tolist()
n_trials_all = n_trials_1ms_df.values * n_ms
cue_time = 500 // n_ms

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

# Log likelihood function
@jit(nopython=True, nogil=True)
def calc_bssm_log_like_cue(ob, param, n):
    n_trials = n_trials_all[n]
    jump, log_var = param
    p_init = ob[:cue_time].sum() / (500 * n_trials // n_ms) 
    mean_init = np.log(p_init / (1 - p_init))
    var = np.exp(log_var)
    return bssm_log_likelihood(ob[cue_time:], var=var, n_particles=64, n_trials=n_trials, 
                               max_iters=3, mean_init=mean_init+jump, var_init=1e-10)

output = infer_dp(obs_all, calc_bssm_log_like_cue, 1.0, samp_base, base_logpdf, n_gibbs_iters=10000, 
                  sds_mh_proposal=1/2 * np.array([0.5, 0.5]), n_aux=5, dump_file='../pickle/cue.p', seed=SEED)
