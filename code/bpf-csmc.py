import numpy as np
from binomial_ssm_csmc import particle_filter, bssm_log_likelihood
from utils.data import load_network, write_pickle
from numba import jit

network_df = load_network()
obs_1ms_df = network_df.apply(lambda x: x.sum(0))
n_trials_1ms_df = network_df.apply(lambda x: x.shape[0])

n_ms = 5
obs_all = obs_1ms_df.apply(lambda x: x.reshape((-1, n_ms)).sum(1)).values.tolist()
n_trials_all = n_trials_1ms_df.values * n_ms
cue_time = 500 // n_ms

#BPF
@jit(nopython=True, nogil=True)
def calc_bssm_log_like_bpf(ob, param, n, n_particles):
    n_trials = n_trials_all[n]
    jump, log_var = param
    p_init = ob[:cue_time].sum() / (500 * n_trials // n_ms) 
    mean_init = np.log(p_init / (1 - p_init))
    var = np.exp(log_var)
    return particle_filter(ob[cue_time:], var=var, n_particles=n_particles, n_trials=n_trials, 
                           mean_init=mean_init+jump, var_init=1e-10)

# cSMC
@jit(nopython=True, nogil=True)
def calc_bssm_log_like_csmc(ob, param, n, n_particles):
    n_trials = n_trials_all[n]
    jump, log_var = param
    p_init = ob[:cue_time].sum() / (500 * n_trials // n_ms) 
    mean_init = np.log(p_init / (1 - p_init))
    var = np.exp(log_var)
    return bssm_log_likelihood(ob[cue_time:], var=var, n_particles=n_particles, n_trials=n_trials, 
                               max_iters=3, mean_init=mean_init+jump, var_init=1e-10, ess_threshold=200)

idx = 25
jumps = np.linspace(-2, 2, 21)
log_vars = np.linspace(-10, 0, 11)

d = {}
for j in jumps:
    d[j] = {}
    for lv in log_vars:
        print(j, lv)
        a = []
        for _ in range(500):
            _, _, log_likelihoods, _, _ = calc_bssm_log_like_bpf(obs_all[idx], np.array([j, lv]), idx, n_particles=1024)
            a.append(log_likelihoods[-1])

        b = []
        for _ in range(500):
            b.append(calc_bssm_log_like_csmc(obs_all[idx], np.array([j, lv]), idx, n_particles=64))
        d[j][lv] = {'bpf': a, 'csmc': b}
        write_pickle(d, '../pickle/bpf-csmc.p')