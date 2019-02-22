import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from seaborn import heatmap
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sns.reset_orig()

def reorder_cluster_ids(cluster_ids, cluster_params, n_clusters):
    n_gibbs = len(cluster_ids)
    n_obs = len(cluster_ids[0])
    for i in range(n_gibbs):
        c_ids = cluster_ids[i]
        c_par = cluster_params[i]
        curr = 0
        orig_order = []
        new_c_ids = np.full(n_obs, -1)
        for n in range(n_obs):
            if c_ids[n] not in orig_order:
                orig_order.append(c_ids[n])
                new_c_ids[c_ids == c_ids[n]] = curr
                curr += 1
        cluster_ids[i] = new_c_ids
        cluster_params[i] = c_par[orig_order]
    return cluster_ids, cluster_params, n_clusters

def vis_heatmap(cluster_ids, show_dendrogram=False):
    
    # Construct similarity matrix
    calc_matching_matrix = lambda x: 1 - (pairwise_distances(x.reshape(-1, 1)) > 0)
    all_sim_mats = [calc_matching_matrix(x) for x in cluster_ids]
    sim_mat = np.mean(all_sim_mats, axis=0)
    sim_mat = pd.DataFrame(sim_mat)
    
    # Create dendrogram
    Z = linkage(sim_mat.values, 'ward')
    dn = dendrogram(Z, no_plot=~show_dendrogram)
    new_idx = np.array(dn['ivl'], dtype='int')
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))
    
    # Find Gibbs sample that best matches similarity matrix
    sim_mat_dists = np.array([np.mean((x - sim_mat.values) ** 2) for x in all_sim_mats])
    best_gibbs = np.argmin(sim_mat_dists)
    new_idx = np.argsort(cluster_ids[best_gibbs])
    heatmap(sim_mat.loc[new_idx, new_idx], ax=ax, cmap='coolwarm')
    
    # Add chosen clustering in outline
    counts = np.bincount(cluster_ids[best_gibbs])
    N = len(new_idx)
    start = 0
    for c in counts:
        rect = patches.Rectangle((start, N-start), c, -c, linewidth=5, edgecolor='lime', facecolor='none')
        start += c
        ax.add_patch(rect)
    ax.set_xlim([-.25, N+.25])
    ax.set_ylim([-.25, N+.25])
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    all_best_gibbs = np.where([np.array_equal(cluster_ids[best_gibbs], x) for x in cluster_ids])[0]
    return sim_mat, all_best_gibbs, all_sim_mats

def plot_raster(raster, x_axis=None, ax=None, ms=10, offset=0):
    plt.grid('off')
    n_trials, trial_len = raster.shape
    if x_axis is None:
        x_axis = np.arange(trial_len)
    for i in range(n_trials):
        mask = (raster[i] > 0)
        if ax:
            ax.plot(x_axis[mask], (i+1+offset) * np.ones(mask.sum()), 'k.', markersize=ms)
        else:
            plt.plot(x_axis[mask], (i+1+offset) * np.ones(mask.sum()), 'k.', markersize=ms)

def import_camera_ready_settings(text_font=16, number_font=14):
    sns.set_style("whitegrid")
    sns.set_context("poster")
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": text_font,               # LaTeX default is 10pt font.
        "axes.titlesize": text_font,
        "font.size": text_font,
        "legend.fontsize": text_font,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": number_font,
        "ytick.labelsize": number_font,
        "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
                ]
            }
    mpl.rcParams.update(pgf_with_latex)