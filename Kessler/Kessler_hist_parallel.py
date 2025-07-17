import os
import numpy as np
import matplotlib
matplotlib.use("Agg")        # Use non-GUI backend in workers
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp

# -----------------------------------------------------------------------------
# MODULE‐LEVEL PLACEHOLDERS (populated by init_worker in each process)
# -----------------------------------------------------------------------------
_bootstraps = None   # will become array of shape (n_i, n_j, n_samples)
_features   = None   # list of feature names, length == n_j
_out_dir    = None   # output directory for PDFs

def freedman_diaconis_bins(data):
    """
    Freedman‐Diaconis Rule:
      bin_width = 2 * IQR / n^(1/3)
      bins      = data_range / bin_width
    """
    data = np.asarray(data)
    n    = data.size
    if n < 2:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr      = q75 - q25
    # avoid zero‐width
    bin_width = 2 * iqr / (n ** (1/3))
    if bin_width <= 0:
        return int(np.ceil(np.sqrt(n)))
    data_range = data.max() - data.min()
    return max(1, int(np.ceil(data_range / bin_width)))

def init_worker(bootstraps, features, out_dir):
    """
    Stash big arrays into globals.  Called once per worker.
    """
    global _bootstraps, _features, _out_dir
    _bootstraps = bootstraps
    _features   = features
    _out_dir    = out_dir

def process_coef(task):
    """
    Worker function: given (i, j), extract bootstrap samples,
    compute FD bins, plot & save histogram.
    """
    i, j = task
    data = _bootstraps[i, j]
    bins = freedman_diaconis_bins(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins=bins, color='skyblue', edgecolor='black', density=True)
    ax.set_title(f"Variable {i}, feature {j}: {_features[j]}", fontsize='x-large')
    ax.set_xlabel("Value", fontsize='x-large')
    ax.set_ylabel("Frequency", fontsize='x-large')
    ax.grid(True)
    fig.tight_layout()

    # build and save output filename
    fname = f"hist_i{i:02d}_j{j:02d}.pdf"
    out_path = os.path.join(_out_dir, fname)
    fig.savefig(out_path)
    plt.close(fig)

    return (i, j, out_path)

# -----------------------------------------------------------------------------
# MAIN: only executed in the "main" process.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) LOAD DATA ONCE
    file_path = input("Path to _esindy.pkl file: ").strip()
    with open(file_path, "rb") as f:
        bootstraps, features = pickle.load(f)

    # arrange axes to [n_i, n_j, n_samples]
    bootstraps = np.transpose(bootstraps, (1, 2, 0))
    n_i, n_j, _ = bootstraps.shape

    # 2) PREPARE OUTPUT DIRECTORY
    base = file_path.removesuffix("_esindy.pkl")
    out_dir = os.path.join(base, "E-SINDy")
    os.makedirs(out_dir, exist_ok=True)

    # 3) FLATTEN TASK LIST
    tasks = [(i, j) for i in range(n_i) for j in range(n_j)]

    # 4) SPAWN POOL
    n_procs = min(mp.cpu_count(), len(tasks))
    print(f"Launching {n_procs} workers for {len(tasks)} histograms…")
    with mp.Pool(
        processes=n_procs,
        initializer=init_worker,
        initargs=(bootstraps, features, out_dir)
    ) as pool:
        results = pool.map(process_coef, tasks)

    # 5) REPORT
    for i, j, path in results:
        print(f" • Saved histogram for (i={i}, j={j}) → {path}")
    print("Done.")