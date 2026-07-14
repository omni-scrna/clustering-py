#!/usr/bin/env python3
"""OPTICS clustering module (scikit-learn-backed) for omnibenchmark.

Same structure and entry points as the scanpy Leiden clustering module
(parse_args / load input / cluster / main, writing one clusters.tsv), but
OPTICS is a different kind of algorithm and that changes what it needs as
input:

Leiden runs on a graph — it needs a *precomputed* neighbor graph
({name}_neighbors.h5 from the knn entrypoint), because "neighbors" for
graph community detection is exactly that graph.

OPTICS is a density-based algorithm — it builds its reachability ordering
by repeatedly asking "who is near this point?" over raw coordinates, and
answering that question fast (via a ball tree / kd-tree) is the whole
point of the --algorithm knob below. This module reads the PCA embedding directly
({name}_pcas.tsv from the pca entrypoint, before the knn step) instead of
the neighbors bundle, and lets scikit-learn build its own neighbor search
using whichever algorithm is selected.

Output:
  {output_dir}/{name}_clusters.tsv — cell_id<TAB>cluster

Density-based clustering can leave points unassigned: cluster "-1" means
noise/outlier, a point that never became a core point or fell in the
reachability neighborhood of one. Leiden never does this — every cell
ends up in some community — so "-1" rows are new behavior to expect
downstream.

Nearest-neighbor search backend (--algorithm)
----------------------------------------------
OPTICS needs a lot of "who is near me" queries while building its
reachability ordering; --algorithm picks how those queries are answered:

  auto       let scikit-learn choose based on data shape/dimensionality
  ball_tree  hierarchical ball partitioning; keeps working well as the
             number of PCA dimensions grows
  kd_tree    axis-aligned space partitioning; fastest in low dimensions
             (roughly <20) but degrades as dimensionality grows
  brute      brute-force pairwise distances; slowest, but always correct
             and the only option for exotic/non-metric distances the
             tree structures can't index

Cluster extraction (--cluster_method)
--------------------------------------
OPTICS itself only produces a reachability ordering; one of two methods
turns that ordering into actual cluster labels:

  xi      (default) cuts the ordering at steep drops in reachability
          distance, governed by --xi; handles clusters of varying
          density, which is the main reason to pick OPTICS over DBSCAN
  dbscan  cuts the ordering at a single flat --eps threshold, recovering
          plain DBSCAN's behavior (uniform density assumption)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from sklearn.cluster import OPTICS

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from readers import read_embedding  # noqa: E402

log = logging.getLogger(__name__)


def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `CLUST-E` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="OPTICS clustering module (scikit-learn-backed)")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "CLUST-E")  # --pcas_tsv (dest: pcas)
    p.add_argument("--min_samples", type=int, required=True,
                   help="Points needed in a neighborhood for a point to be a core point "
                        "(same role as DBSCAN's min_pts); higher = fewer, denser clusters")
    p.add_argument("--algorithm", choices=["ball_tree", "kd_tree", "brute"],
                   required=True, help="Nearest-neighbor search backend — see module docstring")
    p.add_argument("--metric", type=str,
                   help="Distance metric for neighbor search, e.g. ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’ ")
    p.add_argument("--max_eps", type=float, default=float("inf"),
                   help="Max neighborhood radius considered when expanding from a point; "
                        "inf (default) considers all points, a finite value speeds up "
                        "large datasets at the cost of ignoring farther relationships")
    p.add_argument("--cluster_method", choices=["xi", "dbscan"], default="xi",
                   help="How to cut the reachability ordering into clusters — see module docstring")
    p.add_argument("--xi", type=float, default=0.05,
                   help="Steepness threshold for the xi cluster-extraction method "
                        "(ignored when --cluster_method dbscan)")
    p.add_argument("--eps", type=float, default=None,
                   help="Flat reachability cutoff for the dbscan cluster-extraction method; "
                        "defaults to --max_eps if unset (ignored when --cluster_method xi)")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Kept for interface parity with the other clustering entrypoints; "
                        "OPTICS has no random component so this value is accepted but unused")
    return p.parse_args()


def cluster_optics(matrix, min_samples, algorithm, metric, max_eps,
                    cluster_method, xi, eps):
    """Run scikit-learn's OPTICS over a dense (n_cells, n_pcs) embedding matrix.

    Returns cluster labels as strings, one per row of `matrix`, in the same
    order; "-1" marks noise points that OPTICS did not assign to any cluster.
    """
    model = OPTICS(
        min_samples=min_samples,
        algorithm=algorithm,
        metric=metric,
        max_eps=max_eps,
        cluster_method=cluster_method,
        xi=xi,
        eps=eps,
    )
    labels = model.fit_predict(matrix)
    return [str(label) for label in labels]


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(f"Full command: {' '.join(sys.argv)}")
    args = parse_args()
    for k in ("output_dir", "name", "pcas", "min_samples", "algorithm", "metric",
              "max_eps", "cluster_method", "xi", "eps", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    matrix, cell_ids = read_embedding(args.pcas)
    labels = cluster_optics(
        matrix, args.min_samples, args.algorithm, args.metric, args.max_eps,
        args.cluster_method, args.xi, args.eps,
    )

    out = Path(args.output_dir) / f"{args.name}_clusters.tsv"
    pl.DataFrame({"cell_id": cell_ids, "cluster": labels}).write_csv(out, separator="\t")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
