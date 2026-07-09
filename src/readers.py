"""Readers for tabular inputs shared across clustering entrypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def read_embedding(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a PCA embedding TSV produced by the ``pca`` entrypoint.

    Expected layout — header: ``cell_id<TAB>PC1<TAB>...<TAB>PCn``, one row
    per cell, numeric columns parsed as float64.

    Returns
    -------
    matrix : (n_cells, n_components) float64 array, row order matches cell_ids
    cell_ids : list[str], length n_cells
    """
    df = pl.read_csv(path, separator="\t")
    matrix = df.drop("cell_id").to_numpy().astype(np.float64)
    cell_ids = df["cell_id"].to_list()
    return matrix, cell_ids
