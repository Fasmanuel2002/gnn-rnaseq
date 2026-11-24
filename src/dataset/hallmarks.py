import csv

import numpy as np
import pandas as pd


def load_hallmarks(file_path : str) -> dict[str, set[str]]:
    """
    Read the MSigDB Hallmarks CSV.
    Each row looks like: HALLMARK_NAME, url, gene1, gene2, ..., (empty cells at the end)
    Returns: {"HALLMARK_NAME": {"gene1", "gene2", ...}}
    """
    hallmarks = {}

    with open(file_path) as file:
        for row in csv.reader(file):
            if not row:
                continue

            hallmark_name = row[0]
            genes = {gene for gene in row[2:] if gene != ""}  # row[1] is the url
            hallmarks[hallmark_name] = genes

    all_genes = set().union(*hallmarks.values())
    print(f"Hallmarks: {len(hallmarks)} sets, {len(all_genes)} unique genes")
    return hallmarks


def hallmark_adjacency(hallmarks : dict[str, set[str]], genes : list[str]) -> pd.DataFrame:
    """
    Build a gene x gene matrix for REGEN edge weights:
        1 -> the two genes share at least one Hallmark
        0 -> they share none (and always 0 on the diagonal)
    Rows and columns follow the order of `genes`, so pass the fold's selected genes
    in the same order as the feature columns.
    shortcut: matches on gene symbol, so 117 Hallmark genes renamed since TCGA 2015 get no edges; upgrade to the Entrez GMT if coverage matters
    """
    genes = pd.Index(genes)

    # Step 1: membership matrix -> one row per Hallmark, one column per gene (1 = gene is in that Hallmark)
    membership = np.zeros((len(hallmarks), len(genes)))
    for row, hallmark_genes in enumerate(hallmarks.values()):
        membership[row] = genes.isin(list(hallmark_genes))

    # Step 2: shared[i, j] = number of Hallmarks that contain both gene i and gene j
    shared = membership.T @ membership

    # Step 3: we only care IF they share one, not how many; and no self-edges
    adjacency = (shared > 0).astype(float)
    np.fill_diagonal(adjacency, 0)

    # Report how useful this prior is for the current genes
    n_genes = len(genes)
    genes_with_edges = (adjacency.sum(axis=1) > 0).sum()
    density = adjacency.sum() / (n_genes * (n_genes - 1))
    print(f"Genes with >=1 Hallmark edge: {genes_with_edges}/{n_genes} ({genes_with_edges / n_genes:.0%})")
    print(f"Adjacency density: {density:.2%}")

    return pd.DataFrame(adjacency, index=genes, columns=genes)
