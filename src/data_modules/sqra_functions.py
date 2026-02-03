import numpy as np
import scipy.sparse

def grid1(xmin, xmax, xbins):

    """
    xcenters, xedges, xbins, dx = grid1(xmin, xmax, xbins)
    """

    xedges   = np.linspace(xmin, xmax, xbins) 
    dx       = xedges[1] - xedges[0]
    xcenters = xedges + 0.5 * dx
    xcenters = np.delete(xcenters,-1)
    xbins    = len(xcenters)
    
    return xcenters, xedges, xbins, dx


def grid_from_spacing(xmin, xmax, spacing, *, adjust='spacing'):
    """
    Build a 1-D uniform grid by specifying a bin width instead of a bin count.

    Parameters
    ----------
    xmin, xmax : float
        Lower and upper bounds of the interval (inclusive).
    spacing    : float
        Desired bin width (dx or dy).
    adjust     : {'spacing', 'upper'}, optional
        How to handle the case where (xmax-xmin) is not an integer multiple
        of `spacing`:

        * 'spacing' – keep xmin/xmax fixed, tweak the spacing so the bins
          fit exactly (default, safest if you need exact bounds).
        * 'upper'   – keep the spacing exactly as given, extend the upper
          limit so all bins have identical width.

    Returns
    -------
    centres : ndarray, shape (N,)
        Mid-points of each bin.
    edges   : ndarray, shape (N+1,)
        Left/right edges.
    Nbins   : int
        Number of bins.
    spacing : float
        Final spacing actually used.
    """
    length = xmax - xmin
    n_bins = int(np.ceil(length / spacing))

    if adjust == 'spacing':
        spacing = length / n_bins          # subtle tweak so edges line up
        xmax_adj = xmax                    # unchanged
    elif adjust == 'upper':
        xmax_adj = xmin + n_bins * spacing # shift just the upper limit
    else:
        raise ValueError("adjust must be 'spacing' or 'upper'")

    edges   = np.linspace(xmin, xmax_adj, n_bins + 1)
    centres = edges[:-1] + 0.5 * spacing
    return centres, edges, n_bins, spacing

"""
#old, xbins = ybins
def adjancency_matrix_sparse(nbins, nd, periodic=False):
    v = np.zeros(nbins)
    v[1] = 1
    
    if periodic:
        v[-1] = 1
        A0 = scipy.sparse.csc_matrix(scipy.linalg.circulant(v)) #.toarray()
    else:
        A0 = scipy.sparse.csc_matrix(scipy.linalg.toeplitz(v)) #.toarray()
    
    A = A0
    I2 = scipy.sparse.eye(nbins)  #np.eye(nbins)
    for _ in range(1, nd):
        I1 = scipy.sparse.eye(*A.shape) #np.eye(*A.shape)
        A =  scipy.sparse.kron(A0, I1) + scipy.sparse.kron(I2, A)
    return A
"""
import scipy.sparse as spa
import scipy.linalg as la

def adjancency_weight_matrix_sparse(nbins, periodic=False, weights=1):
    """
    Build an n‑D sparse, weighted adjacency matrix.

    Parameters
    ----------
    nbins : int or sequence[int]
        Number of bins per axis.
    periodic : bool or sequence[bool], default False
        Boundary condition per axis.
    weights : float | sequence[float | sequence[float]]
        Per‑axis coupling.  For each axis d:

        * scalar  → same weight everywhere (old behaviour);
        * array‑like → variable edge weights:
              len = nbins[d] – 1   if not periodic
              len = nbins[d]       if periodic
          w[i] is the link weight between sites i and i+1
          (and w[-1] is the wrap‑around link if periodic).

    Returns
    -------
    A : scipy.sparse.csc_matrix  of size  (∏ nbins) × (∏ nbins)
    
    Example
    -------
    # 1) 3‑D, open boundaries, variable weights only along x:
    Nx, Ny, Nz = 5, 4, 3
    Dx = [0.8, 1.0, 1.2, 0.9]           # len = Nx – 1  (open)
    Dy = Dz = 1.0                       # scalars
    A = adjancency_weight_matrix_sparse([Nx, Ny, Nz],
                                periodic=False,
                                weights=[Dx, Dy, Dz])
    
    # 2) 2‑D periodic “torus” with heterogenous weights on both circles:
    Nx, Ny = 6, 8
    Dx = np.linspace(1, 2, Nx)          # len = Nx  (periodic)
    Dy = 0.5 * np.ones(Ny)              # len = Ny  (periodic)
    A = adjacency_matrix_sparse([Nx, Ny],
                                periodic=True,
                                weights=[Dx, Dy])
    """
    # ------------------------------------------------------------------ setup
    if np.isscalar(nbins):
        nbins = [int(nbins)]
    else:
        nbins = list(map(int, nbins))
    nd = len(nbins)

    if isinstance(periodic, bool):
        periodic = [periodic] * nd
    elif len(periodic) != nd:
        raise ValueError("`periodic` must match `nbins` length")

    if np.isscalar(weights):
        weights = [float(weights)] * nd
    elif len(weights) != nd:
        raise ValueError("`weights` must match `nbins` length")

    # ---------------------------------------------------- helper: 1‑D chain
    def chain_adj(n, w_axis, is_periodic):
        """
        Return sparse (n×n) adjacency for one axis with edge‑array `w_axis`.
        """
        # --- expand | validate `w_axis` -----------------------------
        if np.isscalar(w_axis):
            edge_w = np.full(n - 1, w_axis, dtype=float)
            wrap_w = float(w_axis)
        else:
            w_axis = np.asarray(w_axis, dtype=float)
            expected = n if is_periodic else n - 1
            if w_axis.size != expected:
                raise ValueError(
                    f"weight array of length {w_axis.size} does not "
                    f"match expected {expected} for axis of size {n}"
                )
            edge_w = w_axis[: n - 1]
            wrap_w = w_axis[-1] if is_periodic else None

        # --- build sparse -------------------------------
        A = spa.diags([edge_w, edge_w], offsets=[-1, 1], format="csc")
        if is_periodic and n > 2:
            A[0,  n - 1] = wrap_w
            A[n - 1, 0]  = wrap_w
        return A

    # ------------------------------------- Kronecker‑sum over dimensions
    A_total = chain_adj(nbins[0], weights[0], periodic[0])
    for n, w_ax, is_per in zip(nbins[1:], weights[1:], periodic[1:]):
        A_d   = chain_adj(n, w_ax, is_per)
        I_old = spa.eye(A_total.shape[0], format="csc")
        I_new = spa.eye(A_d.shape[0],   format="csc")
        A_total = spa.kron(A_d,  I_old, format="csc") + \
                  spa.kron(I_new, A_total, format="csc")

    return A_total
    
def adjancency_matrix_sparse(nbins, periodic=False):
    """
    Build the sparse adjacency matrix of an n-dimensional tensor grid
    using a Kronecker‐sum of 1-D nearest-neighbour chains.

    Parameters
    ----------
    nbins : int or sequence of ints
        * If int   → same number of bins along every dimension.
        * If list  → nbins[d] is the number of bins along dimension d
                     (e.g. [xbins, ybins, zbins, …]).
    periodic : bool or sequence of bools, optional
        * If bool  → same periodicity for every dimension.
        * If list  → periodic[d] applies to dimension d.  True ⇒ wrap-around
                     neighbours (a ring); False ⇒ open ends (a line).

    Returns
    -------
    A : scipy.sparse.csc_matrix
        (N × N) adjacency matrix where N = ∏ nbins.



    """
    # --- normalise inputs to sequences -------------------------------------
    if np.isscalar(nbins):
        nbins = [int(nbins)]
    else:
        nbins = list(map(int, nbins))

    nd = len(nbins)

    if isinstance(periodic, bool):
        periodic = [periodic] * nd
    elif len(periodic) != nd:
        raise ValueError("`periodic` must have the same length as `nbins`")

    # --- helper: 1-D adjacency ---------------------------------------------
    def chain_adj(n, is_periodic):
        """
        Sparse (n × n) adjacency of a 1-D chain/ring.
        """
        # main ±1 diagonals
        A = spa.diags([np.ones(n-1), np.ones(n-1)], offsets=[-1, 1], format="csc")
        if is_periodic and n > 2:            # connect the ends
            A[0, n-1] = 1
            A[n-1, 0] = 1
        return A

    # --- Kronecker-sum over dimensions -------------------------------------
    A_total = chain_adj(nbins[0], periodic[0])
    for n, is_per in zip(nbins[1:], periodic[1:]):
        A_d   = chain_adj(n, is_per)
        I_old = spa.eye(A_total.shape[0], format="csc")
        I_new = spa.eye(A_d.shape[0], format="csc")
        # Kron-sum:   A ⊗ I + I ⊗ B
        A_total = spa.kron(A_d, I_old, format="csc") + spa.kron(I_new, A_total, format="csc")

    return A_total


def kronecker_sum(a, b):
    A = np.diag(a)  # Convert 1D array to diagonal matrix
    B = np.diag(b)
    
    IA = np.eye(len(a))  # Identity matrix of size len(a)
    IB = np.eye(len(b))  # Identity matrix of size len(b)
    
    return np.kron(A, IB) + np.kron(IA, B)
