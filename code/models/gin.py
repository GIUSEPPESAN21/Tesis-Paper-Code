"""Graph Isomorphism Network encoder over the JSSP disjunctive graph.

Why a hand-rolled GIN instead of ``torch_geometric.nn.GINConv``
---------------------------------------------------------------
The disjunctive graph of a classic JSSP benchmark is small (36 nodes for ``ft06``, 225
for ``la40``) but *dense in the machine-sharing relation*: the machine clique alone has
``O(num_jobs^2)`` edges per machine.  At that size a dense ``A @ H`` matrix product is
both simpler and faster than gather/scatter over an edge index, and it removes
``torch-geometric`` --- a dependency whose wheels are pinned to an exact
torch+CUDA build and which is the single most common reason a published DRL scheduling
repo fails to install.  For a thesis that has to be reproducible on a reviewer's machine
three years from now, that tradeoff is worth ~40 lines of code.  The layer below is
mathematically the standard GIN update; swapping in PyG later is a drop-in change
confined to this module.

The update
----------
Xu et al. (2019) define GIN as

.. math::
    h_v^{(k)} = \\mathrm{MLP}^{(k)}\\Big( (1 + \\epsilon^{(k)}) \\cdot h_v^{(k-1)}
                 + \\sum_{u \\in \\mathcal{N}(v)} h_u^{(k-1)} \\Big)

with **sum** aggregation, which is what makes GIN as discriminative as the
Weisfeiler-Lehman test --- mean and max aggregation cannot distinguish a machine clique
of 5 identical operations from one of 10, and machine load is exactly the kind of signal
a scheduler needs.

:class:`RelationalGINLayer` extends this to the two *distinct* relations present in a
disjunctive graph, aggregating each with its own weight before the MLP::

    m_v = W_pred @ sum_{u -> v} h_u  +  W_succ @ sum_{v -> u} h_u  +  W_mach @ sum_{u ~ v} h_u

Keeping precedence-predecessor, precedence-successor and machine-sharing separate matters
because they mean opposite things: a predecessor constrains when ``v`` can start, a
successor is what ``v`` unblocks, and a machine neighbour is a *competitor*.  Collapsing
them into one symmetric adjacency (the naive encoding) throws that away.

Heterogeneous extension (Week 18)
----------------------------------
:class:`HeteroRelationalGINLayer` and :class:`HeteroGINEncoder` add *machine nodes* as a
second node type and the operation<->machine *eligibility* relation as a fourth (and,
from the machine's side, fifth) relation, per ``GNN_PPO_Architecture_Spec_v2.md`` §2.
They are additive: :class:`RelationalGINLayer` / :class:`GINEncoder` keep their exact
signatures and behaviour, and the homogeneous path remains the pipeline's ablation
configuration.

Why not ``torch_geometric.nn.HeteroConv``
------------------------------------------
For the same reason this module does not use ``GINConv``, only more so.  A
``HeteroConv`` is a dict of per-relation convolutions over an ``edge_index`` per
relation --- which is exactly what the five ``nn.Linear`` + ``bmm`` terms below compute,
with the relation-specific weights ``HeteroConv`` exists to provide, but without adding
``torch-geometric`` (wheels pinned to an exact torch+CUDA build; the single most common
reason a published DRL scheduling repo fails to install three years later).  The
eligibility relation in particular is *dense by construction* --- ``adj_eligibility`` is
an ``(N_ope, N_mach)`` matrix with one nonzero per row under fixed routing and a handful
per row under FJSP --- so a gather/scatter over an edge index would be strictly slower
than the ``bmm`` here at these instance sizes (36-225 operation nodes, 4-15 machine
nodes).  The update below is mathematically the standard heterogeneous message-passing
form; swapping in PyG later remains a drop-in change confined to this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(
    in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, use_layer_norm: bool = True
) -> nn.Sequential:
    """Build the MLP used inside a GIN layer.

    Args:
        in_dim: Input width.
        hidden_dim: Hidden width.
        out_dim: Output width.
        num_layers: Total number of ``Linear`` layers (``>= 1``).
        use_layer_norm: Insert ``LayerNorm`` after each hidden layer.  Preferred over
            ``BatchNorm`` here because rollouts are collected one graph at a time and
            batch statistics over a single graph are meaningless.

    Returns:
        The MLP module.

    Raises:
        ValueError: If ``num_layers < 1``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class RelationalGINLayer(nn.Module):
    """One GIN message-passing step over precedence and machine-sharing relations.

    Args:
        in_dim: Input node-feature width.
        out_dim: Output node-embedding width.
        hidden_dim: Hidden width of the update MLP.
        mlp_layers: Depth of the update MLP.
        learn_eps: Make the GIN ``eps`` a learnable parameter instead of a fixed ``0``.
        use_machine_edges: Include the machine-sharing relation.  Setting ``False`` gives
            the "precedence only" ablation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        mlp_layers: int = 2,
        learn_eps: bool = True,
        use_machine_edges: bool = True,
    ) -> None:
        super().__init__()
        self.use_machine_edges = use_machine_edges
        self.eps = nn.Parameter(torch.zeros(1)) if learn_eps else None

        self.lin_pred = nn.Linear(in_dim, in_dim, bias=False)
        self.lin_succ = nn.Linear(in_dim, in_dim, bias=False)
        self.lin_machine = (
            nn.Linear(in_dim, in_dim, bias=False) if use_machine_edges else None
        )
        self.mlp = build_mlp(in_dim, hidden_dim, out_dim, num_layers=mlp_layers)

    def forward(
        self, x: torch.Tensor, adj_precedence: torch.Tensor, adj_machine: torch.Tensor
    ) -> torch.Tensor:
        """Apply one relational GIN update.

        Args:
            x: Node features, shape ``(B, N, in_dim)``.
            adj_precedence: Directed job-chain adjacency, shape ``(B, N, N)``, where
                ``[b, u, v] = 1`` means ``v`` follows ``u``.
            adj_machine: Symmetric machine-sharing adjacency, shape ``(B, N, N)``.

        Returns:
            Updated node embeddings, shape ``(B, N, out_dim)``.
        """
        # sum over predecessors: A^T @ X gathers, for each v, the rows u with u -> v.
        pred_msg = self.lin_pred(torch.bmm(adj_precedence.transpose(1, 2), x))
        succ_msg = self.lin_succ(torch.bmm(adj_precedence, x))
        aggregated = pred_msg + succ_msg
        if self.lin_machine is not None:
            aggregated = aggregated + self.lin_machine(torch.bmm(adj_machine, x))

        eps = self.eps if self.eps is not None else torch.zeros(1, device=x.device)
        return self.mlp((1.0 + eps) * x + aggregated)


class GINEncoder(nn.Module):
    """Stack of :class:`RelationalGINLayer` producing node and graph embeddings.

    Args:
        in_dim: Number of raw node features (``envs.jssp_env.NUM_NODE_FEATURES``).
        hidden_dim: Width of every hidden and output embedding.
        num_layers: Number of message-passing rounds.  Each round widens a node's
            receptive field by one hop; 2-3 is the usual choice for JSSP graphs, where
            the machine cliques already make the graph low-diameter and more layers
            mainly cause over-smoothing.
        mlp_layers: Depth of the MLP inside each GIN layer.
        use_machine_edges: Pass through to :class:`RelationalGINLayer`.
        pool: Graph-level readout, ``"mean"`` or ``"sum"``.  ``"mean"`` is the default
            because the critic must compare states across instances of different sizes,
            and a sum readout scales with ``N``.

    Attributes:
        out_dim: Width of the returned node embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_layers: int = 2,
        use_machine_edges: bool = True,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if pool not in ("mean", "sum"):
            raise ValueError(f"pool must be 'mean' or 'sum', got {pool!r}")

        self.pool = pool
        self.out_dim = hidden_dim
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            RelationalGINLayer(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                hidden_dim=hidden_dim,
                mlp_layers=mlp_layers,
                use_machine_edges=use_machine_edges,
            )
            for i in range(num_layers)
        )

    def forward(
        self, x: torch.Tensor, adj_precedence: torch.Tensor, adj_machine: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of disjunctive graphs.

        Args:
            x: Node features, shape ``(B, N, in_dim)``.
            adj_precedence: Shape ``(B, N, N)``.
            adj_machine: Shape ``(B, N, N)``.

        Returns:
            ``(node_embeddings, graph_embedding)`` of shapes ``(B, N, hidden_dim)`` and
            ``(B, hidden_dim)``.
        """
        h = x
        for layer in self.layers:
            h = layer(h, adj_precedence, adj_machine)
        graph = h.mean(dim=1) if self.pool == "mean" else h.sum(dim=1)
        return h, graph


class HeteroRelationalGINLayer(nn.Module):
    """One heterogeneous GIN update over operation nodes *and* machine nodes.

    Five relation-specific weight matrices, four updating operation nodes and one
    updating machine nodes (``GNN_PPO_Architecture_Spec_v2.md`` §2.2):

    ================= ============================ ==============================
    weight            relation                     message
    ================= ============================ ==============================
    ``lin_pred``      operation <- job predecessor  ``A_prec^T @ X_ope``
    ``lin_succ``      operation <- job successor    ``A_prec   @ X_ope``
    ``lin_machine``   operation <- machine rival    ``A_mach   @ X_ope``
    ``lin_elig``      operation <- eligible machine ``A_elig   @ X_mach``
    ``lin_cand``      machine   <- candidate op     ``A_elig^T @ X_ope``
    ================= ============================ ==============================

    Every relation uses **sum** aggregation followed by its own MLP, i.e. GIN, rather
    than attention.  The justification is reproduced from the architecture spec §2.1:
    L2D (the closest precedent in objective) uses sum pooling with no attention
    anywhere, and Song et al. (the closest precedent in graph heterogeneity) use
    attention only on the operation->machine edge and a plain sum-then-MLP aggregator
    elsewhere.  Sum aggregation is also what makes GIN as discriminative as 1-WL (Xu et
    al. 2019, Theorem 3), and distinguishing a machine clique of 5 identical operations
    from one of 10 is exactly the signal a scheduler needs.

    Args:
        in_dim_ope: Input width of operation-node features.
        in_dim_mach: Input width of machine-node features.
        out_dim_ope: Output width of operation-node embeddings.
        out_dim_mach: Output width of machine-node embeddings.
        hidden_dim: Hidden width of both update MLPs.
        mlp_layers: Depth of each update MLP.
        learn_eps: Make the two GIN ``eps`` terms learnable instead of fixed ``0``.
        use_machine_edges: Include the operation-machine-rival relation
            (``lin_machine``).  ``False`` gives the "no disjunctive edges" ablation,
            mirroring :class:`RelationalGINLayer`'s flag of the same name.
    """

    def __init__(
        self,
        in_dim_ope: int,
        in_dim_mach: int,
        out_dim_ope: int,
        out_dim_mach: int,
        hidden_dim: int = 64,
        mlp_layers: int = 2,
        learn_eps: bool = True,
        use_machine_edges: bool = True,
    ) -> None:
        super().__init__()
        self.use_machine_edges = use_machine_edges
        self.eps_ope = nn.Parameter(torch.zeros(1)) if learn_eps else None
        self.eps_mach = nn.Parameter(torch.zeros(1)) if learn_eps else None

        # Operation-side: same weight pattern as RelationalGINLayer, plus lin_elig.
        self.lin_pred = nn.Linear(in_dim_ope, in_dim_ope, bias=False)
        self.lin_succ = nn.Linear(in_dim_ope, in_dim_ope, bias=False)
        self.lin_machine = (
            nn.Linear(in_dim_ope, in_dim_ope, bias=False) if use_machine_edges else None
        )
        self.lin_elig = nn.Linear(in_dim_mach, in_dim_ope, bias=False)
        # Machine-side: one relation, machine <- its candidate operations.
        self.lin_cand = nn.Linear(in_dim_ope, in_dim_mach, bias=False)

        self.mlp_ope = build_mlp(in_dim_ope, hidden_dim, out_dim_ope, num_layers=mlp_layers)
        self.mlp_mach = build_mlp(in_dim_mach, hidden_dim, out_dim_mach, num_layers=mlp_layers)

    def forward(
        self,
        x_ope: torch.Tensor,
        x_mach: torch.Tensor,
        adj_precedence: torch.Tensor,
        adj_machine: torch.Tensor,
        adj_eligibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one heterogeneous GIN update.

        Args:
            x_ope: Operation-node features, shape ``(B, N_ope, in_dim_ope)``.
            x_mach: Machine-node features, shape ``(B, N_mach, in_dim_mach)``.
            adj_precedence: Directed job-chain adjacency, ``(B, N_ope, N_ope)``, where
                ``[b, u, v] = 1`` means ``v`` follows ``u``.
            adj_machine: Symmetric machine-sharing adjacency, ``(B, N_ope, N_ope)``.
            adj_eligibility: Operation-machine incidence, ``(B, N_ope, N_mach)``.  One
                nonzero per row under fixed routing (JSSP); several under FJSP.

        Returns:
            ``(h_ope, h_mach)`` of shapes ``(B, N_ope, out_dim_ope)`` and
            ``(B, N_mach, out_dim_mach)``.
        """
        ope_msg = (
            self.lin_pred(torch.bmm(adj_precedence.transpose(1, 2), x_ope))
            + self.lin_succ(torch.bmm(adj_precedence, x_ope))
            + self.lin_elig(torch.bmm(adj_eligibility, x_mach))
        )
        if self.lin_machine is not None:
            ope_msg = ope_msg + self.lin_machine(torch.bmm(adj_machine, x_ope))
        mach_msg = self.lin_cand(torch.bmm(adj_eligibility.transpose(1, 2), x_ope))

        zero = torch.zeros(1, device=x_ope.device, dtype=x_ope.dtype)
        eps_ope = self.eps_ope if self.eps_ope is not None else zero
        eps_mach = self.eps_mach if self.eps_mach is not None else zero

        h_ope = self.mlp_ope((1.0 + eps_ope) * x_ope + ope_msg)
        h_mach = self.mlp_mach((1.0 + eps_mach) * x_mach + mach_msg)
        return h_ope, h_mach


class HeteroRelationalGATHybridLayer(nn.Module):
    """:class:`HeteroRelationalGINLayer` with one relation swapped for attention.

    ``AblationPlan.md`` Axis 1b / Song et al. (2023)-style: every operation-side relation
    (predecessor, successor, machine-rival, eligible-machine-into-operation) stays
    sum+MLP, unchanged from :class:`HeteroRelationalGINLayer`. Only the machine-update
    term (machine <- its candidate operations) is replaced with single-head GAT-style
    attention, so a machine node can weight its eligible operations unevenly instead of
    summing them uniformly. ``num_heads=1`` deliberately, matching Song et al.'s actual
    configuration (not their parameterised capability), so the comparison isolates
    "attention vs. sum aggregation" rather than "one head vs. many."

    Attention (per machine ``m``, over its eligible operations ``i``, masked by
    ``adj_eligibility``)::

        e[m, i]     = LeakyReLU(a_dst . W_dst(h_m) + a_src . W_src(h_i))
        alpha[m, :] = softmax_i(e[m, :] masked to eligible i)
        msg[m]      = sum_i alpha[m, i] * W_src(h_i)

    ``W_src`` doubles as both the attention key/message projection, the standard GAT
    reuse of ``Wh``. Ineligible ``(op, machine)`` pairs are masked to ``-1e9`` before the
    softmax so they receive zero weight; every machine is assumed to have at least one
    eligible operation (guaranteed by both ``JSSPEnv`` and ``FJSPEnv``'s instance data).

    Args: see :class:`HeteroRelationalGINLayer` -- identical signature, plus
        ``negative_slope`` for the ``LeakyReLU`` (GAT's usual ``0.2``).
    """

    def __init__(
        self,
        in_dim_ope: int,
        in_dim_mach: int,
        out_dim_ope: int,
        out_dim_mach: int,
        hidden_dim: int = 64,
        mlp_layers: int = 2,
        learn_eps: bool = True,
        use_machine_edges: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_machine_edges = use_machine_edges
        self.eps_ope = nn.Parameter(torch.zeros(1)) if learn_eps else None
        self.eps_mach = nn.Parameter(torch.zeros(1)) if learn_eps else None

        # Operation-side: unchanged from HeteroRelationalGINLayer (sum+MLP).
        self.lin_pred = nn.Linear(in_dim_ope, in_dim_ope, bias=False)
        self.lin_succ = nn.Linear(in_dim_ope, in_dim_ope, bias=False)
        self.lin_machine = (
            nn.Linear(in_dim_ope, in_dim_ope, bias=False) if use_machine_edges else None
        )
        self.lin_elig = nn.Linear(in_dim_mach, in_dim_ope, bias=False)

        # Machine-side: single-head GAT attention over candidate operations.
        self.attn_src = nn.Linear(in_dim_ope, in_dim_mach, bias=False)
        self.attn_dst = nn.Linear(in_dim_mach, in_dim_mach, bias=False)
        self.attn_a_src = nn.Parameter(torch.empty(in_dim_mach))
        self.attn_a_dst = nn.Parameter(torch.empty(in_dim_mach))
        nn.init.xavier_uniform_(self.attn_src.weight)
        nn.init.xavier_uniform_(self.attn_dst.weight)
        nn.init.uniform_(self.attn_a_src, -0.1, 0.1)
        nn.init.uniform_(self.attn_a_dst, -0.1, 0.1)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.mlp_ope = build_mlp(in_dim_ope, hidden_dim, out_dim_ope, num_layers=mlp_layers)
        self.mlp_mach = build_mlp(in_dim_mach, hidden_dim, out_dim_mach, num_layers=mlp_layers)

    def forward(
        self,
        x_ope: torch.Tensor,
        x_mach: torch.Tensor,
        adj_precedence: torch.Tensor,
        adj_machine: torch.Tensor,
        adj_eligibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one heterogeneous GIN-operation-side / GAT-machine-side update.

        Args: identical to :meth:`HeteroRelationalGINLayer.forward`.

        Returns:
            ``(h_ope, h_mach)``, same shapes as :meth:`HeteroRelationalGINLayer.forward`.
        """
        ope_msg = (
            self.lin_pred(torch.bmm(adj_precedence.transpose(1, 2), x_ope))
            + self.lin_succ(torch.bmm(adj_precedence, x_ope))
            + self.lin_elig(torch.bmm(adj_eligibility, x_mach))
        )
        if self.lin_machine is not None:
            ope_msg = ope_msg + self.lin_machine(torch.bmm(adj_machine, x_ope))

        src_proj = self.attn_src(x_ope)  # (B, N_ope, in_dim_mach) -- doubles as message value
        dst_proj = self.attn_dst(x_mach)  # (B, N_mach, in_dim_mach)
        score_src = (src_proj * self.attn_a_src).sum(-1)  # (B, N_ope)
        score_dst = (dst_proj * self.attn_a_dst).sum(-1)  # (B, N_mach)
        e = self.leaky_relu(score_dst.unsqueeze(2) + score_src.unsqueeze(1))  # (B, N_mach, N_ope)
        elig_mask = adj_eligibility.transpose(1, 2)  # (B, N_mach, N_ope), 1 where i eligible for m
        e = e.masked_fill(elig_mask < 0.5, -1e9)
        alpha = torch.softmax(e, dim=-1)
        mach_msg = torch.bmm(alpha, src_proj)  # (B, N_mach, in_dim_mach)

        zero = torch.zeros(1, device=x_ope.device, dtype=x_ope.dtype)
        eps_ope = self.eps_ope if self.eps_ope is not None else zero
        eps_mach = self.eps_mach if self.eps_mach is not None else zero

        h_ope = self.mlp_ope((1.0 + eps_ope) * x_ope + ope_msg)
        h_mach = self.mlp_mach((1.0 + eps_mach) * x_mach + mach_msg)
        return h_ope, h_mach


class HeteroGINEncoder(nn.Module):
    """Stack of :class:`HeteroRelationalGINLayer`; the heterogeneous twin of :class:`GINEncoder`.

    Args:
        in_dim_ope: Number of raw operation-node features
            (``envs.jssp_env.NUM_NODE_FEATURES``).
        in_dim_mach: Number of raw machine-node features
            (``envs.jssp_env.NUM_MACHINE_FEATURES``).
        hidden_dim: Width of every hidden and output embedding, both node types.
        num_layers: Number of message-passing rounds.
        mlp_layers: Depth of the MLP inside each layer.
        use_machine_edges: Passed through to :class:`HeteroRelationalGINLayer`.
        pool: Graph-level readout, ``"mean"`` or ``"sum"``.  ``"mean"`` by default,
            extending :class:`GINEncoder`'s rationale (the critic must compare states
            across instance sizes, and a sum readout scales with ``N``) to the second
            node type.
        encoder_type: ``"gin"`` (default, sum+MLP everywhere) or ``"gat_hybrid"``
            (``AblationPlan.md`` Axis 1b: :class:`HeteroRelationalGATHybridLayer` --
            attention on the machine-update term only, sum+MLP elsewhere).

    Attributes:
        out_dim: Width of the returned node embeddings (both node types).
        graph_dim: Width of the returned graph embedding, ``2 * hidden_dim`` --- the
            concatenation of the pooled operation and pooled machine embeddings.  The
            critic therefore sees a "how much work is left" signal *and* a "how
            loaded/available are the machines" signal; the homogeneous encoder could
            only supply the first.

    Raises:
        ValueError: If ``encoder_type`` is neither ``"gin"`` nor ``"gat_hybrid"``.
    """

    def __init__(
        self,
        in_dim_ope: int = 8,
        in_dim_mach: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_layers: int = 2,
        use_machine_edges: bool = True,
        pool: str = "mean",
        encoder_type: str = "gin",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if pool not in ("mean", "sum"):
            raise ValueError(f"pool must be 'mean' or 'sum', got {pool!r}")
        if encoder_type not in ("gin", "gat_hybrid"):
            raise ValueError(f"encoder_type must be 'gin' or 'gat_hybrid', got {encoder_type!r}")

        self.pool = pool
        self.out_dim = hidden_dim
        self.graph_dim = 2 * hidden_dim
        layer_cls = (
            HeteroRelationalGATHybridLayer if encoder_type == "gat_hybrid" else HeteroRelationalGINLayer
        )
        dims_ope = [in_dim_ope] + [hidden_dim] * num_layers
        dims_mach = [in_dim_mach] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            layer_cls(
                in_dim_ope=dims_ope[i],
                in_dim_mach=dims_mach[i],
                out_dim_ope=dims_ope[i + 1],
                out_dim_mach=dims_mach[i + 1],
                hidden_dim=hidden_dim,
                mlp_layers=mlp_layers,
                use_machine_edges=use_machine_edges,
            )
            for i in range(num_layers)
        )

    def forward(
        self,
        x_ope: torch.Tensor,
        x_mach: torch.Tensor,
        adj_precedence: torch.Tensor,
        adj_machine: torch.Tensor,
        adj_eligibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of heterogeneous disjunctive graphs.

        Args:
            x_ope: Operation-node features, ``(B, N_ope, in_dim_ope)``.
            x_mach: Machine-node features, ``(B, N_mach, in_dim_mach)``.
            adj_precedence: ``(B, N_ope, N_ope)``.
            adj_machine: ``(B, N_ope, N_ope)``.
            adj_eligibility: ``(B, N_ope, N_mach)``.

        Returns:
            ``(node_embeddings_ope, node_embeddings_mach, graph_embedding)`` of shapes
            ``(B, N_ope, hidden_dim)``, ``(B, N_mach, hidden_dim)`` and
            ``(B, 2 * hidden_dim)``.
        """
        h_ope, h_mach = x_ope, x_mach
        for layer in self.layers:
            h_ope, h_mach = layer(h_ope, h_mach, adj_precedence, adj_machine, adj_eligibility)

        if self.pool == "mean":
            pooled_ope, pooled_mach = h_ope.mean(dim=1), h_mach.mean(dim=1)
        else:
            pooled_ope, pooled_mach = h_ope.sum(dim=1), h_mach.sum(dim=1)
        graph = torch.cat([pooled_ope, pooled_mach], dim=-1)
        return h_ope, h_mach, graph
