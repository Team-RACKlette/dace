# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Fusion of a stencil producer into its consumer through a tile buffer.

:class:`TileStencilFusion` takes the pattern of :class:`~dace.transformation.dataflow.OTFStencilFusion` -- a
producer map writing a transient that a consumer map reads at constant offsets -- and fuses it without recomputing
the producer per read: the consumer map is tiled, the producer runs once per tile over the tile's range extended by
the stencil's halo into a tile buffer (shared memory on a GPU), and the consumer's iterations read their window of
the buffer. The recomputation is the halo's share of the tile ((T + halo) / T) instead of one producer run per read.

Every output of the producer the consumer reads gets a tile buffer from the same producer run, so a body computing
several outputs (the left and the right value at a cell, say) runs once per cell of the tile. The consumer keeps a
per-iteration copy of its window, as in the recomputing fusion: an in-place update of the array by the consumer
stays in that copy, so the transformation applies to such a consumer only when nothing reads the array after it.

The maps inside the tile take ``schedule`` (a GPU thread-block schedule by default: the CUDA code generator takes the
block size from them and synchronises the block between the producer and the consumer) and the buffers ``storage``.
"""

import copy
from typing import Dict, List, Optional, Tuple

import sympy

from dace import dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import EnumProperty, ShapeProperty, make_properties
from dace.sdfg import SDFG, SDFGState, dealias
from dace.sdfg.propagation import propagate_memlets_scope
from dace.sdfg import nodes as nd
from dace.transformation.dataflow.otf_stencil_fusion import OTFStencilFusion
from dace.transformation.dataflow.tiling import MapTiling


def _exact(expr) -> symbolic.SymbolicType:
    """A range end as a sympy expression (the exact one of a strip-mined range's pair)."""
    if isinstance(expr, symbolic.SymExpr):
        return expr.expr
    return symbolic.pystr_to_symbolic(expr)


@make_properties
class TileStencilFusion(OTFStencilFusion):
    """Fuse a producer map into a consumer map through a per-tile buffer of its outputs (see module docs)."""

    tile_sizes = ShapeProperty(
        dtype=tuple, default=(), category="Parameters",
        desc="Tile size per consumer map parameter (empty: a block of 128 threads with the halo: 32 x 4 when the "
             "stencil runs along the last parameter, else 16 along it and 8 along the stencil's; 1 elsewhere)")
    schedule = EnumProperty(
        dtype=dtypes.ScheduleType, default=dtypes.ScheduleType.GPU_ThreadBlock, category="Scheduling",
        desc="Schedule of the maps inside a tile: the producer over the tile and the consumer's inner map")
    storage = EnumProperty(
        dtype=dtypes.StorageType, default=dtypes.StorageType.GPU_Shared, category="Memory",
        desc="Storage of the tile buffers")

    @staticmethod
    def annotates_memlets() -> bool:
        return True  # every new edge is added with its propagated memlet

    @staticmethod
    def copies_scope() -> bool:
        return True  # the producer's scope is copied whole: in-scope nodes feeding the body come along

    # ------------------------------------------------------------------------------------------------------------------
    # Analysis

    @staticmethod
    def _producer_outputs(graph: SDFGState, first_map_exit: nd.MapExit) -> Dict[str, nd.AccessNode]:
        """The access nodes the producer writes, by array."""
        return {graph.memlet_path(e)[-1].dst.data: graph.memlet_path(e)[-1].dst for e in graph.out_edges(first_map_exit)}

    def _stencil(self, graph: SDFGState, sdfg: SDFG) -> Optional[Tuple[Dict[str, List], Dict[str, str],
                                                                          Dict[str, Tuple[int, int]], Dict[str, Tuple]]]:
        """The consumer's reads of the producer's outputs: per array its read edges, per producer parameter the
        consumer parameter it follows with the smallest and largest constant shift over every read, and the
        producer parameters the consumer reads over their whole range instead (a variable dimension the producer
        runs on threads, read as a slab: the tile holds every variable). ``None`` when a read does not take that
        form (a sub-range along a producer parameter, a shift that is not a constant)."""
        first_map_entry = graph.entry_node(self.first_map_exit)
        centry: nd.MapEntry = self.second_map_entry
        writes = self._body_writes(graph, self.first_map_exit)
        outputs = self._producer_outputs(graph, self.first_map_exit)
        # A producer that reads one of its own outputs (an in-place correction) would read the array through the
        # connector the tile copy rebinds to the tile buffer.
        if any(e.data.data in outputs for e in graph.in_edges(first_map_entry)):
            return self._refuse("the producer reads one of its own outputs")
        cparams = set(centry.map.params)
        prange = {p: r for p, r in zip(first_map_entry.map.params, first_map_entry.map.range)}
        reads: Dict[str, List] = {}
        follows: Dict[str, str] = {}
        shifts: Dict[str, Tuple[int, int]] = {}
        whole: Dict[str, Tuple] = {}
        for e in graph.out_edges(centry):
            name = e.data.data
            if name not in outputs or name not in writes:
                continue
            feed = next((x for x in graph.in_edges(centry) if x.data.data == name), None)
            if feed is None or feed.src is not outputs[name]:
                return self._refuse(f"the consumer reads {name}, but not the producer's copy of it")
            solved = self.solve(first_map_entry.map.params, writes[name][2], centry.map.params, e.data.subset)
            if solved is None:
                return self._refuse(f"the read {name}[{e.data.subset}] does not solve against the write "
                                    f"{writes[name][2]}")
            matched, inner = solved
            for p, (lo, hi) in inner.items():
                b, en, _st = prange[p]
                if symbolic.simplify(lo - b) != 0 or symbolic.simplify(hi - en) != 0 or p in follows:
                    return self._refuse(f"{name} reads a part of the producer's range along {p} ({lo}:{hi} of "
                                        f"{b}:{en}), or another read follows a consumer parameter along it")
                whole[p] = (b, en)
            for p, expr in matched.items():
                expr = symbolic.pystr_to_symbolic(expr)
                qs = [s for s in expr.free_symbols if str(s) in cparams]
                if len(qs) != 1:
                    return self._refuse(f"{name}: the producer parameter {p} is read at {expr}, not along one "
                                        f"consumer parameter")
                q = str(qs[0])
                shift = symbolic.simplify(expr - qs[0])  # the symbol as the expression spells it (its assumptions)
                if not shift.is_Integer or follows.setdefault(p, q) != q or p in whole:
                    return self._refuse(f"{name}: the producer parameter {p} is read at {expr}: the shift is not a "
                                        f"constant, or {p} follows another parameter elsewhere")
                lo, hi = shifts.get(p, (int(shift), int(shift)))
                shifts[p] = (min(lo, int(shift)), max(hi, int(shift)))
            reads.setdefault(name, []).append(e)
        if not reads or set(follows) | set(whole) != set(first_map_entry.map.params):
            return self._refuse("no read of the producer's outputs, or a producer parameter no read follows: "
                                f"reads {sorted(reads)}, follows {follows}, whole {sorted(whole)}")
        # A consumer that updates one of those arrays in place (a normalisation of the values it reads, say) reads the
        # tile and writes its window of it: the update stays local to the iteration, as the recomputing fusion keeps
        # it, since the array is dead after the consumer (a precondition of the pair).
        self.refusal = ""
        return reads, follows, shifts, whole

    def _refuse(self, why: str) -> None:
        """Record why ``_stencil`` declines (``refusal``, for diagnostics) and return ``None``."""
        self.refusal = why
        return None

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        return self._stencil(graph, sdfg) is not None

    # ------------------------------------------------------------------------------------------------------------------
    # Application

    def _choose_tile_sizes(self, centry: nd.MapEntry, follows: Dict[str, str], shifts: Dict[str, Tuple[int, int]],
                           variables: int = 1) -> List:
        """``variables``: the product of the producer's whole-range (variable) dimensions, which the producer's tile
        map runs on threads as well: the tile shrinks so that the block stays within 1024 threads."""
        params = centry.map.params
        if self.tile_sizes:
            sizes = list(self.tile_sizes) + [self.tile_sizes[-1]] * (len(params) - len(self.tile_sizes))
            return sizes[:len(params)]
        halo = {follows[p]: hi - lo for p, (lo, hi) in shifts.items() if hi > lo}
        # A block of 128 threads, halo included, so that its warps are whole and aligned with the rows: 32 x 4 when
        # the stencil runs along the last parameter (a tile of 31 + 1), else 16 along it and 8 (7 + 1) along the
        # stencil's.
        sizes = [8 - halo[q] if q in halo else 1 for q in params[:-1]]
        sizes.append((32 if not halo.keys() - {params[-1]} else 16) - halo.get(params[-1], 0))
        if len(params) > 1 and all(s == 1 for s in sizes[:-1]):
            sizes[-2] = 4
        sizes = [max(s, 1) for s in sizes]

        def block_threads() -> int:
            threads = max(1, variables)
            for q, size in zip(params, sizes):
                threads *= size + halo.get(q, 0)
            return threads

        while block_threads() > 1024:  # halve the widest tile dimension but the last (the warps), then the last
            d = max(range(len(sizes) - 1), key=lambda k: sizes[k], default=None)
            if d is not None and sizes[d] > 1:
                sizes[d] = max(1, sizes[d] // 2)
            elif sizes[-1] > 1:
                sizes[-1] = max(1, sizes[-1] // 2)
            else:
                break
        return sizes

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        first_map_exit: nd.MapExit = self.first_map_exit
        first_map_entry: nd.MapEntry = graph.entry_node(first_map_exit)
        centry: nd.MapEntry = self.second_map_entry
        cexit: nd.MapExit = graph.exit_node(centry)
        writes = self._body_writes(graph, first_map_exit)
        outputs = self._producer_outputs(graph, first_map_exit)
        body_node = writes[self.array.data][0]
        reads, follows, shifts, whole = self._stencil(graph, sdfg)
        pparams = list(first_map_entry.map.params)
        cparams = list(centry.map.params)
        cranges = {q: r for q, r in zip(cparams, centry.map.range)}

        # 1. Tile the consumer: an outer map over tiles, the consumer's own map over a tile.
        variables = 1
        for b, en in whole.values():
            extent = symbolic.simplify(_exact(en) - _exact(b) + 1)
            variables *= int(extent) if extent.is_Integer else 8
        sizes = self._choose_tile_sizes(centry, follows, shifts, variables)
        MapTiling.apply_to(sdfg, map_entry=centry, options=dict(tile_sizes=tuple(sizes), tile_trivial=True,
                                                               divides_evenly=False, prefix="tile"), save=False)
        outer: nd.MapEntry = graph.scope_dict()[centry]
        outer_exit: nd.MapExit = graph.exit_node(outer)
        # The strip-mined range ends are exact / approximate pairs whose approximate half (the unclipped tile end)
        # memlet propagation takes, over-approximating the last tile past the arrays; the exact expressions (a Min)
        # propagate within bounds and the CUDA code generator still reads the constant tile size off them.
        centry.map.range = subsets.Range([(_exact(b), _exact(e), _exact(st)) for b, e, st in centry.map.range])
        inner_range = {q: r for q, r in zip(centry.map.params, centry.map.range)}
        tile_syms = {str(s) for s in outer.map.range.free_symbols} | set(outer.map.params)

        # 2. The producer's range over a tile and the tile buffers' extents and origins (in the arrays' coordinates).
        prange = {}
        pstart = {}
        pextent = {}
        for p in pparams:
            if p in whole:
                # A variable dimension: the producer runs over all of it per tile and the tile buffer holds it.
                b, en = (_exact(x) for x in whole[p])
                prange[p] = (b, en, 1)
                pstart[p] = b
                pextent[p] = symbolic.simplify(en - b + 1)
                continue
            q = follows[p]
            lo, hi = shifts[p]
            qb, qe = (_exact(x) for x in inner_range[q][:2])
            # The tile's range shifted by the stencil: the same form as the consumer's inner range (its end clipped
            # by a Min), which the CUDA code generator over-approximates to the constant block size. The producer's
            # own range is not applied: the consumer reads nothing the producer did not write.
            prange[p] = (qb + lo, qe + hi, 1)
            pstart[p] = qb + lo
            pextent[p] = sizes[cparams.index(q)] + hi - lo
        origin_sub = {symbolic.symbol(p): pstart[p] for p in pparams}

        def window(wsubset: subsets.Range) -> Tuple[List, List, List]:
            """Per dimension of a producer write subset: the buffer's extent, the origin of the buffer there, and
            the buffer's strides: the dimension the consumer's last parameter runs along (the threads of a block)
            is contiguous, then the other parameters' dimensions from the last, then the rest -- consecutive
            threads then touch consecutive words of the buffer (no shared-memory bank conflicts)."""
            extents, origin, rank = [], [], []
            for d, (b, e, _) in enumerate(wsubset.ranges):
                b, e = _exact(b), _exact(e)
                ps = [str(s) for s in b.free_symbols if str(s) in pparams]
                if ps:
                    extents.append(pextent[ps[0]])
                    origin.append(b.subs(origin_sub))
                    # A variable dimension is the slowest: the threads of a block run along the consumer's parameters.
                    rank.append((0, -cparams.index(follows[ps[0]])) if ps[0] in follows else (2, d))
                else:
                    extents.append(symbolic.simplify(e - b + 1))
                    origin.append(b)
                    rank.append((1, d))
            strides = [None] * len(extents)
            size = 1
            for d in sorted(range(len(extents)), key=lambda d: rank[d]):
                strides[d] = size
                size = size * extents[d]
            return extents, origin, strides

        buffers = {}  # producer output -> (access node, descriptor, origin)
        for name, (_node, _conn, wsubset) in writes.items():
            extents, origin, strides = window(wsubset)
            kind = "tile" if name in reads else "tile_unused"
            bname, bdesc = sdfg.add_transient(f"{name}_{kind}", extents, sdfg.arrays[name].dtype, storage=self.storage,
                                              lifetime=dtypes.AllocationLifetime.Scope, strides=strides,
                                              find_new_name=True)
            buffers[name] = (graph.add_access(bname), bdesc, origin)

        # 3. The producer over the tile, inside the outer map, writing the buffers: a copy of the producer's scope
        #    (its body and whatever else the scope holds that feeds it), its inputs routed through the outer map.
        if not isinstance(body_node, nd.NestedSDFG):
            raise NotImplementedError("tasklet producer bodies: use OTFMapFusion")
        # The tile map's parameters from the slowest: the variable dimensions, then the parameters in the order of the
        # consumer's (the consumer's last parameter runs on the x threads of the block, along the arrays' rows).
        ordered = [p for p in pparams if p in whole] + sorted((p for p in pparams if p in follows),
                                                              key=lambda p: cparams.index(follows[p]))
        pentry, pexit = graph.add_map(f"{first_map_entry.map.label}_tile", {p: prange[p] for p in ordered},
                                      schedule=self.schedule)
        pscope = graph.scope_subgraph(first_map_entry)
        copies = {}
        for node in pscope.nodes():
            if node is first_map_entry or node is first_map_exit:
                continue
            copies[node] = clone = copy.deepcopy(node)
            graph.add_node(clone)
            if isinstance(clone, nd.NestedSDFG):
                clone.sdfg.parent = graph
                clone.sdfg.parent_sdfg = sdfg
                clone.sdfg.parent_nsdfg_node = clone
                for s in tile_syms:
                    if s not in clone.sdfg.symbols:
                        clone.sdfg.add_symbol(s, sdfg.symbols.get(s, dtypes.int64))
                    clone.symbol_mapping[s] = symbolic.symbol(s)
        body = copies[body_node]
        for e in pscope.edges():
            if e.src in copies and e.dst in copies:
                graph.add_edge(copies[e.src], e.src_conn, copies[e.dst], e.dst_conn, copy.deepcopy(e.data))
            elif e.src is first_map_entry and e.dst in copies:
                if not e.src_conn or not e.src_conn.startswith("OUT_"):
                    raise NotImplementedError("dynamic inputs of the producer map")
                feed = next(x for x in graph.in_edges(first_map_entry) if x.dst_conn == "IN_" + e.src_conn[4:])
                graph.add_memlet_path(feed.src, outer, pentry, copies[e.dst], dst_conn=e.dst_conn,
                                      memlet=copy.deepcopy(e.data))
        for name, (_node, conn, wsubset) in writes.items():
            bnode, bdesc, origin = buffers[name]
            dealias.reduce_connector(body.sdfg, conn, bdesc, offset=subsets.Range.from_indices(subsets.Indices(origin)))
            shifted = subsets.Range([(_exact(b) - o, _exact(e) - o, s)
                                     for (b, e, s), o in zip(wsubset.ranges, origin)])
            graph.add_memlet_path(body, pexit, bnode, src_conn=conn, memlet=Memlet(data=bnode.data, subset=shifted))

        # 4. The consumer's reads: a per-iteration copy of its window of the buffer takes each read's place.
        for name, edges in reads.items():
            bnode, bdesc, origin = buffers[name]
            for edge in edges:
                read: subsets.Range = edge.data.subset
                squeeze = [d for d, (b, e, _) in enumerate(read.ranges) if symbolic.simplify(e - b) == 0]
                shape = [symbolic.simplify(e - b + 1) for d, (b, e, _) in enumerate(read.ranges) if d not in squeeze] or [1]
                loc_name, loc_desc = sdfg.add_transient(f"{name}_window", shape, bdesc.dtype,
                                                        lifetime=dtypes.AllocationLifetime.Scope, find_new_name=True)
                loc_read = graph.add_access(loc_name)
                win = subsets.Range([(_exact(b) - o, _exact(e) - o, s)
                                     for (b, e, s), o in zip(read.ranges, origin)])
                graph.add_memlet_path(bnode, centry, loc_read, memlet=Memlet(
                    data=bnode.data, subset=win, other_subset=subsets.Range.from_array(loc_desc)))
                consumer, conn = edge.dst, edge.dst_conn
                graph.add_edge(loc_read, None, consumer, conn, Memlet.from_array(loc_name, loc_desc))
                graph.remove_edge(edge)
                if not any(x.src_conn == edge.src_conn for x in graph.out_edges(centry)):
                    centry.remove_out_connector(edge.src_conn)
                    self._remove_input_path(graph, centry, "IN_" + edge.src_conn[4:])
                if isinstance(consumer, nd.NestedSDFG):
                    offset = subsets.Range.from_indices(subsets.Indices([b for b, _, _ in read.ranges]))
                    dealias.reduce_connector(consumer.sdfg, conn, loc_desc, offset=offset, squeeze=squeeze or None)
                for out in list(graph.out_edges(consumer)):
                    if out.src_conn == conn and out.data.data == name:
                        loc_write = graph.add_access(loc_name)
                        graph.add_edge(consumer, conn, loc_write, None, Memlet.from_array(loc_name, loc_desc))
                        self._detach_output_path(graph, out)
        centry.map.schedule = self.schedule
        # 5. A consumer left without work (every write of it was an in-place update of a fused array) leaves the
        # whole tile scope without work: the producer's tile fed only it. The exits have no edges, which the scope
        # tree cannot take, so the scope goes before the memlets are propagated.
        dead = graph.in_degree(cexit) == 0
        if dead:
            self._drop_dead_consumer(graph, sdfg, outer, outer_exit)
            if cexit in graph.nodes():  # unreachable from the outer entry: its edges were the detached writes
                graph.remove_node(cexit)
        else:
            # The memlets through the tile maps, from the inside out: the strip-mining left the outer map's edges
            # propagated with the unclipped tile ends.
            tree = graph.scope_tree()
            propagate_memlets_scope(graph.sdfg, graph, [tree[centry], tree[pentry]])
            propagate_memlets_scope(graph.sdfg, graph, [tree[outer]])

        # A producer left without work.
        if all(graph.out_degree(node) == 0 for node in outputs.values()):
            graph.remove_nodes_from(graph.all_nodes_between(first_map_entry, first_map_exit)
                                    | {first_map_entry, first_map_exit})
            for node in outputs.values():
                if node in graph.nodes() and graph.degree(node) == 0:
                    graph.remove_node(node)
            for node in list(graph.nodes()):  # inputs only the producer read
                if isinstance(node, nd.AccessNode) and graph.degree(node) == 0:
                    graph.remove_node(node)

    @staticmethod
    def _remove_input_path(graph: SDFGState, entry: nd.MapEntry, in_conn: str) -> None:
        """Remove the input ``in_conn`` of ``entry`` and, outward through the enclosing map entries, every edge that
        only fed it; a source access node left without edges goes too."""
        edge = next((e for e in graph.in_edges(entry) if e.dst_conn == in_conn), None)
        if edge is None:
            return
        graph.remove_edge(edge)
        entry.remove_in_connector(in_conn)
        src = edge.src
        if isinstance(src, nd.MapEntry) and edge.src_conn:
            if not any(e.src_conn == edge.src_conn for e in graph.out_edges(src)):
                src.remove_out_connector(edge.src_conn)
                TileStencilFusion._remove_input_path(graph, src, "IN_" + edge.src_conn[4:])
        elif isinstance(src, nd.AccessNode) and graph.degree(src) == 0:
            graph.remove_node(src)

    @staticmethod
    def _detach_output_path(graph: SDFGState, out) -> None:
        """Remove the consumer's write of an array to the outside: its path through every enclosing map exit."""
        path = graph.memlet_path(out)
        for e in path:
            graph.remove_edge(e)
            if isinstance(e.dst, nd.MapExit) and e.dst_conn:
                if not any(x.dst_conn == e.dst_conn for x in graph.in_edges(e.dst)):
                    e.dst.remove_in_connector(e.dst_conn)
                    e.dst.remove_out_connector("OUT_" + e.dst_conn[3:])
        last = path[-1].dst
        if isinstance(last, nd.AccessNode) and graph.degree(last) == 0:
            graph.remove_node(last)
