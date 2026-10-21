# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
On-the-fly fusion of a stencil producer into its consumer.

:class:`OTFStencilFusion` fuses a producer map into a consumer map that reads the producer's output at a
per-iteration subset (a point, or a slab along dimensions the producer parametrises, such as ``A[i + 1, j, 0:n]``)
by recomputing the producer for every read, as :class:`~dace.transformation.dataflow.OTFMapFusion` does for
tasklet bodies, and extends it to the shapes nested-SDFG bodies take:

* the producer may write several outputs (``A`` and ``B`` from one body); each recomputation keeps the one the
  read wants and discards the rest into a scope-local scratch buffer,
* a read along dimensions the producer parametrises (``A[i, 0:n]`` against a producer over ``k`` writing
  ``A[i, k]``) becomes a sequential inner map over those parameters filling a per-iteration buffer,
* a consumer that is a nested SDFG reading the array through a connector, and possibly updating it in place,
  reads the per-iteration buffer instead (its connector is restated through
  :func:`~dace.sdfg.dealias.reduce_connector`); in-place updates stay local to the iteration, so the
  transformation applies to such a consumer only when nothing reads the array after it.

The producer map is removed once none of its outputs has a reader left, and so is a consumer left without any
output (all it wrote were in-place updates, now local to the iteration) unless its body has side effects.
"""

import copy
import re
from typing import Dict, List, Optional, Set, Tuple

import sympy

from dace import dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, dealias
from dace.sdfg import nodes as nd
from dace.sdfg.propagation import propagate_memlets_scope
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, LoopRegion
from dace.transformation import transformation


def _written_conditionally(sdfg: SDFG, conn: str) -> bool:
    """Whether ``sdfg`` (a producer's body) writes ``conn`` in a branch of a conditional, here or in a nested SDFG
    that writes it: such a body writes fewer elements than its memlets claim (an interface-sharpening step that
    overwrites the reconstructed states of interface cells only), and a fused consumer would read the rest unwritten."""
    for state in sdfg.all_states():
        for node in state.data_nodes():
            if node.data != conn or state.in_degree(node) == 0:
                continue
            region = state.parent_graph
            while region is not None and region is not sdfg:
                if isinstance(region, ConditionalBlock):
                    return True
                region = region.parent_graph
            for e in state.in_edges(node):
                if isinstance(e.src, nd.NestedSDFG) and _written_conditionally(e.src.sdfg, e.src_conn):
                    return True
    return False


def _read_after(sdfg: SDFG, state: SDFGState, data: str) -> bool:
    """Whether ``data`` may be read after ``state`` executed: by a control flow block of ``sdfg`` that can run
    after it (``state`` itself excluded; a loop around it runs all of its blocks again), or by the control flow
    on the way (interstate edges, loop and branch conditions). Conservative: a read after a full overwrite counts."""
    word = re.compile(rf"\b{re.escape(data)}\b")
    after: Set[ControlFlowBlock] = set()
    code: List[str] = []
    block: ControlFlowBlock = state
    while not isinstance(block, SDFG):
        region = block.parent_graph
        if isinstance(region, ConditionalBlock):
            block = region  # the branches of a conditional do not reach each other
            continue
        reached, stack = set(), list(region.successors(block))
        while stack:
            b = stack.pop()
            if b not in reached:
                reached.add(b)
                stack.extend(region.successors(b))
        after |= reached
        code += [
            str(e.data.assignments) + " " + e.data.condition.as_string
            for e in region.edges()
            if e.src is block or e.src in reached
        ]
        if isinstance(region, LoopRegion):
            after.add(region)  # the next iteration
        block = region
    for b in after:
        if not isinstance(b, SDFGState):
            code += [str(e.data.assignments) + " " + e.data.condition.as_string for e in b.all_interstate_edges()]
            for r in [b, *b.all_control_flow_regions()]:
                if isinstance(r, LoopRegion):
                    code += [c.as_string for c in (r.loop_condition, r.init_statement, r.update_statement) if c]
                elif isinstance(r, ConditionalBlock):
                    code += [c.as_string for c, _ in r.branches if c is not None]
        for st in [b] if isinstance(b, SDFGState) else b.all_states():
            if st is state:
                continue
            if any(
                n.data == data and (st.out_degree(n) > 0 or any(e.data.wcr for e in st.in_edges(n)))
                for n in st.data_nodes()
            ):
                return True
    return any(word.search(c) for c in code)


class OTFStencilFusion(transformation.SingleStateTransformation):
    """Fuse a producer map into a consumer map by recomputing the producer per consumer read (see module docs)."""

    first_map_exit = transformation.PatternNode(nd.ExitNode)
    array = transformation.PatternNode(nd.AccessNode)
    second_map_entry = transformation.PatternNode(nd.EntryNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]

    @staticmethod
    def copies_scope() -> bool:
        """Whether the fusion copies the producer's whole scope (a subclass) or its body alone."""
        return False

    # ------------------------------------------------------------------------------------------------------------------
    # Analysis

    @staticmethod
    def _body_writes(graph: SDFGState, first_map_exit: nd.MapExit) -> Dict[str, Tuple[nd.Node, str, subsets.Range]]:
        """Per output array of the producer: the body node writing it, its connector and the per-iteration subset."""
        writes = {}
        for e in graph.in_edges(first_map_exit):
            if e.data.data is None or e.data.data in writes:
                return {}
            writes[e.data.data] = (e.src, e.src_conn, e.data.subset)
        return writes

    @staticmethod
    def solve(
        first_params: List[str], write: subsets.Range, second_params: List[str], read: subsets.Range
    ) -> Optional[Tuple[Dict[str, symbolic.SymExpr], Dict[str, Tuple[symbolic.SymExpr, symbolic.SymExpr]]]]:
        """Match the producer's write subset against one read subset of the consumer.

        Every dimension of ``write`` is a point. A point dimension of ``read`` fixes the producer parameter that
        appears there (``matched``); a range dimension turns that parameter into an inner iteration
        (``inner``, with the parameter's range). Returns ``None`` when the dimensions cannot be matched this way.
        """
        matched: Dict[str, symbolic.SymExpr] = {}
        inner: Dict[str, Tuple[symbolic.SymExpr, symbolic.SymExpr]] = {}
        if len(write) != len(read):
            return None
        for (wb, we, ws), (rb, re_, rs) in zip(write.ranges, read.ranges):
            wb, we = symbolic.pystr_to_symbolic(wb), symbolic.pystr_to_symbolic(we)
            rb, re_ = symbolic.pystr_to_symbolic(rb), symbolic.pystr_to_symbolic(re_)
            if symbolic.simplify(ws - 1) != 0 or symbolic.simplify(rs - 1) != 0:
                return None  # consecutive elements only
            params = [s for s in wb.free_symbols if str(s) in first_params]
            if symbolic.simplify(we - wb) != 0:
                # The producer writes a whole slab along this dimension per iteration (a body that covers every
                # member): the read must take the same slab or a constant part of it (one member), and no
                # parameter is involved.
                if params or any(str(s) in second_params for s in (rb.free_symbols | re_.free_symbols)):
                    return None
                below, above = symbolic.simplify(rb - wb), symbolic.simplify(we - re_)
                if not (below.is_Integer and above.is_Integer and below >= 0 and above >= 0):
                    return None
                continue
            if len(params) > 1:
                return None
            if not params:
                if symbolic.simplify(wb - rb) != 0 or symbolic.simplify(we - re_) != 0:
                    return None  # a constant dimension read elsewhere
                continue
            psym = params[0]
            p = str(psym)
            # wb is affine in p: solve wb(p) == rb for p (point read) or for the range ends (slab read)
            sol = sympy.solve(sympy.Eq(wb, rb), psym)
            if len(sol) != 1:
                return None
            if symbolic.simplify(re_ - rb) == 0:
                if p in matched or p in inner:
                    return None
                matched[p] = sol[0]
            else:
                sol_e = sympy.solve(sympy.Eq(wb, re_), psym)
                if len(sol_e) != 1 or p in matched or p in inner:
                    return None
                lo, hi = sol[0], sol_e[0]
                # The ends may be symbolic (a read by a map over tiles): swap them only when the order is known.
                if symbolic.simplify(hi - lo).is_negative:
                    lo, hi = hi, lo
                inner[p] = (lo, hi)
        if len(matched) + len(inner) != len(first_params):
            return None
        return matched, inner

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        array = self.array.data
        if not sdfg.arrays[array].transient:
            return False
        if graph.in_degree(self.array) != 1:
            return False
        first_map_entry = graph.entry_node(self.first_map_exit)
        for dnode in graph.scope_subgraph(first_map_entry).data_nodes():
            if not sdfg.arrays[dnode.data].transient:
                return False
        writes = self._body_writes(graph, self.first_map_exit)
        if array not in writes:
            return False
        body = {w[0] for w in writes.values()}
        if len(body) != 1 or not isinstance(next(iter(body)), nd.NestedSDFG):
            return False  # one nested SDFG writes all outputs (a tasklet body is OTFMapFusion's case)
        if not self.copies_scope() and any(e.src is not first_map_entry for e in graph.in_edges(next(iter(body)))):
            return False  # the recomputation copies the body alone: its inputs must be the map's
        if any(e.data.data in writes for e in graph.in_edges(first_map_entry)):
            return False  # the producer reads an output of its own (an in-place correction): its copy would read the
            # array through the connector the recomputation narrows to the buffer it writes
        body_node, conn, _subset = writes[array]
        if _written_conditionally(body_node.sdfg, conn):
            return False
        in_place = False
        for e in graph.out_edges(self.second_map_entry):
            if e.data.data != array:
                continue
            if self.solve(first_map_entry.map.params, writes[array][2], self.second_map_entry.map.params,
                          e.data.subset) is None:
                return False
            in_place |= any(o.src_conn == e.dst_conn and o.data.data == array for o in graph.out_edges(e.dst))
        if in_place:
            # The consumer's update of the array stays in the iteration: nothing may read the updated array, in
            # this state (beyond the producer's output the consumer reads) or after it.
            if any(n is not self.array and n.data == array and graph.out_degree(n) > 0 for n in graph.data_nodes()):
                return False
            if _read_after(sdfg, graph, array):
                return False
        # The producer goes once it is fused: none of its outputs may have a reader but this consumer, in this
        # state or after it (a later state reading another member of what it wrote, say). Fusing while the producer
        # stays would only add its work to the consumer's.
        outputs = {graph.memlet_path(e)[-1].dst for e in graph.out_edges(self.first_map_exit)}
        for node in outputs:
            if any(e.dst is not self.second_map_entry for e in graph.out_edges(node)):
                return False
        if any(_read_after(sdfg, graph, node.data) for node in outputs):
            return False
        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Application

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        array = self.array.data
        first_map_exit: nd.MapExit = self.first_map_exit
        first_map_entry: nd.MapEntry = graph.entry_node(first_map_exit)
        second_map_entry: nd.MapEntry = self.second_map_entry
        second_map_exit: nd.MapExit = graph.exit_node(second_map_entry)
        writes = self._body_writes(graph, first_map_exit)
        body_node = writes[array][0]
        desc = sdfg.arrays[array]

        # The consumer's per-iteration reads of the array, each already a point-or-slab subset.
        for edge in list(graph.out_edges(second_map_entry)):
            if edge.data.data != array:
                continue
            read: subsets.Range = edge.data.subset
            mapping = self.solve(first_map_entry.map.params, writes[array][2], second_map_entry.map.params, read)
            if mapping is None:
                raise ValueError(f"cannot match the read {read} of {array} against the producer")
            matched, inner = mapping

            # 1. A per-iteration buffer takes the read's place, filled by a copy of the producer body that runs
            #    over the unmatched parameters. Along a dimension the producer writes as a slab the buffer holds
            #    the whole slab (the read may take one member of it).
            buf = []
            for (rb, re_, rs), (wb, we, _) in zip(read.ranges, writes[array][2].ranges):
                wb, we = symbolic.pystr_to_symbolic(wb), symbolic.pystr_to_symbolic(we)
                slab = symbolic.simplify(we - wb) != 0 and not any(
                    str(x) in first_map_entry.map.params for x in wb.free_symbols)
                buf.append((wb, we, rs) if slab else (rb, re_, rs))
            squeeze = [d for d, (b, e, _) in enumerate(buf) if symbolic.simplify(e - b) == 0]
            shape = [symbolic.simplify(e - b + 1) for d, (b, e, _) in enumerate(buf) if d not in squeeze] or [1]
            loc_name, loc_desc = sdfg.add_transient(
                f"{array}_otf", shape, desc.dtype, lifetime=dtypes.AllocationLifetime.Scope, find_new_name=True
            )
            offset = subsets.Range.from_indices(subsets.Indices([b for b, _, _ in buf]))
            loc_read = self._recompute(graph, sdfg, first_map_entry, second_map_entry, body_node, writes, array,
                                       matched, inner, loc_name, loc_desc, offset, squeeze)

            # 2. The consumer reads the buffer; an in-place update of the array by the consumer stays in it.
            consumer, conn = edge.dst, edge.dst_conn
            graph.add_edge(loc_read, None, consumer, conn, Memlet.from_array(loc_name, loc_desc))
            graph.remove_edge(edge)
            if not any(x.src_conn == edge.src_conn for x in graph.out_edges(second_map_entry)):
                second_map_entry.remove_out_connector(edge.src_conn)
            if isinstance(consumer, nd.NestedSDFG):
                dealias.reduce_connector(consumer.sdfg, conn, loc_desc, offset=offset, squeeze=squeeze or None)
            for out in list(graph.out_edges(consumer)):
                if out.src_conn == conn and out.data.data == array:
                    loc_write = graph.add_access(loc_name)
                    graph.add_edge(consumer, conn, loc_write, None, Memlet.from_array(loc_name, loc_desc))
                    self._detach_output(graph, out, second_map_exit)

        # The consumer no longer reads the array from outside.
        for e in list(graph.in_edges(second_map_entry)):
            if e.data.data == array and not any(
                    o.data.data == array for o in graph.out_edges(second_map_entry)):
                graph.remove_edge_and_connectors(e)
        if graph.in_degree(second_map_exit) == 0:
            self._drop_dead_consumer(graph, sdfg, second_map_entry, second_map_exit)
        if graph.out_degree(self.array) == 0 and all(
                graph.out_degree(graph.memlet_path(e)[-1].dst) == 0 for e in graph.out_edges(first_map_exit)):
            # Nothing reads any output of the producer: drop it with its output nodes.
            outputs = [graph.memlet_path(e)[-1].dst for e in graph.out_edges(first_map_exit)]
            graph.remove_nodes_from(graph.all_nodes_between(first_map_entry, first_map_exit)
                                    | {first_map_entry, first_map_exit})
            for node in outputs:
                if node in graph.nodes() and graph.degree(node) == 0:
                    graph.remove_node(node)
            for node in list(graph.nodes()):  # inputs only the producer read
                if isinstance(node, nd.AccessNode) and graph.degree(node) == 0:
                    graph.remove_node(node)
        # The consumer's scope is the only one whose memlets changed: propagated here, not over the whole SDFG, which
        # may hold loop bodies the frontend left with sliced connectors (an in-place correction) that the SDFG-wide
        # propagation refuses.
        if second_map_entry in graph.nodes():
            tree = graph.scope_tree()
            propagate_memlets_scope(sdfg, graph, [tree[second_map_entry]])

    @staticmethod
    def annotates_memlets() -> bool:
        return True

    @staticmethod
    def _drop_dead_consumer(graph: SDFGState, sdfg: SDFG, entry: nd.MapEntry, exit_: nd.MapExit) -> None:
        """A consumer whose every write was an in-place update of a fused array writes nothing outside the
        iteration any more, and its exit is left without edges (outside the scope, which the scope tree cannot
        take). Remove the scope, unless a node in it has side effects; then reconnect its sinks to the exit."""
        scope, stack = set(), [entry]
        while stack:  # everything the entry reaches: the exit has no edges left, so this is the scope
            node = stack.pop()
            if node not in scope:
                scope.add(node)
                stack.extend(graph.successors(node))
        if any(isinstance(n, nd.CodeNode) and n.has_side_effects(sdfg) for n in scope):
            for n in scope:
                if graph.out_degree(n) == 0:
                    graph.add_nedge(n, exit_, Memlet())
            return
        sources = set()
        for e in graph.in_edges(entry):
            sources.add(e.src)
            graph.remove_edge_and_connectors(e)
        graph.remove_nodes_from(scope | {exit_})
        for node in sources:
            if isinstance(node, nd.AccessNode) and graph.degree(node) == 0:
                graph.remove_node(node)

    @staticmethod
    def _detach_output(graph: SDFGState, out, second_map_exit: nd.MapExit) -> None:
        """Remove the consumer's write of the array to the outside (the path through the map exit)."""
        path = graph.memlet_path(out)
        for e in path:
            graph.remove_edge(e)
            if e.dst is second_map_exit and e.dst_conn and graph.in_degree(second_map_exit) >= 0:
                if not any(x.dst_conn == e.dst_conn for x in graph.in_edges(second_map_exit)):
                    second_map_exit.remove_in_connector(e.dst_conn)
                    second_map_exit.remove_out_connector("OUT_" + e.dst_conn[3:])
        last = path[-1].dst
        if isinstance(last, nd.AccessNode) and graph.degree(last) == 0:
            graph.remove_node(last)

    def _recompute(self, graph: SDFGState, sdfg: SDFG, first_map_entry: nd.MapEntry, second_map_entry: nd.MapEntry,
                   body_node: nd.Node, writes, array: str, matched, inner, loc_name: str, loc_desc,
                   offset: subsets.Range, squeeze: List[int]) -> nd.AccessNode:
        """A copy of the producer body inside the consumer map, iterating the unmatched parameters, writing the
        wanted array into a new per-iteration buffer (returned as its access node) and the others into scratch."""
        body = copy.deepcopy(body_node)
        graph.add_node(body)
        if isinstance(body, nd.NestedSDFG):
            body.sdfg.parent = graph
            body.sdfg.parent_sdfg = sdfg
            body.sdfg.parent_nsdfg_node = body
            for p, expr in matched.items():
                body.symbol_mapping[p] = expr
        else:
            raise NotImplementedError("tasklet producer bodies: use OTFMapFusion")

        # The inner iteration over the producer parameters the read spans.
        if inner:
            ranges = {p: f"{lo}:{hi} + 1" for p, (lo, hi) in inner.items()}
            inner_entry, inner_exit = graph.add_map(f"{array}_recompute", ranges, schedule=dtypes.ScheduleType.Sequential)
            for p in inner:
                body.symbol_mapping[p] = symbolic.symbol(p)
        else:
            inner_entry = inner_exit = None

        # Inputs of the producer body come from the consumer map's entry (through the inner map).
        for e in graph.in_edges(body_node):
            if e.src is not first_map_entry:
                raise NotImplementedError("producer body inputs must come from its map entry")
            src_conn = e.src_conn
            in_conn = "IN_" + src_conn[4:]
            outer = next(x for x in graph.in_edges(first_map_entry) if x.dst_conn == in_conn)
            # Route the outer input into the consumer map once per array.
            existing = [x for x in graph.in_edges(second_map_entry) if x.data.data == outer.data.data]
            if existing:
                out_conn = "OUT_" + existing[0].dst_conn[3:]
            else:
                new_in = "IN_" + second_map_entry.next_connector(src_conn[4:])
                out_conn = "OUT_" + new_in[3:]
                second_map_entry.add_in_connector(new_in)
                second_map_entry.add_out_connector(out_conn)
                graph.add_edge(outer.src, outer.src_conn, second_map_entry, new_in, copy.deepcopy(outer.data))
            memlet = copy.deepcopy(e.data)
            if inner_entry is not None:
                iconn = "IN_" + inner_entry.next_connector(src_conn[4:])
                inner_entry.add_in_connector(iconn)
                inner_entry.add_out_connector("OUT_" + iconn[3:])
                graph.add_edge(second_map_entry, out_conn, inner_entry, iconn, copy.deepcopy(memlet))
                graph.add_edge(inner_entry, "OUT_" + iconn[3:], body, e.dst_conn, memlet)
            else:
                graph.add_edge(second_map_entry, out_conn, body, e.dst_conn, memlet)

        # Outputs: the wanted array into the per-iteration buffer, the others into scratch of the same shape.
        loc_read = graph.add_access(loc_name)
        for name, (_node, conn, wsubset) in writes.items():
            if name == array:
                target, tdesc, toffset = loc_read, loc_desc, offset
            else:
                tname, tdesc = sdfg.add_transient(f"{name}_otf_unused", list(loc_desc.shape), sdfg.arrays[name].dtype,
                                                  lifetime=dtypes.AllocationLifetime.Scope, find_new_name=True)
                target = graph.add_access(tname)
                origin = {symbolic.symbol(p): expr for p, expr in matched.items()}
                origin.update({symbolic.symbol(p): lo for p, (lo, _hi) in inner.items()})
                toffset = subsets.Range.from_indices(subsets.Indices(
                    [symbolic.pystr_to_symbolic(b).subs(origin) for b, _, _ in wsubset.ranges]))
            if isinstance(body, nd.NestedSDFG):
                dealias.reduce_connector(body.sdfg, conn, tdesc, offset=toffset, squeeze=squeeze or None)
            memlet = Memlet.from_array(target.data, tdesc)
            if inner_exit is not None:
                iconn = "IN_" + inner_exit.next_connector(conn)
                inner_exit.add_in_connector(iconn)
                inner_exit.add_out_connector("OUT_" + iconn[3:])
                graph.add_edge(body, conn, inner_exit, iconn, copy.deepcopy(memlet))
                graph.add_edge(inner_exit, "OUT_" + iconn[3:], target, None, memlet)
            else:
                graph.add_edge(body, conn, target, None, memlet)
        return loc_read
