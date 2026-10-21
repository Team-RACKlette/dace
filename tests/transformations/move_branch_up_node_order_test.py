# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""move_branch_cfg_up_discard_conditions on a branch body whose end block comes before its start block in node order,
in a parent graph that has no explicit start block."""
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.helpers import move_branch_cfg_up_discard_conditions


def test_branch_body_listed_end_first():
    sdfg = dace.SDFG("move_branch_up_node_order")
    sdfg.add_array("A", [1], dace.float64)
    first = sdfg.add_state("first")  # the only source block: the start, with no explicit marking
    cond = ConditionalBlock("cond", sdfg, sdfg)
    sdfg.add_node(cond)
    last = sdfg.add_state("last")
    sdfg.add_edge(first, cond, dace.InterstateEdge())
    sdfg.add_edge(cond, last, dace.InterstateEdge())

    body = ControlFlowRegion("body", sdfg, cond)
    end = body.add_state("end")
    start = body.add_state("start", is_start_block=True)
    body.add_edge(start, end, dace.InterstateEdge())
    t = end.add_tasklet("set", {}, {"o"}, "o = 3")
    end.add_edge(t, "o", end.add_write("A"), None, dace.Memlet("A[0]"))
    cond.add_branch(CodeBlock("True"), body)

    move_branch_cfg_up_discard_conditions(cond, body)
    sdfg.validate()
    assert sdfg.start_block is first
    assert [n.label for n in sdfg.nodes() if sdfg.in_degree(n) == 0] == ["first"]
    import numpy as np
    A = np.zeros(1)
    sdfg(A=A)
    assert A[0] == 3
