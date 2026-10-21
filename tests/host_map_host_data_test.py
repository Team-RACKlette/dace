import dace
import pytest


def create_assign_sdfg():
    sdfg = dace.SDFG("single_iteration_map")
    state = sdfg.add_state()
    array_size = 1
    A, _ = sdfg.add_array("A", [array_size], dace.float32)
    map_entry, map_exit = state.add_map("map_1_iter", {"i": "0:1"})
    tasklet = state.add_tasklet("set_to_1", {}, {"OUT__a"}, "_a = 1")
    map_exit.add_in_connector("IN__a")
    map_exit.add_out_connector("OUT__a")
    tasklet.add_out_connector("OUT__a")
    an = state.add_write("A")
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, "OUT__a", map_exit, "IN__a", dace.Memlet(f"A[0]"))
    state.add_edge(map_exit, "OUT__a", an, None, dace.Memlet(f"A[0]"))
    sdfg.validate()
    return A, sdfg


def create_assign_sdfg_with_views():
    sdfg = dace.SDFG("single_iteration_map")
    state = sdfg.add_state()
    array_size = 5
    _, _ = sdfg.add_array("A", [array_size], dace.float32)
    v_A, _ = sdfg.add_view("v_A", [1], dace.float32)
    map_entry, map_exit = state.add_map("map_1_iter", {"i": "0:1"})
    tasklet = state.add_tasklet("set_to_1", {}, {"OUT__a"}, "_a = 1")
    map_exit.add_in_connector("IN__a")
    map_exit.add_out_connector("OUT__a")
    tasklet.add_out_connector("OUT__a")
    an = state.add_write("v_A")
    an2 = state.add_write("A")
    an.add_out_connector("views")
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet(None))
    state.add_edge(tasklet, "OUT__a", map_exit, "IN__a", dace.Memlet(f"v_A[0]"))
    state.add_edge(map_exit, "OUT__a", an, None, dace.Memlet(f"v_A[0]"))
    state.add_edge(an, "views", an2, None, dace.Memlet(f"A[0:1]"))
    sdfg.validate()
    return v_A, sdfg


def create_increment_sdfg():
    sdfg = dace.SDFG("increment_map")
    state = sdfg.add_state()
    array_size = 500
    A, _ = sdfg.add_array("A", [array_size], dace.float32)
    map_entry, map_exit = state.add_map("map_1_iter", {"i": f"0:{array_size}"})
    tasklet = state.add_tasklet("inc_by_1", {}, {"OUT__a"}, "_a = _a + 1")
    map_entry.add_in_connector("IN__a")
    map_entry.add_out_connector("OUT__a")
    map_exit.add_in_connector("IN__a")
    map_exit.add_out_connector("OUT__a")
    tasklet.add_in_connector("IN__a")
    tasklet.add_out_connector("OUT__a")
    an1 = state.add_read("A")
    an2 = state.add_write("A")
    state.add_edge(an1, None, map_entry, "IN__a", dace.Memlet(f"A[0:{array_size}]"))
    state.add_edge(map_entry, "OUT__a", tasklet, "IN__a", dace.Memlet("A[i]"))
    state.add_edge(tasklet, "OUT__a", map_exit, "IN__a", dace.Memlet(f"A[i]"))
    state.add_edge(map_exit, "OUT__a", an2, None, dace.Memlet(f"A[0:{array_size}]"))
    sdfg.validate()
    return A, sdfg


def create_increment_sdfg_with_views():
    sdfg = dace.SDFG("increment_map")
    state = sdfg.add_state()
    array_size = 500
    view_size = 100
    _, _ = sdfg.add_array("A", [array_size], dace.float32)
    v_A, _ = sdfg.add_view("v_A", [view_size], dace.float32)
    map_entry, map_exit = state.add_map("map_1_iter", {"i": f"0:{view_size}"})
    tasklet = state.add_tasklet("inc_by_1", {}, {"OUT__a"}, "_a = _a + 1")
    map_entry.add_in_connector("IN__a")
    map_entry.add_out_connector("OUT__a")
    map_exit.add_in_connector("IN__a")
    map_exit.add_out_connector("OUT__a")
    tasklet.add_in_connector("IN__a")
    tasklet.add_out_connector("OUT__a")
    an1 = state.add_read("A")
    an2 = state.add_write("A")
    an3 = state.add_read("v_A")
    an4 = state.add_write("v_A")
    an3.add_in_connector("views")
    an4.add_out_connector("views")
    state.add_edge(an1, None, an3, "views", dace.Memlet(f"A[0:100]"))
    state.add_edge(an3, None, map_entry, "IN__a", dace.Memlet(f"v_A[i]"))
    state.add_edge(map_entry, "OUT__a", tasklet, "IN__a", dace.Memlet("v_A[i]"))
    state.add_edge(tasklet, "OUT__a", map_exit, "IN__a", dace.Memlet(f"v_A[i]"))
    state.add_edge(map_exit, "OUT__a", an4, None, dace.Memlet(f"v_A[i]"))
    state.add_edge(an4, "views", an2, None, dace.Memlet(f"A[0:100]"))
    sdfg.validate()
    return v_A, sdfg


@pytest.mark.parametrize(
    "sdfg_creator",
    [create_assign_sdfg, create_increment_sdfg, create_assign_sdfg_with_views, create_increment_sdfg_with_views],
)
class TestHostDataHostMapParams:
    def test_host_data(self, sdfg_creator):
        """Test that arrays marked as host_data remain on host after GPU transformation."""
        A, sdfg = sdfg_creator()
        sdfg.apply_gpu_transformations(host_data=[A])
        sdfg.validate()

        assert sdfg.arrays["A"].storage != dace.dtypes.StorageType.GPU_Global

    def test_host_map(self, sdfg_creator):
        """Test that maps marked as host_maps remain on host after GPU transformation."""
        A, sdfg = sdfg_creator()
        host_maps = [n.guid for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.EntryNode)]
        sdfg.apply_gpu_transformations(host_maps=host_maps)
        sdfg.validate()
        assert sdfg.arrays["A"].storage != dace.dtypes.StorageType.GPU_Global

    @pytest.mark.parametrize("pass_empty", [True, False])
    def test_no_host_map_or_data(self, sdfg_creator, pass_empty):
        """Test default GPU transformation behavior with no host constraints."""
        A, sdfg = sdfg_creator()

        if pass_empty:
            sdfg.apply_gpu_transformations(host_maps=[], host_data=[])
        else:
            sdfg.apply_gpu_transformations()

        sdfg.validate()

        # Verify array storage locations
        assert "A" in sdfg.arrays and "gpu_A" in sdfg.arrays
        assert sdfg.arrays["A"].storage != dace.dtypes.StorageType.GPU_Global
        assert sdfg.arrays["gpu_A"].storage == dace.dtypes.StorageType.GPU_Global

        # Verify map schedules
        for s in sdfg.states():
            for n in s.nodes():
                if isinstance(n, dace.nodes.MapEntry):
                    assert n.map.schedule == dace.ScheduleType.GPU_Device


def test_host_data_keeps_a_library_node_between_host_arrays_on_the_host():
    """A library node reading or writing host data stays on the host, like a map would."""
    from dace.libraries.standard.nodes.copy.node import CopyLibraryNode

    sdfg = dace.SDFG("host_copy")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    state = sdfg.add_state()
    copy = CopyLibraryNode(name="copy_a")
    state.add_edge(state.add_read("A"), None, copy, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("A[0:4]"))
    state.add_edge(copy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write("B"), None, dace.Memlet("B[0:4]"))

    sdfg.apply_gpu_transformations(host_data=["A", "B"])
    sdfg.validate()

    assert copy.schedule != dace.ScheduleType.GPU_Device
    assert all(sdfg.arrays[n].storage != dace.dtypes.StorageType.GPU_Global for n in ("A", "B"))


def _record_member_copies_sdfg(device_output: bool):
    """``dst[i] = src[1, i]`` for i in 0..2 as three free tasklets, then ``out[:] = dst[:]``: the element-wise
    copy of a small record member. With ``device_output`` a fourth tasklet also writes ``dev[0] = src[0, 0]``
    into an array that is not host data."""
    sdfg = dace.SDFG("record_member_copies" + ("_dev" if device_output else ""))
    sdfg.add_array("src", [3, 3], dace.float64)
    sdfg.add_array("dst", [3], dace.float64)
    sdfg.add_array("out", [3], dace.float64)
    state = sdfg.add_state()
    src = state.add_read("src")
    dst = state.add_access("dst")
    for i in range(3):
        t = state.add_tasklet(f"copy_{i}", {"a"}, {"b"}, "b = a")
        state.add_edge(src, None, t, "a", dace.Memlet(f"src[1, {i}]"))
        state.add_edge(t, "b", dst, None, dace.Memlet(f"dst[{i}]"))
    state.add_nedge(dst, state.add_write("out"), dace.Memlet("dst[0:3]"))
    if device_output:
        sdfg.add_array("dev", [3], dace.float64)
        t = state.add_tasklet("copy_dev", {"a"}, {"b"}, "b = a")
        state.add_edge(src, None, t, "a", dace.Memlet("src[0, 0]"))
        state.add_edge(t, "b", state.add_write("dev"), None, dace.Memlet("dev[0]"))
    sdfg.validate()
    return sdfg


def test_free_tasklets_between_host_data_stay_on_the_host():
    """Tasklets that only copy elements between host data have nothing to do on the device: no device copy
    of the arrays, no one-thread kernels, and the program runs (on the host) as before."""
    import numpy as np

    sdfg = _record_member_copies_sdfg(device_output=False)
    sdfg.apply_gpu_transformations(host_data=["src", "dst", "out"])
    sdfg.validate()

    assert not any(name.startswith("gpu_") for name in sdfg.arrays)
    assert all(d.storage != dace.dtypes.StorageType.GPU_Global for d in sdfg.arrays.values())
    assert not any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())

    src = np.arange(9, dtype=np.float64).reshape(3, 3).copy()
    out = np.zeros(3)
    sdfg(src=src, dst=np.zeros(3), out=out)
    assert np.array_equal(out, src[1])


def test_host_data_that_a_device_tasklet_reads_is_copied_to_the_device():
    """A tasklet that also writes device data runs in a one-thread kernel: the host data it reads must have a
    device copy, and then the host-side tasklets reading the same array use that copy too."""
    sdfg = _record_member_copies_sdfg(device_output=True)
    sdfg.apply_gpu_transformations(host_data=["src", "dst", "out"])
    sdfg.validate()

    assert sdfg.arrays["gpu_src"].storage == dace.dtypes.StorageType.GPU_Global
    assert sdfg.arrays["dev"].storage != dace.dtypes.StorageType.GPU_Global
    kernels = [n for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device]
    assert kernels


def test_inputs_of_a_host_map_are_not_copied_to_the_device():
    """A map of ``host_maps`` reads its input where it is: no device copy of it is made and dropped."""
    A, sdfg = create_increment_sdfg()
    host_maps = [n.guid for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.EntryNode)]
    sdfg.apply_gpu_transformations(host_maps=host_maps)
    sdfg.validate()
    assert "gpu_A" not in sdfg.arrays
    assert all(n.map.schedule != dace.ScheduleType.GPU_Device
               for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))


def test_tasklets_whose_result_a_kernel_reads_run_on_the_device():
    """``dst`` is also read by a map: it needs a device copy, so the tasklets writing it run on the device, and
    the array they read needs one too. The generated code must not hand a host array to a kernel."""
    sdfg = _record_member_copies_sdfg(device_output=False)
    sdfg.add_array("sum", [3], dace.float64)
    state = sdfg.start_block
    dst = next(n for n in state.data_nodes() if n.data == "dst")
    state.add_mapped_tasklet("scale", {"i": "0:3"}, {"x": dace.Memlet("dst[i]")}, "y = 2 * x",
                             {"y": dace.Memlet("sum[i]")}, input_nodes={dst}, external_edges=True)
    sdfg.validate()
    sdfg.apply_gpu_transformations(host_data=["src", "dst", "out"])
    sdfg.validate()

    assert {"gpu_src", "gpu_dst"} <= set(sdfg.arrays)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet) and node.label.startswith("copy_"):
                assert state.entry_node(node).map.schedule == dace.ScheduleType.GPU_Device
    sdfg.generate_code()


def test_a_copy_library_node_between_host_data_reads_its_input_on_the_host():
    """A copy library node between host data stays on the host; the array it reads gets no device copy."""
    from dace.libraries.standard.nodes.copy.node import CopyLibraryNode

    sdfg = dace.SDFG("host_copy_of_member")
    sdfg.add_array("A", [3], dace.float64)
    sdfg.add_array("B", [3], dace.float64)
    state = sdfg.add_state()
    copy = CopyLibraryNode(name="copy_a")
    state.add_edge(state.add_read("A"), None, copy, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("A[0:3]"))
    state.add_edge(copy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write("B"), None, dace.Memlet("B[0:3]"))

    sdfg.apply_gpu_transformations(host_data=["A", "B"])
    sdfg.validate()
    assert "gpu_A" not in sdfg.arrays and "gpu_B" not in sdfg.arrays
    assert copy.schedule != dace.ScheduleType.GPU_Device


if __name__ == "__main__":
    pytest.main([__file__])
