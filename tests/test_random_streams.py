from random_streams import RandomStreams


def test_keyed_uniform_is_independent_of_call_order():
    first = RandomStreams(seed=42)
    expected_a = first.uniform("exit", 3, "miner-1")
    expected_b = first.uniform("exit", 3, "miner-2")

    second = RandomStreams(seed=42)
    actual_b = second.uniform("exit", 3, "miner-2")
    actual_a = second.uniform("exit", 3, "miner-1")

    assert actual_a == expected_a
    assert actual_b == expected_b


def test_one_stream_does_not_shift_another_stream():
    expected = RandomStreams(seed=42).uniform("exit", 4, "miner-9")

    streams = RandomStreams(seed=42)
    streams.uniform("repair_acceptance", 4, "dataset-1", "miner-2")
    streams.uniform("repair_acceptance", 4, "dataset-2", "miner-3")

    assert streams.uniform("exit", 4, "miner-9") == expected


def test_ordered_result_does_not_depend_on_input_order():
    streams = RandomStreams(seed=7)
    items = ["miner-3", "miner-1", "miner-2"]

    forward = streams.ordered("placement", 0, items, key=lambda item: item)
    backward = streams.ordered("placement", 0, reversed(items), key=lambda item: item)

    assert forward == backward
    assert sorted(forward) == sorted(items)


def test_uniform_draw_is_in_half_open_unit_interval():
    streams = RandomStreams(seed=17)
    values = [streams.uniform("stream", time, time) for time in range(100)]
    assert all(0.0 <= value < 1.0 for value in values)
