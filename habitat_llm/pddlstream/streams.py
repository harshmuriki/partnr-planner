def make_stream_map(name_to_entity):
    # All reachable facts are pre-certified into the PDDLStream init directly,
    # so no stream functions are needed. Return an empty map for pure STRIPS.
    return {}
