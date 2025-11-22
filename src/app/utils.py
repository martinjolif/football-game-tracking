def get_class_keys(mapping, class_name):
    # Returns a list of keys matching the class_name
    return [k for k, v in mapping.items() if v == class_name]


def collect_class_ids(
    results,
    endpoint: str = "http://localhost:8000/player-detection/image",
    mapping_key: str = "mapping_class",
    roles=None,
) -> list:
    """
    Retrieves class keys for provided roles from `results[endpoint][mapping_key]`,
    converts to int, ignores non-convertible values and returns a sorted list of unique integers.
    """
    if roles is None:
        raise ValueError("roles is missing.")

    mapping = results.get(endpoint, {}).get(mapping_key, {})
    ids = set()

    for role in roles:
        try:
            keys = get_class_keys(mapping, role)
        except Exception:
            keys = []
        for k in keys or []:
            try:
                ids.add(int(k))
            except (ValueError, TypeError):
                continue

    return sorted(ids)