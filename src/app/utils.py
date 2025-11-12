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
    Récupère les clés de classes pour les rôles fournis depuis `results[endpoint][mapping_key]`,
    convertit en int, ignore les valeurs non convertibles et renvoie une liste d'entiers uniques triés.
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