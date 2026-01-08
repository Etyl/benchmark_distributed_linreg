import pathlib

ACTIVE_SOLVERS = []


def get_data_path(filename: str) -> str:
    """
    Get the absolute path to a data file in the benchmark's data directory.
    """
    base_path = pathlib.Path(__file__).parent.parent
    data_dir = base_path / "data"
    return str(data_dir / filename)
