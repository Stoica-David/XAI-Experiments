from pathlib import Path


def find_project_root(current_path: Path, markers=('requirements.txt', '.git')):
    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    return current_path


current_file = Path(__file__).resolve()
project_root = find_project_root(current_file)

print(f"Project root: {project_root}")
