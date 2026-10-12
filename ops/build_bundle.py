from pathlib import Path
import tarfile
root = Path(__file__).resolve().parents[1]
destination = root / "delivery"
destination.mkdir(exist_ok=True)
with tarfile.open(destination / "code.tar.gz", "w:gz") as archive:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "data", "logs", "runs", "delivery", ".pytest_cache", "__pycache__", ".venv"} for part in relative.parts):
            continue
        if path.is_file():
            archive.add(path, arcname=str(relative).replace("\\", "/"))
with tarfile.open(destination / "wheels.tar.gz", "w:gz") as archive:
    archive.add(root / "data/wheels", arcname="wheels")
print(destination)
