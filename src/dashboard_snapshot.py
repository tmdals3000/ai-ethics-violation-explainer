"""Prepare the bundled dev metadata for the dashboard; never load source text."""
import hashlib
import os
from pathlib import Path
from zipfile import ZipFile


FILES = ("texts.csv", "nodes.csv", "examples.csv", "dev_sample.csv", "splits.csv", "similarity.csv", "coords.csv")


def configure(cfg, root):
    configured = Path(cfg["paths"]["data_dir"]) / "derived"
    forced = os.environ.get("AIETHICS_DASHBOARD_SNAPSHOT") == "1"
    if not forced and all((configured / name).is_file() for name in FILES):
        return
    archive = root / "assets/dev_dashboard.zip"
    if not archive.is_file():
        raise FileNotFoundError("assets/dev_dashboard.zip is missing. Download the complete repository.")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    snapshot = root / "data/dashboard_snapshot"
    derived = snapshot / "derived"
    marker = snapshot / ".archive.sha256"
    if not (marker.is_file() and marker.read_text(encoding="ascii") == digest
            and all((derived / name).is_file() for name in FILES)):
        derived.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as bundle:
            # Read only known filenames; archive paths never become filesystem paths.
            for name in FILES:
                (derived / name).write_bytes(bundle.read("derived/" + name))
        marker.write_text(digest, encoding="ascii")
    cfg["paths"].update(data_dir=str(snapshot), raw_dir=str(root / "data/raw"),
                        ta_dir=str(snapshot / "ta"), hf_home=str(root / "data/hf-cache"))
    cfg["_dashboard_snapshot"] = True
