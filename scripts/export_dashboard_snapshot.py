"""Bundle actual dev metadata and saved similarities, excluding all source text."""
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adapters, data
from src.common import ROOT, load_config


def main():
    cfg = load_config()
    texts, nodes, examples, sample = data.load(cfg)
    meta = texts[texts.text_id.isin(sample.text_id) & (texts.split == "dev")][data.SCHEMA].copy()
    assert meta.text_id.is_unique and set(meta.text_id) == set(sample.text_id)
    meta[["text", "context"]] = ""
    frames = {"texts.csv": meta, "nodes.csv": nodes, "examples.csv": examples.iloc[:0],
              "dev_sample.csv": sample[sample.text_id.isin(meta.text_id)]}
    for name in ("splits", "similarity", "coords"):
        frame, _ = adapters.read(cfg, name)
        frames[name + ".csv"] = frame[frame.text_id.isin(meta.text_id)]
    manifest = {"description": "Actual baseline_v2 dev results; source text and context removed",
                "prompt_version": cfg["judge"]["team_prompt_versions"][-1],
                "n_texts": len(meta), "sources": meta.source.value_counts().to_dict(), "files": {}}
    output = ROOT / "assets/dev_dashboard.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, frame in frames.items():
            content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            bundle.writestr("derived/" + name, content)
            manifest["files"][name] = {"rows": len(frame), "sha256": hashlib.sha256(content).hexdigest()}
        bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"{output.name}: {len(meta):,} dev texts, {output.stat().st_size:,} bytes; no source text/context/examples")


if __name__ == "__main__":
    main()
