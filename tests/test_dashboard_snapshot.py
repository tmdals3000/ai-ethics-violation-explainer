"""A clone must open actual saved dev results without source text or model inference."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from src import dashboard_snapshot, effects, judges, retrieval
from src.common import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_contains_only_dev_metadata_and_matches_manifest():
    with ZipFile(ROOT / "assets/dev_dashboard.zip") as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        for name in dashboard_snapshot.FILES:
            content = bundle.read("derived/" + name)
            assert hashlib.sha256(content).hexdigest() == manifest["files"][name]["sha256"]
            assert len(pd.read_csv(io.BytesIO(content))) == manifest["files"][name]["rows"]
        texts = pd.read_csv(io.BytesIO(bundle.read("derived/texts.csv")), keep_default_na=False)
        assert len(texts) == manifest["n_texts"] == 3539
        assert set(texts.split) == {"dev"} and texts.text_id.is_unique
        assert texts[["text", "context"]].eq("").all().all()
        assert pd.read_csv(io.BytesIO(bundle.read("derived/examples.csv"))).empty
        for name in ("dev_sample", "splits", "coords", "similarity"):
            other = pd.read_csv(io.BytesIO(bundle.read(f"derived/{name}.csv")))
            assert set(other.text_id) == set(texts.text_id)


def test_dashboard_fallback_ignores_machine_paths_and_is_opt_in_for_pipeline(monkeypatch):
    monkeypatch.setenv("AIETHICS_DASHBOARD_SNAPSHOT", "1")
    cfg = load_config(dashboard=True)
    assert cfg["_dashboard_snapshot"]
    assert cfg["_data"].resolve() == (ROOT / "data/dashboard_snapshot").resolve()
    assert (cfg["_data"] / "derived/texts.csv").is_file()
    assert "_dashboard_snapshot" not in load_config()


def test_snapshot_rejects_uncached_candidates_before_model_import(monkeypatch):
    cfg = dict(_dashboard_snapshot=True, judge=dict(team_prompt_versions=["v"], aggregation="max"),
               retrieval=dict(tau=.2))
    meta = pd.DataFrame(dict(text_id=["a"], split=["dev"], source=["s"], human_label=[0]))
    baseline = meta.assign(model="team", condition="base", threshold=.5, prediction=0, p_violation=.3)
    raw = pd.DataFrame(dict(text_id=["a"], model=["team"], condition=["base"], prompt_version=["v"],
                            candidate_rule_id=["A"], p_violation=[.3]))
    monkeypatch.setattr(retrieval, "Tree", lambda _: None)
    monkeypatch.setattr(retrieval, "retrieve_all", lambda *args: pd.DataFrame(dict(text_id=["a"], candidates=["B"])))
    monkeypatch.setattr(effects, "base_items", lambda *args: [dict(text_id="a", condition="base", candidate_rule_id="B")])
    monkeypatch.setattr(judges, "score_rule_model", lambda *args: pytest.fail("Snapshot must not call model inference"))
    nodes = pd.DataFrame(dict(node_id=["A", "B"], rule_text=["ra", "rb"]))
    with pytest.raises(ValueError, match="저장되지 않은"):
        effects.live_settings(cfg, meta, nodes, {}, baseline, raw, "team", .1, "max")
    with pytest.raises(ValueError, match="원문"):
        effects.rule_edit(cfg, "A", "new rule")
