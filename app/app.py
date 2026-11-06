"""최소 화면: 규칙 찾기 · 판정 비교 · 오류 진단 · 수정 효과.
실행: streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adapters, data, diagnose, effects, retrieval  # noqa: E402
from src.common import load_config, read_csv_safe  # noqa: E402
from src.metrics import MODELS, binary_metrics, load_joined  # noqa: E402

st.set_page_config(page_title="AI 윤리 판정 설명", layout="wide")
cfg = load_config()
RES = cfg["_res"]


csv = read_csv_safe


@st.cache_data
def load_all():
    texts, nodes, _, sample = data.load(cfg)
    tgt = retrieval.targets(cfg, texts, sample)
    sim, sim_tmp = adapters.read(cfg, "similarity")
    coords, co_tmp = adapters.read(cfg, "coords")
    _, sp_tmp = adapters.read(cfg, "splits")
    m, j_tmp = load_joined(cfg) if (RES / "judgments.csv").exists() or adapters.has_ta(cfg, "judgments") else (pd.DataFrame(), True)
    per = pd.read_csv(RES / "p1/retrieval_per_text.csv", keep_default_na=False)
    status = csv(RES / "model_status.csv")
    return tgt, nodes, sim, coords, m, per, status, {"split": sp_tmp, "similarity": sim_tmp, "coords": co_tmp, "judgments": j_tmp}


tgt, nodes, sim, coords, M, PER, STATUS, TMP = load_all()
SW = retrieval.wide(sim)
TREE = retrieval.Tree(nodes)
PARENT = dict(zip(nodes.node_id, nodes.parent_id))
tgt["rule_group"] = [PARENT.get(g, g) if s == "do_not_answer" else (g or "(none)") for s, g in zip(tgt.source, tgt.gold_rule_id)]

# ---------- 상단: 임시 데이터·모델 상태 표시 ----------
st.title("텍스트 AI 윤리 규칙 위반 판정 설명")
tmp = [k for k, v in TMP.items() if v]
if tmp:
    st.warning(f"⚠ 임시 데이터 사용 중 (조교 파일 아님): {', '.join(tmp)} · split 은 고정 시드 temporary")
if len(STATUS):
    bad = STATUS[STATUS.status != "ok"]
    if len(bad):
        st.error("실행하지 못한 판정 모델: " + ", ".join(f"{r.model}({r.status})" for r in bad.itertuples()))

# ---------- 공통 필터 ----------
sb = st.sidebar
sb.header("공통 필터")
f_src = sb.multiselect("출처", sorted(tgt.source.unique()), default=sorted(tgt.source.unique()))
f_split = sb.multiselect("분할", sorted(tgt.split.unique()), default=sorted(tgt.split.unique()))
f_rule = sb.multiselect("규칙(범주)", sorted(tgt.rule_group.unique()), default=[])
models_avail = [x for x in MODELS if len(M) and x in set(M.model)]
f_model = sb.selectbox("판정 모델", models_avail or ["(없음)"])
f_out = sb.radio("판정 결과", ["전체", "맞음", "틀림"], horizontal=True)

sel_ids = tgt[tgt.source.isin(f_src) & tgt.split.isin(f_split) & (tgt.rule_group.isin(f_rule) if f_rule else True)].text_id
if f_out != "전체" and len(M):
    b = M[(M.model == f_model) & (M.condition == "base")]
    ok = set(b[b.correct == (1 if f_out == "맞음" else 0)].text_id)
    sel_ids = sel_ids[sel_ids.isin(ok)]
IDS = set(sel_ids)
sb.caption(f"필터 결과: 텍스트 {len(IDS)}건")

def heat_bad(v):  # 0→흰색, 1→빨강
    return "" if v != v else f"background-color: rgba(220,40,40,{v:.2f})"


def heat_good(v):  # 낮을수록 빨강 (자주 틀리는 가지)
    return "" if v != v else f"background-color: rgba(220,40,40,{1 - v:.2f})"


# ---------- 선택된 텍스트 (모든 탭 공통) ----------
SEL = st.session_state.get("sel")


def pick(df, key):
    """표에서 한 행을 고르면 모든 탭의 선택 텍스트가 된다. 선택 텍스트는 노란색."""
    sty = df.style.apply(lambda r: ["background-color: #fff3a0" if r.get("text_id") == SEL else "" for _ in r], axis=1)
    ev = st.dataframe(sty, key=key, on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch")
    rows = ev.selection.rows if ev and ev.selection else []
    if rows and "text_id" in df and df.iloc[rows[0]].text_id != SEL:
        st.session_state.sel = df.iloc[rows[0]].text_id
        st.rerun()


with st.expander(f"선택한 텍스트: {SEL or '(없음) — 아래 표에서 한 행을 고르세요'}", expanded=bool(SEL)):
    if SEL and SEL in set(tgt.text_id):
        r = tgt.set_index("text_id").loc[SEL]
        c1, c2, c3 = st.columns(3)
        c1.metric("출처", r.source)
        c2.metric("사람 라벨", "위반" if r.human_label else "정상")
        c3.metric("정답 규칙", r.gold_rule_id or "-")
        if st.toggle("원문 보기 (유해할 수 있는 문장입니다)", key=f"show_{SEL}"):
            st.text_area("맥락", r.context, disabled=True)
            st.text_area("텍스트", r.text, disabled=True)
        else:
            st.caption("원문은 기본으로 가려져 있습니다.")

t1, t2, t3, t4 = st.tabs(["P1 규칙 찾기", "P2 판정 비교", "P3 오류 진단", "P4 수정 효과"])

# ---------- P1 ----------
with t1:
    tau = st.slider("τ (중분류 임계값)", 0.0, 0.6, float(cfg["retrieval"]["tau"]), 0.01)
    sub = tgt[tgt.text_id.isin(IDS)]
    per = retrieval.retrieve_all(cfg, sub, TREE, SW, tau) if len(sub) else PER.iloc[:0]
    if len(per):
        st.dataframe(retrieval.summarize(per), hide_index=True)
    cols = [c for c in ["text_id", "source", "gold_rule_id", "top1", "hit1", "hit3", "gold_rank", "first_divergence",
                        "boundary", "min_gap_to_tau", "efficiency"] if c in per]
    a, b = st.columns([3, 2])
    with a:
        st.subheader("텍스트별 결과")
        pick(per[cols], "p1_tbl")
    with b:
        st.subheader("선택 텍스트의 단계별 유사도")
        if SEL in SW:
            s = SW[SEL]
            res = retrieval.bfs(s, TREE, tau, tuple(cfg["retrieval"]["tau_levels"]))
            gold = set(tgt.set_index("text_id").loc[SEL].gold_path.split(">")) - {""}
            bfs_path = set(TREE.path(res["ranking"][0])) if res["ranking"] else set()
            kept = {n for v in res["kept"].values() for n in v}
            show = nodes[nodes.node_id.isin(kept | gold | bfs_path)][["node_id", "level", "parent_id", "name"]].copy()
            show["sim"] = show.node_id.map(s).round(3)
            show["BFS유지"] = show.node_id.isin(kept)
            show["BFS경로"] = show.node_id.isin(bfs_path)
            show["정답경로"] = show.node_id.isin(gold)
            show = show.sort_values(["level", "sim"], ascending=[True, False])
            st.caption(f"τ = {tau:.2f} · 처음 갈라진 단계: "
                       f"{retrieval.first_divergence(TREE.path(res['ranking'][0]), sorted(gold, key=lambda n: TREE.level[n])) if gold else '-'}")
            st.dataframe(show.style.apply(lambda r: ["background-color: #ffd6d6" if r["BFS경로"] != r["정답경로"] else "" for _ in r], axis=1),
                         hide_index=True)
    st.subheader("경계 사례 (τ±δ 에서 top-1 이 바뀜)")
    pick(per[per.boundary][cols] if len(per) else per, "p1_boundary")
    st.subheader("규칙 노드")
    nid = st.selectbox("노드", nodes.node_id, format_func=lambda n: f"{n} · {nodes.set_index('node_id').name[n]}")
    st.write(nodes.set_index("node_id").rule_text[nid])
    c1, c2 = st.columns(2)
    c1.caption("이 규칙이 정답인 텍스트")
    c1.dataframe(per[per.get("gold_rule_id", pd.Series(dtype=str)) == nid][["text_id", "top1", "hit1"]] if "gold_rule_id" in per else per.iloc[:0], hide_index=True)
    c2.caption("이 규칙으로 찾아진 텍스트 (top-1)")
    c2.dataframe(per[per.top1 == nid][["text_id", "source"]], hide_index=True)
    if "gold_rule_id" in per:
        g = per[per.gold_rule_id != ""]
        acc = g.groupby("gold_rule_id").agg(n=("text_id", "size"), top1_acc=("hit1", "mean"), top3_acc=("hit3", "mean")).round(3)
        st.caption("노드별 top-k 정확도 (낮은 순 — 자주 틀리는 가지)")
        st.dataframe(acc.sort_values("top1_acc").style.map(heat_good, subset=["top1_acc", "top3_acc"]))

# ---------- P2 ----------
with t2:
    if not len(M):
        st.info("판정 결과가 없습니다. scripts/run.py --steps judge 를 먼저 실행하세요.")
    else:
        mm = M[M.text_id.isin(IDS)]
        cond = st.selectbox("조건", ["base", "rule_excluded", "rule_only", "rule_plus_examples"])
        x = mm[mm.condition == cond]
        met = pd.DataFrame([{"source": s, "model": mo} | binary_metrics(g) for (s, mo), g in x.groupby(["source", "model"])])
        st.dataframe(met, hide_index=True)
        by = st.radio("행렬 묶음 기준", ["rule_group", "response_llm", "source"], horizontal=True)
        xb = x if by != "response_llm" else x[x.source == "do_not_answer"]
        cell = xb.groupby(["model", by]).agg(n=("correct", "size"), wrong=("correct", lambda c: 1 - c.mean())).reset_index()
        piv = cell.pivot(index="model", columns=by, values="wrong")
        nn = cell.pivot(index="model", columns=by, values="n")
        st.caption("칸 = 잘못된 판정 비율. 건수 < 5 인 칸은 흐리게.")
        st.dataframe(piv.style.format("{:.2f}").map(heat_bad)
                     .apply(lambda _: nn.map(lambda n: "opacity: 0.35" if n < 5 else ""), axis=None))
        st.dataframe(nn, width="stretch")
        c1, c2 = st.columns(2)
        cm = c1.selectbox("칸 선택: 모델", sorted(cell.model.unique()))
        cg = c2.selectbox("칸 선택: 묶음", sorted(cell[by].astype(str).unique()))
        ids = set(xb[(xb.model == cm) & (xb[by].astype(str) == cg)].text_id)
        probs = xb[xb.text_id.isin(ids)].pivot_table(index="text_id", columns="model", values="p_violation").round(3).reset_index()
        pick(probs.merge(tgt[["text_id", "human_label"]], on="text_id"), "p2_cell")
        if len(models_avail) >= 2:
            a, b = st.columns(2)
            ma = a.selectbox("모델 A", models_avail, 0)
            mb = b.selectbox("모델 B", models_avail, 1)
            pp = x.pivot_table(index="text_id", columns="model", values="p_violation")[[ma, mb]].dropna().reset_index()
            pp["diff"] = (pp[ma] - pp[mb]).abs().round(3)
            st.altair_chart(alt.Chart(pp).mark_circle().encode(x=ma, y=mb, color=alt.condition(alt.datum.text_id == (SEL or ""), alt.value("orange"), alt.value("steelblue")),
                                                               tooltip=["text_id", ma, mb]), width="stretch")
            st.caption("대각선에서 먼 텍스트 = 두 모델의 판단이 엇갈림")
            pick(pp.sort_values("diff", ascending=False).head(20), "p2_pair")
        kap = csv(RES / "p2/pairwise_kappa.csv")
        st.caption("모델 쌍 κ (base)" if len(kap) else "모델 쌍 κ: 판정 모델이 2개 이상 있어야 계산됩니다.")
        if len(kap):
            st.dataframe(kap, hide_index=True)

# ---------- P3 ----------
with t3:
    if not len(M):
        st.info("판정 결과가 없습니다.")
    else:
        k = st.slider("'규칙 찾기 문제' 로 볼 상위 k", 1, 10, int(cfg["diagnose"]["retrieval_top_k"]))
        d = diagnose.diagnose(M[M.text_id.isin(IDS)], PER, k)
        d = d[d.model == f_model] if f_model in set(d.model) else d
        cnt = d.cause.value_counts().reindex(diagnose.CAUSES, fill_value=0).rename("n").reset_index()
        st.caption(f"모델 {f_model} · 잘못된 판정 {len(d)}건 · unclassified {int((d.cause == 'unclassified').sum())}건")
        if len(d) and (~d.rule2_applicable).any():
            st.warning(f"결과가 빠진 판정 모델이 있어 {int((~d.rule2_applicable).sum())}건은 '라벨 확인 필요'를 적용하지 않았습니다 "
                       f"(빠진 모델: {', '.join(sorted({x for v in d.missing_models for x in v.split('|') if x}))}).")
        st.altair_chart(alt.Chart(cnt).mark_bar().encode(x="n", y=alt.Y("cause", sort=diagnose.CAUSES)), width="stretch")
        focus = st.selectbox("강조할 원인", ["(전체)"] + diagnose.CAUSES)
        if coords is not None:
            pts = coords.merge(tgt[["text_id", "source"]], on="text_id")
            pts = pts[pts.text_id.isin(IDS)].merge(d[["text_id", "cause"]], on="text_id", how="left").fillna({"cause": "맞음/진단대상 아님"})
            pts["강조"] = (pts.cause == focus) | (focus == "(전체)") | (pts.text_id == SEL)
            brush = alt.selection_interval(name="brush")
            ch = alt.Chart(pts).mark_circle(size=70).encode(
                x="x", y="y", color="cause", opacity=alt.condition(alt.datum["강조"], alt.value(1), alt.value(0.15)),
                tooltip=["text_id", "source", "cause"]).add_params(brush)
            ev = st.altair_chart(ch, on_select="rerun", key="p3_map", width="stretch")
            box = ev.selection.get("brush") if ev and ev.selection else None
            if box and "x" in box:
                inb = pts[pts.x.between(*box["x"]) & pts.y.between(*box["y"])]
                st.caption(f"영역 선택 {len(inb)}건: 원인 분포 / 자주 나온 규칙")
                c1, c2 = st.columns(2)
                c1.dataframe(inb.cause.value_counts())
                c2.dataframe(tgt[tgt.text_id.isin(inb.text_id)].rule_group.value_counts())
            st.caption("coords 는 임시(PCA 2D)" if TMP["coords"] else "")
        pick(d[(d.cause == focus) | (focus == "(전체)")], "p3_tbl")
        if SEL:
            st.subheader(f"진단: {SEL}")
            pr = PER.set_index("text_id")
            if SEL in pr.index:
                st.write(f"찾은 규칙 top-5: `{pr.loc[SEL, 'candidates']}` · 정답 규칙: `{tgt.set_index('text_id').gold_rule_id.get(SEL, '')}`")
            st.dataframe(M[M.text_id == SEL].pivot_table(index="model", columns="condition", values="p_violation").round(3))
            dd = d[d.text_id == SEL]
            if len(dd):
                st.dataframe(dd[["model", "cause", "decided_by", "missing_models"]], hide_index=True)

# ---------- P4 ----------
with t4:
    if not len(M):
        st.info("판정 결과가 없습니다.")
    else:
        st.subheader("판정 임계값 바꾸기")
        cur = M[(M.model == f_model) & (M.condition == "base") & M.text_id.isin(IDS)]
        th0 = float(cur.threshold.iloc[0]) if len(cur) else 0.5
        if st.button("원래 값으로 되돌리기"):
            st.session_state["th_alt"] = th0
        th = st.slider("새 임계값", 0.0, 1.0, th0, 0.01, key="th_alt")
        if len(cur):
            meta = tgt[["text_id", "source", "human_label"]]
            after = cur.assign(prediction=(cur.p_violation >= th).astype(int))
            res = effects.compare("live", {}, cur[["text_id", "p_violation", "prediction"]], after[["text_id", "p_violation", "prediction"]], meta)
            c1, c2 = st.columns(2)
            c1.caption(f"원래 (임계값 {th0})")
            c1.dataframe(pd.DataFrame(res["metrics_before"]).T[["n", "roc_auc", "f1", "kappa_human", "error_rate"]])
            c2.caption(f"바꾼 설정 (임계값 {th})")
            c2.dataframe(pd.DataFrame(res["metrics_after"]).T[["n", "roc_auc", "f1", "kappa_human", "error_rate"]])
            st.write(f"뒤집힌 판정 {res['n_flipped']} · 맞게 바뀜 **{res['n_fixed']}** · 틀리게 바뀜 **{res['n_broken']}**")
            fl = pd.DataFrame(res["flipped"])
            if len(fl):
                pick(fl, "p4_flip")
        st.subheader("처음 보는 규칙: AUC 하락폭·회복폭")
        p = RES / "p4/auc_drop_recovery.csv"
        adr = csv(p)
        if len(adr):
            g = st.selectbox("규칙 제외 실험의 규칙", sorted(adr.group.astype(str).unique()))
            st.dataframe(adr[adr.group.astype(str) == g], hide_index=True)
        else:
            st.info("규칙을 받는 모델(ShieldGemma/팀) 결과가 없어 계산되지 않았습니다.")
        st.subheader("규칙 문장 고치고 다시 판정")
        leaf = nodes[~nodes.node_id.isin(set(nodes.parent_id))]
        nid = st.selectbox("고칠 규칙", leaf.node_id, key="edit_node")
        new = st.text_area("새 규칙 문장", leaf.set_index("node_id").rule_text[nid], key=f"edit_{nid}")
        if st.button("다시 판정 (유사도 재계산 + 팀 판정기, GPU 1~3분)"):
            if "team" not in models_avail:
                st.error("팀 판정기(ShieldGemma) 결과가 없어 다시 판정할 수 없습니다.")
            else:
                with st.spinner("다시 판정 중..."):
                    prop = effects.rule_edit(cfg, nid, new)
                    prop["name"] = f"rule_edit_{nid}_app"
                    effects.save(cfg, prop)
                st.success(f"고쳐진 판정 {prop['n_fixed']} · 새로 틀린 판정 {prop['n_broken']} → results/rule_proposals/ 에 저장")
        st.subheader("저장된 수정안")
        for f in sorted((RES / "rule_proposals").glob("*.json")):
            pr = json.loads(f.read_text(encoding="utf-8"))
            with st.expander(f"{f.stem} · 뒤집힘 {pr['n_flipped']} (맞게 {pr['n_fixed']}/틀리게 {pr['n_broken']})"):
                st.json(pr["change"])
                st.dataframe(pd.DataFrame({"before": pr["metrics_before"], "after": pr["metrics_after"]}).T)
