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
from src import adapters, data, diagnose, effects, retrieval, visuals  # noqa: E402
from src.common import load_config, read_csv_safe  # noqa: E402
from src.metrics import MODELS, binary_metrics, load_joined, pairwise_kappa  # noqa: E402

st.set_page_config(page_title="AI 윤리 판정 설명", layout="wide")
cfg = load_config()
RES = cfg["_res"]
PV = cfg["judge"]["team_prompt_versions"][-1]
FILES = [RES / "judgments.csv", RES / "p1/retrieval_per_text.csv", RES / "model_status.csv"]
FILES += list((cfg["_data"] / "derived").glob("*.csv"))
STAMP = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in FILES if p.exists())


csv = read_csv_safe


@st.cache_data
def load_all(stamp, prompt_version):
    texts, nodes, _, sample = data.load(cfg)
    tgt = retrieval.targets(cfg, texts, sample)
    sim, sim_tmp = adapters.read(cfg, "similarity")
    coords, co_tmp = adapters.read(cfg, "coords")
    _, sp_tmp = adapters.read(cfg, "splits")
    m, j_tmp = load_joined(cfg) if (RES / "judgments.csv").exists() or adapters.has_ta(cfg, "judgments") else (pd.DataFrame(), True)
    per = pd.read_csv(RES / "p1/retrieval_per_text.csv", keep_default_na=False)
    status = csv(RES / "model_status.csv")
    return tgt, nodes, sim, coords, m, per, status, {"split": sp_tmp, "similarity": sim_tmp, "coords": co_tmp, "judgments": j_tmp}


tgt, nodes, sim, coords, M, PER, STATUS, TMP = load_all(STAMP, PV)
SW = retrieval.wide(sim)
TREE = retrieval.Tree(nodes)
PARENT = dict(zip(nodes.node_id, nodes.parent_id))
tgt["rule_group"] = [PARENT.get(g, g) if s == "do_not_answer" else (g or "(none)") for s, g in zip(tgt.source, tgt.gold_rule_id)]

# ---------- 상단: 임시 데이터·모델 상태 표시 ----------
st.title("텍스트 AI 윤리 규칙 위반 판정 설명")
st.caption(f"팀 판정기: {PV} · 설정 탐색과 규칙 수정: dev · 최종 test 평가는 별도 실행")
if len(M) and "team" not in set(M.model):
    st.info(f"{PV}의 판정이 아직 저장되지 않았습니다. judge --models team 실행이 끝나면 새로고침하세요.")
tmp = [k for k, v in TMP.items() if v]
if tmp:
    st.warning(f"⚠ 임시 데이터 사용 중 (조교 파일 아님): {', '.join(tmp)} · split 은 고정 시드 temporary")
if len(STATUS):
    bad = STATUS[~STATUS.status.isin(["ok", "ta_file"])]
    if len(bad):
        st.error("실행하지 못한 판정 모델: " + ", ".join(f"{r.model}({r.status})" for r in bad.itertuples()))

# ---------- 공통 필터 ----------
sb = st.sidebar
sb.header("공통 필터")
f_src = sb.multiselect("출처", sorted(tgt.source.unique()), default=sorted(tgt.source.unique()))
f_split = sb.multiselect("분할", sorted(tgt.split.unique()), default=sorted(tgt.split.unique()))
f_rule = sb.multiselect("규칙(범주)", sorted(tgt.rule_group.unique()), default=[])
models_avail = [x for x in MODELS if len(M) and x in set(M.model)]
f_model = sb.selectbox("판정 모델", models_avail or ["(없음)"], index=models_avail.index("team") if "team" in models_avail else 0)
f_out = sb.radio("판정 결과", ["전체", "맞음", "틀림"], horizontal=True)

sel_ids = tgt[tgt.source.isin(f_src) & tgt.split.isin(f_split) & (tgt.rule_group.isin(f_rule) if f_rule else True)].text_id
if f_out != "전체" and len(M):
    b = M[(M.model == f_model) & (M.condition == "base")]
    ok = set(b[b.correct == (1 if f_out == "맞음" else 0)].text_id)
    sel_ids = sel_ids[sel_ids.isin(ok)]
IDS = set(sel_ids)
sb.caption(f"필터 결과: 텍스트 {len(IDS)}건")
if st.session_state.get("sel") not in IDS:
    st.session_state.sel = None
if not IDS:
    st.info("현재 필터에 해당하는 텍스트가 없습니다. 공통 필터를 바꿔 주세요.")
    st.stop()

def heat_bad(v):  # 0→흰색, 1→빨강
    return "" if v != v else f"background-color: rgba(220,40,40,{v:.2f})"


def heat_good(v):  # 낮을수록 빨강 (자주 틀리는 가지)
    return "" if v != v else f"background-color: rgba(220,40,40,{1 - v:.2f})"


# ---------- 선택된 텍스트 (모든 탭 공통) ----------
SEL = st.session_state.get("sel")


def chart_selection(ev, param, field, key, target="sel", popup=False):
    """새 클릭만 반영해, 다른 뷰에서 바꾼 선택을 오래된 이벤트가 덮지 않게 한다."""
    value = visuals.selection_value(ev, param, field)
    previous = st.session_state.get(f"seen_{key}")
    st.session_state[f"seen_{key}"] = value
    if value is not None and value != previous:
        st.session_state[target] = value
        if popup:
            st.session_state.open_diagnosis = True
        st.rerun()


def pick(df, key):
    """표에서 한 행을 고르면 모든 탭의 선택 텍스트가 된다. 선택 텍스트는 노란색."""
    sty = df.style.apply(lambda r: ["background-color: #fff3a0" if r.get("text_id") == SEL else "" for _ in r], axis=1)
    ev = st.dataframe(sty, key=key, on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch")
    rows = ev.selection.rows if ev and ev.selection else []
    value = str(df.iloc[rows[0]].text_id) if rows and "text_id" in df else None
    chart_selection({"selection": {"row": {"text_id": [value] if value else []}}}, "row", "text_id", key)


def effect_metrics(prop):
    rows = [dict(setting=setting, source=src, **values) for setting in ("metrics_before", "metrics_after")
            for src, values in prop.get(setting, {}).items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)


@st.cache_data(show_spinner=False)
def settings_result(stamp, prompt_version, ids, model, tau, aggregation, threshold, _texts, _m, _raw):
    return effects.live_settings(cfg, _texts, nodes, SW, _m, _raw, model, tau, aggregation, threshold)


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
    st.subheader("규칙 트리 · BFS 경로와 정답 경로")
    selected_sim = SW.get(SEL, {})
    selected_result = retrieval.bfs(selected_sim, TREE, tau, tuple(cfg["retrieval"]["tau_levels"])) if selected_sim else {}
    predicted = TREE.path(selected_result["ranking"][0]) if selected_result.get("ranking") else []
    gold = tgt.set_index("text_id").gold_path.get(SEL, "").split(">") if SEL else []
    gold = [n for n in gold if n]
    divergence = next((p for p, g in zip(predicted, gold) if p != g), "")
    points, edges = visuals.tree_data(nodes, per, selected_sim, predicted, gold, divergence)
    color_mode = st.radio("노드 색", ["정답 가지별 top-1 정확도", "단계별 유사도"], horizontal=True)
    path_scale = alt.Scale(domain=["기타", "BFS", "정답", "BFS + 정답"], range=["#d5dae0", "#f08c00", "#1971c2", "#9c36b5"])
    links = alt.Chart(edges).mark_rule(strokeWidth=2).encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), x2="x2:Q", y2="y2:Q", color=alt.Color("path:N", scale=path_scale, legend=None))
    node_color = alt.Color("similarity:Q", scale=alt.Scale(scheme="blues"), title="유사도") if color_mode == "단계별 유사도" else alt.Color("top1_acc:Q", scale=alt.Scale(domain=[0, 1], scheme="redyellowgreen"), title="top-1 정확도")
    node_pick = alt.selection_point(name="rule_node", fields=["node_id"], toggle=False)
    dots = alt.Chart(points).mark_point(filled=True, size=130, strokeWidth=3).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0.8, 4.2])), y=alt.Y("y:Q", axis=None), color=node_color,
        stroke=alt.Stroke("path:N", scale=path_scale, legend=None),
        shape=alt.condition(alt.datum.divergence, alt.value("square"), alt.value("circle")),
        tooltip=["node_id", "name", "similarity", "path", "n", "top1_acc", "top3_acc"]).add_params(node_pick)
    labels = alt.Chart(points).mark_text(align="left", dx=10, fontSize=10).encode(x="x:Q", y="y:Q", text="label:N")
    chart = (links + dots + labels).resolve_scale(color="independent").properties(height=max(450, int((~nodes.node_id.isin(nodes.parent_id)).sum()) * 15))
    ev = st.altair_chart(chart, on_select="rerun", selection_mode="rule_node", key="p1_tree", width="stretch")
    chart_selection(ev, "rule_node", "node_id", "p1_tree", target="node_choice")
    st.caption(f"τ={tau:.2f} · 테두리 주황: BFS top-1 경로 / 파랑: 정답 / 보라: 공통 / 사각형: 처음 갈라진 노드. 노드를 클릭해 아래 상세를 확인하세요.")
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
    nid = st.selectbox("노드", nodes.node_id, key="node_choice", format_func=lambda n: f"{n} · {nodes.set_index('node_id').name[n]}")
    st.write(nodes.set_index("node_id").rule_text[nid])
    c1, c2 = st.columns(2)
    c1.caption("이 규칙이 정답인 텍스트")
    c1.dataframe(per[per.gold_rule_id.fillna("").isin(TREE.subtree(nid))][["text_id", "top1", "hit1"]] if "gold_rule_id" in per else per.iloc[:0], hide_index=True)
    c2.caption("이 규칙으로 찾아진 텍스트 (top-1)")
    c2.dataframe(per[per.top1.isin(TREE.subtree(nid))][["text_id", "source"]] if len(per) else per, hide_index=True)
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
        st.dataframe(met[met.model == f_model] if len(met) else met, hide_index=True)
        st.caption("선택한 판정 모델의 지표입니다. 아래 행렬과 산점도는 같은 텍스트에서 기준 모델도 함께 비교합니다.")
        by = st.radio("행렬 묶음 기준", ["rule_group", "response_llm", "source"], horizontal=True)
        xb = x if by != "response_llm" else x[x.source == "do_not_answer"]
        cell = xb.groupby(["model", by]).agg(n=("correct", "size"), wrong=("correct", lambda c: 1 - c.mean())).reset_index()
        cell["group"] = cell[by].astype(str)
        cell["label"] = [f"{w:.0%} ({round(w*n)}/{n})" for w, n in zip(cell.wrong, cell.n)]
        st.caption("칸 = 오류율 (오류/전체 건수). 건수 < 5인 칸은 흐리게. 칸을 누르면 아래 목록이 바뀝니다.")
        cell_pick = alt.selection_point(name="cell_pick", fields=["model", "group"], toggle=False)
        base_chart = alt.Chart(cell).encode(x=alt.X("group:N", title=by), y="model:N")
        boxes = base_chart.mark_rect(stroke="white").encode(color=alt.Color("wrong:Q", scale=alt.Scale(domain=[0, 1], scheme="reds")),
                    stroke=alt.condition(alt.datum.model == f_model, alt.value("#343a40"), alt.value("white")),
                    opacity=alt.condition(alt.datum.n < 5, alt.value(0.3), alt.value(1)), tooltip=["model", "group", "wrong", "n"]).add_params(cell_pick)
        labels = base_chart.mark_text(fontSize=11).encode(text="label:N", color=alt.condition(alt.datum.wrong > 0.6, alt.value("white"), alt.value("black")))
        ev = st.altair_chart(boxes + labels, on_select="rerun", selection_mode="cell_pick", key="p2_matrix", width="stretch")
        clicked_model = visuals.selection_value(ev, "cell_pick", "model")
        clicked_group = visuals.selection_value(ev, "cell_pick", "group")
        c1, c2 = st.columns(2)
        cm = c1.selectbox("칸 선택: 모델", sorted(cell.model.unique()))
        cg = c2.selectbox("칸 선택: 묶음", sorted(cell[by].astype(str).unique()))
        if clicked_model and clicked_group:
            cm, cg = clicked_model, clicked_group
            st.caption(f"클릭한 칸: {cm} · {cg}")
        ids = set(xb[(xb.model == cm) & (xb[by].astype(str) == cg)].text_id)
        probs = xb[xb.text_id.isin(ids)].pivot_table(index="text_id", columns="model", values="p_violation").round(3).reset_index()
        pick(probs.merge(tgt[["text_id", "human_label"]], on="text_id"), "p2_cell")
        if len(models_avail) >= 2:
            a, b = st.columns(2)
            ma = a.selectbox("모델 A", models_avail, models_avail.index(f_model))
            mb = b.selectbox("모델 B", [mo for mo in models_avail if mo != ma])
            pp = x.pivot_table(index="text_id", columns="model", values="p_violation").reindex(columns=[ma, mb]).dropna().reset_index()
            pp["diff"] = (pp[ma] - pp[mb]).abs().round(3)
            plotted = pp.rename(columns={ma: "prob_a", mb: "prob_b"})
            pair_pick = alt.selection_point(name="pair_text", fields=["text_id"], toggle=False, nearest=True)
            dots = alt.Chart(plotted).mark_circle().encode(x=alt.X("prob_a:Q", title=ma, scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("prob_b:Q", title=mb, scale=alt.Scale(domain=[0, 1])),
                size=alt.condition(alt.datum.text_id == (SEL or ""), alt.value(180), alt.value(45)),
                color=alt.condition(alt.datum.text_id == (SEL or ""), alt.value("orange"), alt.value("steelblue")),
                tooltip=["text_id", "prob_a", "prob_b", "diff"]).add_params(pair_pick)
            diagonal = alt.Chart(pd.DataFrame({"prob_a": [0, 1], "prob_b": [0, 1]})).mark_line(strokeDash=[4, 4], color="#868e96").encode(x="prob_a:Q", y="prob_b:Q")
            ev = st.altair_chart(diagonal + dots, on_select="rerun", selection_mode="pair_text", key="p2_scatter", width="stretch")
            chart_selection(ev, "pair_text", "text_id", "p2_scatter")
            st.caption("대각선에서 먼 텍스트 = 두 모델의 판단이 엇갈림")
            pick(pp.sort_values("diff", ascending=False).head(20), "p2_pair")
        kap = pairwise_kappa(mm)
        st.caption("모델 쌍 κ (base)" if len(kap) else "모델 쌍 κ: 판정 모델이 2개 이상 있어야 계산됩니다.")
        if len(kap):
            st.dataframe(kap, hide_index=True)

# ---------- P3 ----------
with t3:
    if not len(M):
        st.info("판정 결과가 없습니다.")
    else:
        k = st.slider("'규칙 찾기 문제' 로 볼 상위 k", 1, 10, int(cfg["diagnose"]["retrieval_top_k"]), key="diagnosis_k")
        d = diagnose.diagnose(M[M.text_id.isin(IDS)], PER, k)
        d = d[d.model == f_model] if f_model in set(d.model) else d
        cnt = d.cause.value_counts().reindex(diagnose.CAUSES, fill_value=0).rename("n").reset_index()
        cnt["ratio"] = cnt.n / max(len(d), 1)
        st.caption(f"모델 {f_model} · 잘못된 판정 {len(d)}건 · unclassified {int((d.cause == 'unclassified').sum())}건")
        if len(d) and (~d.rule2_applicable).any():
            st.warning(f"결과가 빠진 판정 모델이 있어 {int((~d.rule2_applicable).sum())}건은 '라벨 확인 필요'를 적용하지 않았습니다 "
                       f"(빠진 모델: {', '.join(sorted({x for v in d.missing_models for x in v.split('|') if x}))}).")
        cause_pick = alt.selection_point(name="cause_pick", fields=["cause"], toggle=False)
        ev = st.altair_chart(alt.Chart(cnt).mark_bar().encode(x="n:Q", y=alt.Y("cause:N", sort=diagnose.CAUSES), color="cause:N", tooltip=["cause", "n", "ratio"]).add_params(cause_pick),
                              on_select="rerun", selection_mode="cause_pick", key="p3_causes", width="stretch")
        chart_selection(ev, "cause_pick", "cause", "p3_causes", target="cause_focus")
        focus = st.selectbox("강조할 원인", ["(전체)"] + diagnose.CAUSES, key="cause_focus")
        if coords is not None:
            pts = coords.merge(tgt[["text_id", "source"]], on="text_id")
            pts = pts[pts.text_id.isin(IDS)].merge(d[["text_id", "cause"]], on="text_id", how="left").fillna({"cause": "맞음/진단대상 아님"})
            pts["강조"] = (pts.cause == focus) | (focus == "(전체)") | (pts.text_id == SEL)
            brush = alt.selection_interval(name="brush")
            map_pick = alt.selection_point(name="map_text", fields=["text_id"], toggle=False, nearest=True)
            ch = alt.Chart(pts).mark_circle(strokeWidth=2).encode(
                x="x", y="y", color="cause", opacity=alt.condition(alt.datum["강조"], alt.value(1), alt.value(0.15)),
                stroke=alt.condition(alt.datum.text_id == (SEL or ""), alt.value("orange"), alt.value("transparent")),
                size=alt.condition(alt.datum.text_id == (SEL or ""), alt.value(220), alt.value(60)),
                tooltip=["text_id", "source", "cause"]).add_params(brush, map_pick)
            ev = st.altair_chart(ch, on_select="rerun", selection_mode=["brush", "map_text"], key="p3_map", width="stretch")
            chart_selection(ev, "map_text", "text_id", "p3_map", popup=True)
            box = ev.selection.get("brush") if ev and ev.selection else None
            if box and "x" in box and "y" in box:
                inb = pts[pts.x.between(*box["x"]) & pts.y.between(*box["y"])]
                st.caption(f"영역 선택 {len(inb)}건: 원인 분포 / 자주 나온 규칙")
                c1, c2 = st.columns(2)
                c1.dataframe(inb.cause.value_counts())
                c2.dataframe(tgt[tgt.text_id.isin(inb.text_id)].rule_group.value_counts())
            st.caption("coords 는 임시(PCA 2D)" if TMP["coords"] else "")
            st.caption("점 클릭: 텍스트 진단 · 드래그: 영역 분석 · 주황 테두리: 모든 뷰의 공통 선택")
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
            st.caption("‘라벨 확인 필요’는 재검토 후보입니다. 모델이 함께 놓친 유형일 수도 있습니다.")

# ---------- P4 ----------
with t4:
    if not len(M):
        st.info("판정 결과가 없습니다.")
    else:
        st.subheader("판정 임계값 바꾸기")
        cur = M[(M.model == f_model) & (M.condition == "base") & M.text_id.isin(IDS)]
        th0 = float(cur.threshold.iloc[0]) if len(cur) else 0.5
        if st.button("원래 값으로 되돌리기"):
            st.session_state.update(th_alt=th0, th_custom=False, tau_alt=float(cfg["retrieval"]["tau"]), agg_alt=cfg["judge"]["aggregation"])
        custom = st.toggle("판정 임계값 직접 지정", key="th_custom")
        st.session_state.setdefault("th_alt", th0)
        st.session_state.setdefault("tau_alt", float(cfg["retrieval"]["tau"]))
        st.session_state.setdefault("agg_alt", cfg["judge"]["aggregation"])
        th = st.slider("새 임계값", 0.0, 1.0, value=None, step=0.01, key="th_alt", disabled=not custom)
        rule_model = f_model in effects.RULE_MODELS
        tau_alt = st.slider("P4 검색 τ", 0.0, 0.6, value=None, step=0.01, key="tau_alt", disabled=not rule_model)
        agg_alt = st.radio("후보 우선순위", ["max", "top1"], index=None, key="agg_alt", horizontal=True, disabled=not rule_model)
        st.caption("max: 후보의 최대 위반 확률 / top1: 유사도 1위 후보. 임계값 직접 지정을 끄면 출처별 원래 값을 유지합니다. 새 후보만 재판정하고 캐시합니다.")
        if len(cur):
            scope = tgt[tgt.text_id.isin(IDS) & (tgt.split == "dev")]
            raw, _ = adapters.read(cfg, "judgments")
            try:
                with st.spinner("설정 변경 효과 계산 중..."):
                    res = settings_result(STAMP, PV, tuple(sorted(scope.text_id)), f_model, tau_alt, agg_alt,
                                          th if custom else None, scope, M, raw)
                effect_metrics(res)
                st.write(f"뒤집힌 판정 {res['n_flipped']} · 맞게 바뀜 **{res['n_fixed']}** · 틀리게 바뀜 **{res['n_broken']}**")
                fl = pd.DataFrame(res["flipped"])
                if len(fl):
                    pick(fl, "p4_flip")
                if st.button("설정 변경안 저장"):
                    effects.save(cfg, res)
                    st.success("현재 필터 범위의 dev 비교를 저장했습니다.")
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(f"설정 변경 효과를 계산할 수 없습니다: {exc}")
        st.subheader("처음 보는 규칙: AUC 하락폭·회복폭")
        adr = effects.auc_drop_recovery(M[M.text_id.isin(IDS) & (M.model == f_model)])
        if len(adr):
            groups = adr[adr.group_by == "rule_group"]
            if len(groups):
                g = st.selectbox("규칙 제외 실험의 규칙", sorted(groups.group.astype(str).unique()))
                st.dataframe(groups[groups.group.astype(str) == g], hide_index=True)
            st.caption("출처별 합계")
            st.dataframe(adr[adr.group_by == "source"], hide_index=True)
        else:
            st.info("정답 규칙과 네 조건 판정이 있는 DNA·Jigsaw의 ShieldGemma/팀 판정에서 계산합니다.")
        st.subheader("규칙 문장 고치고 다시 판정")
        leaf = nodes[~nodes.node_id.isin(set(nodes.parent_id))]
        nid = st.selectbox("고칠 규칙", leaf.node_id, key="edit_node")
        new = st.text_area("새 규칙 문장", leaf.set_index("node_id").rule_text[nid], key=f"edit_{nid}")
        reason = st.text_input("규칙 수정 근거", placeholder="진단에서 찾은 오류와 수정 이유")
        with st.expander("위반·비위반 예시 추가 (한 줄에 하나)"):
            pos = st.text_area("위반 예시", key="positive_examples")
            neg = st.text_area("비위반 예시", key="negative_examples")
        st.caption("현재 공통 필터의 dev 텍스트를 같은 임계값으로 비교합니다. 결과는 test 미확인 후보로 저장합니다.")
        if st.button("다시 판정 (유사도 재계산 + 팀 판정기, GPU 사용)"):
            if "team" not in models_avail:
                st.error("팀 판정기(ShieldGemma) 결과가 없어 다시 판정할 수 없습니다.")
            else:
                try:
                    examples = pd.DataFrame([(line.strip(), label) for value, label in [(pos, 1), (neg, 0)] for line in value.splitlines() if line.strip()], columns=["text", "label"])
                    with st.spinner("다시 판정 중..."):
                        prop = effects.rule_edit(cfg, nid, new, examples=examples, text_ids=IDS)
                        prop["name"] = f"rule_edit_{nid}_app"
                        prop["evidence"] = reason
                        effects.save(cfg, prop)
                    st.success(f"고쳐진 판정 {prop['n_fixed']} · 새로 틀린 판정 {prop['n_broken']} → results/rule_proposals/ 에 저장")
                    effect_metrics(prop)
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(f"재판정할 수 없습니다: {exc}")
        st.subheader("저장된 수정안")
        for f in sorted((RES / "rule_proposals").glob("*.json")):
            pr = json.loads(f.read_text(encoding="utf-8"))
            counts = visuals.proposal_counts(pr)
            with st.expander(f"{pr.get('title', f.stem)} · 뒤집힘 {counts['n_flipped']} (맞게 {counts['n_fixed']}/틀리게 {counts['n_broken']})"):
                st.caption(f"{pr.get('prompt_version', '과거 실험')} · {pr.get('status', pr.get('split', 'dev'))}")
                st.json(pr.get("change", {}))
                if "metrics_before" in pr:
                    effect_metrics(pr)
                elif pr.get("metrics"):
                    st.dataframe(pd.DataFrame(pr["metrics"]), hide_index=True)


if st.session_state.pop("open_diagnosis", False) and SEL:
    @st.dialog("선택 텍스트 판정 진단", width="large")
    def diagnosis_dialog():
        st.write(SEL)
        selected = M[M.text_id == SEL]
        st.dataframe(selected.pivot_table(index="model", columns="condition", values="p_violation").round(3))
        details = diagnose.diagnose(selected, PER[PER.text_id == SEL], st.session_state.get("diagnosis_k", cfg["diagnose"]["retrieval_top_k"]))
        st.dataframe(details[["model", "cause", "decided_by", "gold_rule_id"]], hide_index=True)
        st.caption("원문은 상단의 ‘원문 보기’에서 확인할 수 있습니다.")
    diagnosis_dialog()
