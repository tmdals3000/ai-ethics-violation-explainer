"""
5단계: 화면 (Streamlit) — 탭 4개 (P1~P4) + 공통 필터 + 선택 연동 + 원문 가리기

실행: streamlit run app.py
입력: texts.csv, rules.csv, similarity.csv, coords.csv, results/ (analyze.py 결과)
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score
from retrieval import Tree, TAU, TOP_K, gold_leaf
from diagnose import diagnose, CAUSES, takes_rules, is_base

st.set_page_config(page_title="AI 윤리 판정 설명", layout="wide")
MIN_N = 10
SRC_NAME = {"do_not_answer": "Do-Not-Answer", "jigsaw": "Jigsaw", "prosocial": "ProsocialDialog"}
CAUSE_COLORS = {"1. 규칙 찾기 문제": "#d1495b", "2. 라벨 확인 필요": "#edae49",
                "3. 규칙 문장 문제": "#00798c", "4. 판정 모델 문제": "#7b4fa0"}


@st.cache_data
def load():
    t = pd.read_csv("texts.csv", keep_default_na=False)
    df = t[t["sample"] == 1].reset_index(drop=True)
    df["gold"] = df.apply(gold_leaf, axis=1)
    rules = pd.read_csv("rules.csv")
    sim = pd.read_csv("similarity.csv").set_index("text_id").loc[df["text_id"]]
    j = pd.read_csv("results/judgments_all.csv")
    wide = j.pivot_table(index="text_id", columns="judge_cond", values="p_violation")
    thr = pd.read_csv("results/thresholds.csv")
    coords = pd.read_csv("coords.csv")
    tau_sweep = pd.read_csv("results/p1_tau_sweep.csv")
    bc = pd.read_csv("base_candidates.csv") if os.path.exists("base_candidates.csv") else None
    extra = {n: pd.read_csv(f"results/{n}.csv") for n in ["p4_base_vs_rule_only", "p4_team_prompt_comparison"]
             if os.path.exists(f"results/{n}.csv")}
    return df, rules, sim, wide, thr, coords, tau_sweep, bc, extra


df, rules, sim, wide, thr, coords, tau_sweep, bc, extra = load()
tree = Tree(rules)
JUDGES = list(wide.columns)
TH0 = {(r.judge, r.source): r.threshold for r in thr.itertuples()}
base_argmax = (bc.loc[bc.groupby("text_id").p_violation.idxmax()].set_index("text_id")["cand"].to_dict()
               if bc is not None else {})

for k, v in {"sel": None, "proposals": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def select(tid):
    if tid and tid != st.session_state.sel:
        st.session_state.sel = tid
        st.rerun()


def show_text(tid, key):
    r = df[df.text_id == tid].iloc[0]
    rule = f"`{r.gold}`" if r.gold else "없음"
    st.markdown(f"**{tid}** · {SRC_NAME[r.source]}" + (f" · 응답 LLM `{r.response_llm}`" if r.response_llm else "") +
                f" · 사람 라벨 **{'위반' if r.human_harmful == 1 else '정상'}** · 정답 규칙 {rule}")
    if st.toggle("원문 보기 (유해한 내용이 있을 수 있어요)", key=key, value=False):
        if r.question:
            st.markdown(f"**{'질문' if r.source == 'do_not_answer' else '맥락'}**: {r.question}")
        st.markdown(f"**{'응답' if r.source == 'do_not_answer' else '글'}**: {r.text}")


# ---------------------------------------------------------------
# 사이드바: 공통 필터
# ---------------------------------------------------------------
st.sidebar.title("공통 필터")
st.sidebar.caption("분할: dev (DNA 전체 · Jigsaw·ProsocialDialog 표본 50개씩)")
f_src = st.sidebar.multiselect("출처", list(SRC_NAME), default=list(SRC_NAME), format_func=SRC_NAME.get)
f_l1 = st.sidebar.multiselect("규칙 (대분류)", sorted(x for x in df.rule_l1.unique() if x))
f_llm = st.sidebar.multiselect("응답 LLM (DNA)", sorted(x for x in df.response_llm.unique() if x))
f_judges = st.sidebar.multiselect("판정 모델", JUDGES, default=JUDGES)
f_label = st.sidebar.radio("사람 라벨", ["전체", "위반", "정상"], horizontal=True)
f_err = st.sidebar.radio("판정 결과", ["전체", "오판 있음", "모두 맞음"], horizontal=True)
override = st.sidebar.checkbox("임계값 직접 지정 (끄면: 기준 모델 0.5, 팀 판정기 dev 최적)")
th_all = st.sidebar.slider("임계값", 0.05, 0.95, 0.5, 0.05, disabled=not override)


def TH(jc, src):
    return th_all if override else TH0.get((jc, src), 0.5)


judges = f_judges or JUDGES
view = df[df.source.isin(f_src or list(SRC_NAME))]
if f_l1:
    view = view[view.rule_l1.isin(f_l1)]
if f_llm:
    view = view[view.response_llm.isin(f_llm)]
if f_label != "전체":
    view = view[view.human_harmful == (1 if f_label == "위반" else 0)]
if f_err != "전체":
    wrong = []
    for r in view.itertuples():
        ps = [wide.at[r.text_id, jc] for jc in judges if pd.notna(wide.at[r.text_id, jc])]
        js = [jc for jc in judges if pd.notna(wide.at[r.text_id, jc])]
        wrong.append(any(int(p >= TH(jc, r.source)) != r.human_harmful for p, jc in zip(ps, js)))
    wrong = np.array(wrong, dtype=bool)
    view = view[wrong] if f_err == "오판 있음" else view[~wrong]
st.sidebar.metric("필터 후 텍스트 수", f"{len(view):,}")
st.sidebar.caption(" · ".join(f"{SRC_NAME[s]} {n}" for s, n in view.source.value_counts().items()))

st.sidebar.divider()
st.sidebar.subheader("선택한 텍스트")
ids = view.text_id.tolist()
if st.session_state.sel and st.session_state.sel not in ids:
    ids = [st.session_state.sel] + ids
pick = st.sidebar.selectbox("텍스트 ID", ids, index=ids.index(st.session_state.sel) if st.session_state.sel in ids else None,
                            placeholder="뷰에서 클릭하거나 여기서 고르기")
if pick:
    select(pick)
sel = st.session_state.sel

st.title("AI 윤리 규칙 위반 판정 설명 (데모)")
tab1, tab2, tab3, tab4 = st.tabs(["P1 규칙 찾기", "P2 판정 비교", "P3 오판 원인", "P4 수정 효과"])


def metrics(y, p, th):
    pred = (p >= th).astype(int)
    two = len(set(y)) > 1
    return {"ROC-AUC": roc_auc_score(y, p) if two else np.nan, "PR-AUC": average_precision_score(y, p) if two else np.nan,
            "F1": f1_score(y, pred, zero_division=0), "κ(사람)": cohen_kappa_score(y, pred) if two else np.nan,
            "오판 비율": (pred != y).mean(), "임계값": th}


# ---------------------------------------------------------------
# P1 규칙 찾기
# ---------------------------------------------------------------
with tab1:
    c0, c00 = st.columns([1, 3])
    p1_src = c0.radio("출처 (정답 규칙이 있는 것만)", ["do_not_answer", "jigsaw"], format_func=SRC_NAME.get)
    tau = c00.slider("τ (중분류를 남기는 유사도 기준)", 0.0, 0.6, TAU, 0.01)
    sub = view[(view.source == p1_src) & (view.gold != "")]
    res = {r.text_id: tree.bfs(sim.loc[r.text_id], r.source, tau) for r in sub.itertuples()}
    gold = dict(zip(sub.text_id, sub.gold))
    if res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("top-1 정확도", f"{np.mean([v[0][0] == gold[k] for k, v in res.items()]):.3f}")
        c2.metric("top-3 정확도", f"{np.mean([gold[k] in v[0][:3] for k, v in res.items()]):.3f}")
        c3.metric("규칙 찾기 효율 (비교 노드 비율)", f"{np.mean([v[2] for v in res.values()]):.3f}")
        c4.metric("텍스트 수", f"{len(res):,}")
    left, right = st.columns([3, 2])
    with left:
        if sel and sel in set(df.text_id):
            g = df[df.text_id == sel].iloc[0]
            tname = {"do_not_answer": "dna", "prosocial": "dna", "jigsaw": "jigsaw"}[g.source]
            rs = sim.loc[sel]
            ranked, kept, _ = tree.bfs(rs, g.source, tau)
            st.subheader(f"규칙 트리 위의 유사도 — {sel}")
            t = rules[rules.tree == tname].copy()
            gold_path = {g.gold_l1, g.gold_l2, g.gold_l3} - {""}
            t["similarity"] = t.node_id.map(rs)
            t["label"] = t.node_id.map(lambda n: ("★정답 " if n in gold_path else "") +
                                       ("✓BFS " if n in kept or n in ranked[:TOP_K] else "")) + t["name"]
            t["parent_id"] = t["parent"].replace("ROOT", "")
            fig = px.treemap(t, ids="node_id", names="label", parents="parent_id", color="similarity",
                             color_continuous_scale="Blues", hover_data={"similarity": ":.3f"})
            fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=440)
            st.plotly_chart(fig, width="stretch")
            st.caption("★정답 = 정답 경로 · ✓BFS = τ 로 남긴 중분류와 상위 5개 후보 · 색이 진할수록 유사도가 높음")
            cand = pd.DataFrame({"후보": [tree.name[n] for n in ranked[:TOP_K]],
                                 "유사도": [round(rs[n], 3) for n in ranked[:TOP_K]],
                                 "정답": ["★" if n == g.gold else "" for n in ranked[:TOP_K]]})
            if base_argmax.get(sel):
                cand["'기본' 판정에서 최댓값"] = ["◀" if n == base_argmax[sel] else "" for n in ranked[:TOP_K]]
            st.dataframe(cand, hide_index=True, width="stretch")
            show_text(sel, "p1_text")
        else:
            st.info("왼쪽에서 텍스트를 고르거나 오른쪽 경계 사례 표에서 한 줄을 클릭하세요.")
    with right:
        st.subheader("경계 사례 (τ±0.02 에서 1위 규칙이 바뀜)")
        b = []
        for r in sub.itertuples():
            a0 = res[r.text_id][0][0]
            lo = tree.bfs(sim.loc[r.text_id], r.source, tau - 0.02)[0][0]
            hi = tree.bfs(sim.loc[r.text_id], r.source, tau + 0.02)[0][0]
            if a0 != lo or a0 != hi:
                b.append({"text_id": r.text_id, "1위": tree.name[a0], "τ-0.02": tree.name[lo], "τ+0.02": tree.name[hi]})
        b = pd.DataFrame(b, columns=["text_id", "1위", "τ-0.02", "τ+0.02"])
        st.caption(f"{len(b)}개")
        ev = st.dataframe(b, hide_index=True, width="stretch", height=240, on_select="rerun",
                          selection_mode="single-row", key="p1_boundary")
        if ev.selection.rows:
            select(b.iloc[ev.selection.rows[0]]["text_id"])
        st.subheader("자주 틀리는 가지 (정답 중분류별 top-1)")
        if res:
            acc = pd.DataFrame({"gold_l2": sub.gold_l2.values, "ok": [res[t][0][0] == gold[t] for t in sub.text_id]})
            acc = acc.groupby("gold_l2").agg(n=("ok", "size"), top1=("ok", "mean")).reset_index()
            acc["가지"] = acc.gold_l2.map(tree.name)
            fig = px.bar(acc.sort_values("top1"), x="top1", y="가지", orientation="h", text="n", color="top1",
                         color_continuous_scale="RdYlGn", range_color=(0, 1))
            fig.update_layout(height=360, margin=dict(t=10, l=0, r=0, b=0), coloraxis_showscale=False, yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
            st.caption("막대 옆 숫자 = 건수")
    with st.expander("τ별 비교표 (analyze.py 결과)"):
        st.dataframe(tau_sweep, hide_index=True)


# ---------------------------------------------------------------
# P2 판정 비교
# ---------------------------------------------------------------
with tab2:
    st.subheader("출처별 · 판정 모델별 지표")
    rows = []
    for s, g in view.groupby("source"):
        y = g.human_harmful.values
        for jc in judges:
            p = wide.loc[g.text_id, jc]
            if p.notna().all() and len(p):
                rows.append({"출처": SRC_NAME[s], "판정 모델": jc, "n": len(p), "위반": int(y.sum()),
                             **metrics(y, p.values, TH(jc, s))})
    mt = pd.DataFrame(rows)
    st.dataframe(mt.round(3), hide_index=True, width="stretch")
    st.caption("DNA 는 위반이 3% 정도로 드물어서 ROC-AUC 보다 PR-AUC 를 같이 봐야 해요 · Jigsaw·ProsocialDialog 는 50개라 차이를 과하게 해석하지 마세요")

    group = st.radio("묶음 기준", ["source", "rule_l1", "rule_l2", "response_llm"], horizontal=True,
                     format_func={"source": "출처", "rule_l1": "규칙(대분류)", "rule_l2": "규칙(중분류)",
                                  "response_llm": "응답 LLM (DNA만)"}.get)
    rows = []
    for jc in judges:
        for r in view.itertuples():
            p = wide.at[r.text_id, jc]
            gname = getattr(r, group)
            if pd.isna(p) or not gname:
                continue
            rows.append({"판정 모델": jc, "묶음": SRC_NAME.get(gname, gname), "err": int((p >= TH(jc, r.source)) != r.human_harmful)})
    if rows:
        mat = pd.DataFrame(rows).groupby(["판정 모델", "묶음"]).agg(n=("err", "size"), 오판=("err", "sum")).reset_index()
        mat["비율"] = mat.오판 / mat.n
        mat["표시"] = mat.apply(lambda r: f"{r.비율:.0%}<br>({r.오판}/{r.n})", axis=1)
        mat["비율_표시"] = np.where(mat.n < MIN_N, np.nan, mat.비율)
        piv = mat.pivot(index="묶음", columns="판정 모델", values="비율_표시")
        txt = mat.pivot(index="묶음", columns="판정 모델", values="표시").fillna("")
        fig = px.imshow(piv, color_continuous_scale="Reds", aspect="auto", zmin=0)
        fig.update_traces(text=txt.values, texttemplate="%{text}")
        fig.update_layout(height=140 + 38 * len(piv), margin=dict(t=10, l=0, r=0, b=0))
        st.subheader("판정 모델 × 묶음 오판 비율 (괄호: 오판/건수)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"건수가 {MIN_N}건 미만인 칸은 색을 빼서 흐리게 표시했어요 · 빈칸 = 그 묶음에 판정이 없음")

    st.subheader("두 모델의 위반 확률 비교")
    a, b2 = st.columns(2)
    ja = a.selectbox("x축 모델", judges, index=0)
    jb = b2.selectbox("y축 모델", judges, index=min(1, len(judges) - 1))
    sc = view[["text_id", "source", "human_harmful"]].copy()
    sc[ja], sc[jb] = wide.loc[sc.text_id, ja].values, wide.loc[sc.text_id, jb].values
    sc = sc.dropna()
    sc["사람 라벨"] = np.where(sc.human_harmful == 1, "위반", "정상")
    sc["출처"] = sc.source.map(SRC_NAME)
    fig = px.scatter(sc, x=ja, y=jb, color="사람 라벨", symbol="출처", custom_data=["text_id"], opacity=0.7,
                     color_discrete_map={"위반": "#d1495b", "정상": "#9bb7d4"})
    fig.update_traces(marker=dict(size=7))
    if sel in set(sc.text_id):
        c = sc[sc.text_id == sel]
        fig.add_scatter(x=c[ja], y=c[jb], mode="markers", name="선택한 텍스트", hoverinfo="skip",
                        marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=3, color="black")))
    fig.update_layout(height=480)
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="p2_scatter")
    if ev and ev.selection.points and ev.selection.points[0].get("customdata"):
        select(ev.selection.points[0]["customdata"][0])
    st.caption("점을 클릭하면 그 텍스트가 모든 탭에서 선택돼요")
    if sel:
        show_text(sel, "p2_text")


# ---------------------------------------------------------------
# P3 오판 원인 (모델마다 따로 진단 — diagnose.py)
# ---------------------------------------------------------------
with tab3:
    st.caption("잘못된 판정 하나 = (텍스트, 판정 모델) 한 쌍 · 위에서부터 처음 해당하는 원인 하나만: "
               "1 규칙 찾기 → 2 라벨 확인 → 3 규칙 문장 → 4 판정 모델 · 정답 규칙이 있는 DNA·Jigsaw 만")
    c1, c2 = st.columns([1, 3])
    p3_src = c1.radio("출처", ["do_not_answer", "jigsaw"], format_func=SRC_NAME.get, key="p3_src")
    k = c2.slider("규칙 찾기 문제 기준: 정답 규칙이 상위 몇 개 후보 안에 없으면?", 1, 10, TOP_K)
    g3 = view[(view.source == p3_src) & (view.gold != "")]
    m3 = g3[["text_id", "human_harmful"]].reset_index(drop=True)
    js3 = [jc for jc in judges if len(g3) and wide.loc[g3.text_id, jc].notna().all()]
    for jc in js3:
        m3[jc] = wide.loc[m3.text_id, jc].values
    gold_in_topk = {r.text_id: r.gold in tree.bfs(sim.loc[r.text_id], r.source, tau)[0][:k] for r in g3.itertuples()}
    d = diagnose(m3, js3, {jc: TH(jc, p3_src) for jc in js3}, gold_in_topk, base_argmax,
                 dict(zip(g3.text_id, g3.gold))).merge(coords, on="text_id", how="left") if js3 else pd.DataFrame()
    if not any(is_base(j) for j in js3):
        st.info("규칙을 입력받는 모델의 '기본' 판정(BFS 후보 규칙으로 판정)이 없어서 원인 1은 0건이에요.")
    if len(d):
        st.subheader("판정 모델 × 원인 (잘못된 판정 수)")
        tbl = d.pivot_table(index="judge", columns="cause", values="text_id", aggfunc="count", fill_value=0) \
            .reindex(columns=CAUSES, fill_value=0)
        tbl["합계"] = tbl.sum(axis=1)
        st.dataframe(tbl, width="stretch")
        jsel = st.selectbox("지도에 표시할 판정 모델", js3, key="p3_judge")
        dj = d[d.judge == jsel].copy()
        left, right = st.columns([3, 2])
        with left:
            base_pts = coords[coords.text_id.isin(g3.text_id) & ~coords.text_id.isin(dj.text_id)]
            fig = px.scatter(base_pts, x="x", y="y")
            fig.update_traces(marker=dict(color="#c9c9c9", size=4, opacity=0.3), name="맞은 판정",
                              showlegend=True, hoverinfo="skip")
            dj["모양"] = np.where(dj.gold_outside_topk == 1, f"정답 규칙 상위 {k} 밖", f"정답 규칙 상위 {k} 안")
            fig2 = px.scatter(dj, x="x", y="y", color="cause", symbol="모양", custom_data=["text_id"],
                              color_discrete_map=CAUSE_COLORS,
                              symbol_map={f"정답 규칙 상위 {k} 밖": "diamond", f"정답 규칙 상위 {k} 안": "circle"},
                              hover_data={"text_id": True, "p_violation": ":.3f", "error_type": True, "x": False, "y": False})
            fig2.update_traces(marker=dict(size=8, opacity=0.85))
            for tr in fig2.data:
                fig.add_trace(tr)
            if sel in set(coords.text_id):
                c = coords[coords.text_id == sel]
                fig.add_scatter(x=c.x, y=c.y, mode="markers", name="선택한 텍스트", hoverinfo="skip",
                                marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=3, color="black")))
            fig.update_layout(height=500, xaxis_visible=False, yaxis_visible=False, legend_title=None)
            ev = st.plotly_chart(fig, width="stretch", on_select="rerun",
                                 selection_mode=("points", "box", "lasso"), key="p3_map")
            st.caption("비슷한 텍스트끼리 가깝게 놓인 지도 · 회색 = 이 모델이 맞힌 텍스트 · ◆ = 정답 규칙이 상위 k 후보 밖")
        sel_ids = [p["customdata"][0] for p in ev.selection.points if p.get("customdata")] if ev and ev.selection.points else []
        with right:
            area = dj[dj.text_id.isin(sel_ids)] if len(sel_ids) > 1 else dj
            st.subheader(("선택 영역" if len(sel_ids) > 1 else "전체") + " 원인 분포")
            cnt = area.cause.value_counts().reindex(CAUSES, fill_value=0).rename_axis("원인").reset_index(name="건수")
            cnt["비율"] = (cnt.건수 / max(cnt.건수.sum(), 1)).round(3)
            fig = px.bar(cnt, x="건수", y="원인", orientation="h", color="원인", text="건수", color_discrete_map=CAUSE_COLORS)
            fig.update_layout(height=250, showlegend=False, margin=dict(t=10, l=0, r=0, b=0), yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(cnt, hide_index=True, width="stretch")
        if len(sel_ids) == 1:
            select(sel_ids[0])
    if sel and sel in set(m3.text_id):
        st.subheader(f"진단 — {sel}")
        r = m3[m3.text_id == sel].iloc[0]
        ds = d[d.text_id == sel].set_index("judge") if len(d) else pd.DataFrame()
        st.dataframe(pd.DataFrame({
            "판정 모델": js3, "조건": [jc.split(" / ")[1] for jc in js3],
            "위반 확률": [round(r[c], 4) for c in js3], "임계값": [TH(c, p3_src) for c in js3],
            "판정": ["위반" if r[c] >= TH(c, p3_src) else "정상" for c in js3],
            "맞음": ["✗" if c in ds.index else "✓" for c in js3],
            "원인": [ds.loc[c, "cause"] if c in ds.index else "" for c in js3]}), hide_index=True, width="stretch")
        st.caption("조건: default = 규칙 없이 판정 · base(기본) = BFS 상위 5개 후보 규칙마다 판정 후 최댓값 · "
                   "rule_only(규칙 문장만) = 정답 규칙만 넣고 판정")
        show_text(sel, "p3_text")


# ---------------------------------------------------------------
# P4 수정 효과
# ---------------------------------------------------------------
with tab4:
    st.caption("데모 범위: ① 규칙 찾기를 정답 규칙으로 바꿨을 때 ② 팀 판정기 원칙 문장 수정 전후 ③ 임계값 수정. "
               "'규칙 제외'·'규칙+예시' 조건과 '규칙 문장 편집 → 다시 판정'은 다음 단계")
    if "p4_base_vs_rule_only" in extra and len(extra["p4_base_vs_rule_only"]):
        st.subheader("① '기본'(BFS 후보 규칙) → '규칙 문장만'(정답 규칙) 바꿨을 때")
        e = extra["p4_base_vs_rule_only"].copy()
        e["source"] = e.source.map(SRC_NAME)
        st.dataframe(e.round(3), hide_index=True, width="stretch")
        st.caption("AUC 변화가 양수 = 정답 규칙을 주면 좋아짐 → 규칙 찾기를 고치면 효과가 있음")
    if "p4_team_prompt_comparison" in extra:
        st.subheader("② 팀 판정기 원칙 문장 수정 전후 (DNA dev 300개, '규칙 문장만')")
        tp = extra["p4_team_prompt_comparison"]
        fig = px.bar(tp.melt(id_vars=["version"], value_vars=["roc_auc", "pr_auc"]), x="version", y="value",
                     color="variable", barmode="group", text_auto=".3f")
        fig.update_layout(height=320, margin=dict(t=10, l=0, r=0, b=0), legend_title=None, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("shieldgemma(기준) = 원칙 추가 없음 · team_v1 거절 인정 · team_v2 응답만 판정 · team_v3 둘 다 + '실제로 도움을 줬는가'")

    st.subheader("③ 임계값 수정")
    c1, c2, c3 = st.columns([2, 1, 1])
    jc = c1.selectbox("수정할 판정 모델", judges, key="p4_judge")
    src4 = c2.selectbox("출처", [s for s in SRC_NAME if s in set(view.source)], format_func=SRC_NAME.get, key="p4_src")
    g4 = view[view.source == src4]
    p = wide.loc[g4.text_id, jc]
    if p.isna().any() or len(p) == 0:
        st.warning("이 출처에는 이 모델의 판정이 없어요.")
    else:
        base_th = TH0.get((jc, src4), 0.5)
        if c3.button("원래 값으로 되돌리기"):
            st.session_state.pop("p4_th", None)
            st.rerun()
        new_th = st.slider(f"바꾼 임계값 (원래 {base_th:.2f})", 0.01, 0.99, st.session_state.get("p4_th", base_th), 0.01, key="p4_th")
        y, pv = g4.human_harmful.values, p.values
        before, after = (pv >= base_th).astype(int), (pv >= new_th).astype(int)
        fl = before != after
        a, b2 = st.columns(2)
        a.markdown(f"**원래 설정 ({base_th:.2f})**")
        a.dataframe(pd.Series(metrics(y, pv, base_th)).round(3).rename("값"))
        b2.markdown(f"**바꾼 설정 ({new_th:.2f})**")
        b2.dataframe(pd.Series(metrics(y, pv, new_th)).round(3).rename("값"))
        m1, m2, m3c = st.columns(3)
        m1.metric("뒤집힌 판정", int(fl.sum()))
        m2.metric("맞게 바뀜", int((fl & (after == y)).sum()))
        m3c.metric("틀리게 바뀜", int((fl & (after != y)).sum()))
        if fl.any():
            ft = pd.DataFrame({"text_id": g4.text_id.values[fl], "위반 확률": pv[fl].round(4),
                               "사람 라벨": np.where(y[fl] == 1, "위반", "정상"),
                               "결과": np.where(after[fl] == y[fl], "맞게 바뀜", "틀리게 바뀜")})
            ev = st.dataframe(ft, hide_index=True, width="stretch", height=220, on_select="rerun",
                              selection_mode="single-row", key="p4_flips")
            if ev.selection.rows:
                select(ft.iloc[ev.selection.rows[0]]["text_id"])
        note = st.text_input("수정안 메모")
        if st.button("수정안 저장"):
            st.session_state.proposals.append({"판정 모델": jc, "출처": SRC_NAME[src4], "임계값": new_th,
                                               "맞게 바뀜": int((fl & (after == y)).sum()),
                                               "틀리게 바뀜": int((fl & (after != y)).sum()), "메모": note})
    if st.session_state.proposals:
        pr = pd.DataFrame(st.session_state.proposals)
        st.dataframe(pr, hide_index=True, width="stretch")
        st.download_button("저장한 수정안 CSV 로 받기", pr.to_csv(index=False).encode("utf-8-sig"), "proposals.csv")
