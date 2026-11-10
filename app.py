"""
5단계: 화면 (Streamlit) — 탭 4개 (P1~P4) + 공통 필터 + 선택 연동 + 원문 가리기

실행: streamlit run app.py
입력: demo 폴더의 dna_long.csv, rules.csv, similarity.csv, judgments.csv, results/ (analyze.py 결과)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score

st.set_page_config(page_title="AI 윤리 판정 설명", layout="wide")
MIN_N = 10
CAUSE_COLORS = {"1. 규칙 찾기 문제": "#d1495b", "2. 라벨 확인 필요": "#edae49",
                "3. 규칙 문장 문제": "#00798c", "4. 판정 모델 문제": "#7b4fa0", "정상 판정": "#c9c9c9"}


# ---------------------------------------------------------------
# 데이터 읽기 (한 번만 읽고 캐시)
# ---------------------------------------------------------------
@st.cache_data
def load():
    df = pd.read_csv("dna_long.csv")
    df = df[df["split"] == "dev"].reset_index(drop=True)
    rules = pd.read_csv("rules.csv")
    sim = pd.read_csv("similarity.csv").set_index("text_id").loc[df["text_id"]]
    j = pd.read_csv("judgments.csv")
    j["judge_cond"] = j["judge"] + " / " + j["condition"]
    wide = j.pivot_table(index="text_id", columns="judge_cond", values="p_violation")
    diag = pd.read_csv("results/p3_diagnosis.csv")
    p1 = pd.read_csv("results/p1_retrieval.csv")
    tau_sweep = pd.read_csv("results/p1_tau_sweep.csv")
    return df, rules, sim, wide, diag, p1, tau_sweep


df, rules, sim, wide, diag, p1_saved, tau_sweep = load()
JUDGES = list(wide.columns)
name_of = dict(zip(rules["node_id"], rules["name"]))
children = rules.groupby("parent")["node_id"].apply(list).to_dict()
L1 = rules[rules.level == 1]["node_id"].tolist()
L2 = rules[rules.level == 2]["node_id"].tolist()

if "sel" not in st.session_state:
    st.session_state.sel = None
if "proposals" not in st.session_state:
    st.session_state.proposals = []


def select(tid):
    """어느 뷰에서든 텍스트를 고르면 여기로 → 모든 뷰에서 같은 텍스트가 강조됨"""
    if tid and tid != st.session_state.sel:
        st.session_state.sel = tid
        st.rerun()


def show_text(tid, key):
    """원문은 기본으로 가리고, 누를 때만 보여 줌"""
    r = df[df.text_id == tid].iloc[0]
    st.markdown(f"**{tid}** · 응답 LLM `{r.response_llm}` · 사람 라벨 "
                f"**{'위반' if r.human_harmful == 1 else '정상'}** · 정답 규칙 `{r.rule_l1}` › `{r.rule_l2}` › `{r.rule_l3}`")
    if st.toggle("원문 보기 (유해한 내용이 있을 수 있어요)", key=key, value=False):
        st.markdown(f"**질문**: {r.question}")
        st.markdown(f"**응답**: {r.response}")


# ---------------------------------------------------------------
# 사이드바: 공통 필터 + 선택한 텍스트
# ---------------------------------------------------------------
st.sidebar.title("공통 필터")
st.sidebar.caption("출처: Do-Not-Answer · 분할: dev (데모)")
f_l1 = st.sidebar.multiselect("규칙 (대분류)", sorted(df.rule_l1.unique()))
f_llm = st.sidebar.multiselect("응답 LLM", sorted(df.response_llm.unique()))
f_judges = st.sidebar.multiselect("판정 모델", JUDGES, default=JUDGES)
f_label = st.sidebar.radio("사람 라벨", ["전체", "위반", "정상"], horizontal=True)
f_err = st.sidebar.radio("판정 결과", ["전체", "오판 있음", "모두 맞음"], horizontal=True)
threshold = st.sidebar.slider("위반 판정 임계값", 0.05, 0.95, 0.5, 0.05)

view = df.copy()
if f_l1:
    view = view[view.rule_l1.isin(f_l1)]
if f_llm:
    view = view[view.response_llm.isin(f_llm)]
if f_label != "전체":
    view = view[view.human_harmful == (1 if f_label == "위반" else 0)]
judges = f_judges or JUDGES
preds = (wide.loc[view.text_id, judges] >= threshold).astype(int)
wrong = preds.ne(view.set_index("text_id")["human_harmful"], axis=0).any(axis=1)
if f_err == "오판 있음":
    view = view[view.text_id.isin(wrong[wrong].index)]
elif f_err == "모두 맞음":
    view = view[view.text_id.isin(wrong[~wrong].index)]
st.sidebar.metric("필터 후 텍스트 수", f"{len(view):,}")

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


# ---------------------------------------------------------------
# P1 규칙 찾기
# ---------------------------------------------------------------
def bfs(row_sim, tau):
    s2 = row_sim[L2]
    kept = s2[s2 >= tau].index.tolist() or [s2.idxmax()]
    leaves = [c for n in kept for c in children.get(n, [])]
    return row_sim[leaves].sort_values(ascending=False).index.tolist(), kept, len(L1) + len(L2) + len(leaves)


with tab1:
    tau = st.slider("τ (중분류를 남기는 유사도 기준)", 0.0, 0.6, 0.20, 0.01)
    # τ 를 바꾸면 바로 다시 계산 (필터된 텍스트만)
    sub = sim.loc[view.text_id]
    res = [bfs(sub.iloc[i], tau) for i in range(len(sub))]
    gold = view.set_index("text_id")["gold_l3"]
    top1 = np.mean([r[0][0] == gold.iloc[i] for i, r in enumerate(res)]) if res else 0
    top3 = np.mean([gold.iloc[i] in r[0][:3] for i, r in enumerate(res)]) if res else 0
    eff = np.mean([r[2] for r in res]) / len(rules) if res else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("top-1 정확도", f"{top1:.3f}")
    c2.metric("top-3 정확도", f"{top3:.3f}")
    c3.metric("규칙 찾기 효율 (비교 노드 비율)", f"{eff:.3f}")
    c4.metric("텍스트 수", f"{len(res):,}")

    left, right = st.columns([3, 2])
    with left:
        if sel:
            st.subheader(f"규칙 트리 위의 유사도 — {sel}")
            rs = sim.loc[sel]
            ranked, kept, _ = bfs(rs, tau)
            g = df[df.text_id == sel].iloc[0]
            gold_path = {g.gold_l1, g.gold_l2, g.gold_l3}
            bfs_path = set(kept) | set(ranked[:5]) | {r for r in rules[rules.level == 1].node_id}
            t = rules.copy()
            t["similarity"] = t["node_id"].map(rs)
            t["표시"] = t["node_id"].map(lambda n: ("★정답 " if n in gold_path else "") +
                                         ("✓BFS " if n in kept or n in ranked[:5] else ""))
            t["label"] = t["표시"] + t["name"]
            t["parent_id"] = t["parent"].replace("ROOT", "")
            fig = px.treemap(t, ids="node_id", names="label", parents="parent_id",
                             color="similarity", color_continuous_scale="Blues",
                             hover_data={"similarity": ":.3f"})
            fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=460)
            st.plotly_chart(fig, width="stretch")
            st.caption("★정답 = 정답 경로 · ✓BFS = τ 로 남긴 중분류와 상위 5개 후보 · 색이 진할수록 유사도가 높음")
            st.markdown("**상위 5개 후보**")
            st.dataframe(pd.DataFrame({"후보": [name_of[n] for n in ranked[:5]],
                                       "유사도": [round(rs[n], 3) for n in ranked[:5]],
                                       "정답": ["★" if n == g.gold_l3 else "" for n in ranked[:5]]}),
                         hide_index=True, width="stretch")
            show_text(sel, "p1_text")
        else:
            st.info("왼쪽에서 텍스트를 고르거나 오른쪽 경계 사례 표에서 한 줄을 클릭하세요.")
    with right:
        st.subheader("경계 사례 (τ±0.02 에서 1위 규칙이 바뀜)")
        bres_lo = [bfs(sub.iloc[i], tau - 0.02)[0][0] for i in range(len(sub))]
        bres_hi = [bfs(sub.iloc[i], tau + 0.02)[0][0] for i in range(len(sub))]
        b = pd.DataFrame({"text_id": sub.index, "1위": [name_of[r[0][0]] for r in res],
                          "τ-0.02": [name_of[x] for x in bres_lo], "τ+0.02": [name_of[x] for x in bres_hi]})
        b = b[(b["1위"] != b["τ-0.02"]) | (b["1위"] != b["τ+0.02"])]
        st.caption(f"{len(b)}개")
        ev = st.dataframe(b, hide_index=True, width="stretch", height=260,
                          on_select="rerun", selection_mode="single-row", key="p1_boundary")
        if ev.selection.rows:
            select(b.iloc[ev.selection.rows[0]]["text_id"])

        st.subheader("자주 틀리는 가지 (정답 중분류별 top-1)")
        acc = pd.DataFrame({"gold_l2": view.gold_l2.values,
                            "correct": [r[0][0] == gold.iloc[i] for i, r in enumerate(res)]})
        acc = acc.groupby("gold_l2").agg(n=("correct", "size"), top1=("correct", "mean")).reset_index()
        acc["중분류"] = acc.gold_l2.map(name_of)
        fig = px.bar(acc.sort_values("top1"), x="top1", y="중분류", orientation="h", text="n",
                     color="top1", color_continuous_scale="RdYlGn", range_color=(0, 1))
        fig.update_layout(height=380, margin=dict(t=10, l=0, r=0, b=0), coloraxis_showscale=False, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("막대 옆 숫자 = 건수")

    with st.expander("τ별 비교표 (analyze.py 결과)"):
        st.dataframe(tau_sweep, hide_index=True)


# ---------------------------------------------------------------
# P2 판정 비교
# ---------------------------------------------------------------
def metrics(y, p, th):
    pred = (p >= th).astype(int)
    return {"ROC-AUC": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
            "PR-AUC": average_precision_score(y, p) if y.sum() > 0 else np.nan,
            "F1": f1_score(y, pred, zero_division=0),
            "κ(사람)": cohen_kappa_score(y, pred) if len(set(y)) > 1 or len(set(pred)) > 1 else np.nan,
            "오판 비율": (pred != y).mean()}


with tab2:
    vm = view.set_index("text_id")
    y = vm["human_harmful"].values
    mt = pd.DataFrame({jc: metrics(y, wide.loc[vm.index, jc].values, threshold) for jc in judges}).T.round(3)
    st.subheader("판정 모델별 지표")
    st.dataframe(mt, width="stretch")
    st.caption(f"위반 {int(y.sum())}건 / 전체 {len(y)}건 — 위반이 드물어서 ROC-AUC 보다 PR-AUC 를 같이 봐야 해요")

    group = st.radio("묶음 기준", ["rule_l1", "rule_l2", "response_llm"], horizontal=True,
                     format_func={"rule_l1": "규칙(대분류)", "rule_l2": "규칙(중분류)", "response_llm": "응답 LLM"}.get)
    rows = []
    for jc in judges:
        pred = (wide.loc[vm.index, jc] >= threshold).astype(int).values
        e = pd.DataFrame({"g": vm[group].values, "err": (pred != y).astype(int)})
        for gname, gg in e.groupby("g"):
            rows.append({"판정 모델": jc, "묶음": gname, "n": len(gg), "오판": gg.err.sum(), "비율": gg.err.mean()})
    mat = pd.DataFrame(rows)
    mat["표시"] = mat.apply(lambda r: f"{r.비율:.0%}<br>({r.오판}/{r.n})", axis=1)
    mat["비율_표시"] = np.where(mat.n < MIN_N, np.nan, mat.비율)   # 건수 적은 칸은 흐리게(색 없음)
    piv = mat.pivot(index="묶음", columns="판정 모델", values="비율_표시")
    txt = mat.pivot(index="묶음", columns="판정 모델", values="표시")
    fig = px.imshow(piv, color_continuous_scale="Reds", aspect="auto", zmin=0)
    fig.update_traces(text=txt.values, texttemplate="%{text}")
    fig.update_layout(height=120 + 38 * len(piv), margin=dict(t=10, l=0, r=0, b=0))
    st.subheader("판정 모델 × 묶음 오판 비율 (괄호: 오판/건수)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"건수가 {MIN_N}건 미만인 칸은 색을 빼서 흐리게 표시했어요")

    st.subheader("두 모델의 위반 확률 비교")
    a, b2 = st.columns(2)
    ja = a.selectbox("x축 모델", judges, index=0)
    jb = b2.selectbox("y축 모델", judges, index=min(1, len(judges) - 1))
    sc = pd.DataFrame({"text_id": vm.index, ja: wide.loc[vm.index, ja].values, jb: wide.loc[vm.index, jb].values,
                       "사람 라벨": np.where(y == 1, "위반", "정상")})
    sc["선택"] = np.where(sc.text_id == sel, 14, 5)
    fig = px.scatter(sc, x=ja, y=jb, color="사람 라벨", size="선택", size_max=14, custom_data=["text_id"],
                     color_discrete_map={"위반": "#d1495b", "정상": "#9bb7d4"}, opacity=0.7)
    fig.add_vline(x=threshold, line_dash="dot"); fig.add_hline(y=threshold, line_dash="dot")
    fig.update_layout(height=480)
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="p2_scatter")
    if ev and ev.selection.points:
        select(ev.selection.points[0]["customdata"][0])
    st.caption("점을 클릭하면 그 텍스트가 모든 탭에서 선택돼요 · 점선 = 임계값")
    if sel:
        show_text(sel, "p2_text")


# ---------------------------------------------------------------
# P3 오판 원인
# ---------------------------------------------------------------
with tab3:
    st.caption("원인 규칙은 위에서부터 처음 해당하는 하나만 적용: 1 규칙 찾기 → 2 라벨 확인 → 3 규칙 문장 → 4 판정 모델 "
               "(데모 한계: 팀 판정기가 없어 3·4번은 ShieldGemma(rule_only) 하나로 판단)")
    k = st.slider("규칙 찾기 문제 기준: 정답 규칙이 상위 몇 개 안에 없으면?", 1, 10, 5)
    d = diag[diag.text_id.isin(view.text_id)].copy()

    def gold_in_topk(tid):
        g = df.loc[df.text_id == tid, "gold_l3"].iloc[0]
        ranked, _, _ = bfs(sim.loc[tid], tau)   # P1 탭의 τ 와 같은 값
        return g in ranked[:k]

    # 기준(k)이 바뀌면 원인을 바로 다시 계산
    pj = (d[judges] >= threshold).astype(int)
    d["is_error"] = pj.ne(d["human_harmful"], axis=0).any(axis=1)
    d["all_agree"] = pj.nunique(axis=1) == 1
    rule_j = [c for c in judges if "rule_only" in c]
    topk = {tid: gold_in_topk(tid) for tid in d.text_id[d.is_error]}

    def cause(r):
        if not r.is_error:
            return "정상 판정"
        if not topk[r.text_id]:
            return "1. 규칙 찾기 문제"
        if r.all_agree:
            return "2. 라벨 확인 필요"
        if rule_j and any(int(r[c] >= threshold) != r.human_harmful for c in rule_j):
            return "3. 규칙 문장 문제"
        return "4. 판정 모델 문제"

    d["cause"] = d.apply(cause, axis=1)
    show_ok = st.checkbox("정상 판정도 지도에 표시", value=False)
    dm = d if show_ok else d[d.cause != "정상 판정"]
    left, right = st.columns([3, 2])
    with left:
        dm = dm.assign(크기=np.where(dm.text_id == sel, 16, 6))
        fig = px.scatter(dm, x="x", y="y", color="cause", size="크기", size_max=16,
                         color_discrete_map=CAUSE_COLORS, custom_data=["text_id"], opacity=0.8)
        fig.update_layout(height=520, xaxis_visible=False, yaxis_visible=False, legend_title=None)
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun",
                             selection_mode=("points", "box", "lasso"), key="p3_map")
        st.caption("비슷한 텍스트끼리 가깝게 놓인 지도예요 · 점 클릭 = 선택 · 상자/올가미로 영역 선택")
    sel_ids = [p["customdata"][0] for p in ev.selection.points] if ev and ev.selection.points else []
    with right:
        area = d[d.text_id.isin(sel_ids)] if len(sel_ids) > 1 else d[d.cause != "정상 판정"]
        st.subheader("원인별 건수" + (" (선택 영역)" if len(sel_ids) > 1 else " (오판 전체)"))
        cnt = area.cause.value_counts().rename_axis("원인").reset_index(name="건수")
        cnt["비율"] = (cnt.건수 / cnt.건수.sum()).round(3)
        fig = px.bar(cnt, x="건수", y="원인", orientation="h", color="원인", text="건수",
                     color_discrete_map=CAUSE_COLORS)
        fig.update_layout(height=260, showlegend=False, margin=dict(t=10, l=0, r=0, b=0), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(cnt, hide_index=True, width="stretch")
    if len(sel_ids) == 1:
        select(sel_ids[0])
    if sel and sel in d.text_id.values:
        r = d[d.text_id == sel].iloc[0]
        st.subheader(f"진단 — {sel}: {r.cause}")
        st.dataframe(pd.DataFrame({"판정 모델": judges, "위반 확률": [round(r[c], 4) for c in judges],
                                   "판정": ["위반" if r[c] >= threshold else "정상" for c in judges]}),
                     hide_index=True)
        st.caption("PDF 의 4가지 조건(기본·규칙 제외·규칙 문장만·규칙+예시) 중 데모에는 '기본'(Llama Guard·Qwen3Guard)과 "
                   "'규칙 문장만'(ShieldGemma)만 있어요")
        show_text(sel, "p3_text")


# ---------------------------------------------------------------
# P4 수정 효과 (데모: 임계값 수정만)
# ---------------------------------------------------------------
with tab4:
    st.info("데모 버전: '규칙 제외'·'규칙+예시' 조건과 '규칙 문장 편집 → 다시 판정'은 GPU 판정이 더 필요해서, "
            "여기서는 **판정 임계값을 바꿨을 때의 효과**만 비교해요.")
    jc = st.selectbox("수정할 판정 모델", judges, key="p4_judge")
    if "p4_th" not in st.session_state:
        st.session_state.p4_th = 0.5
    c1, c2 = st.columns([4, 1])
    new_th = c1.slider("바꾼 임계값", 0.01, 0.99, key="p4_th", step=0.01)
    if c2.button("원래 값(0.5)으로 되돌리기"):
        del st.session_state["p4_th"]
        st.rerun()
    vm = view.set_index("text_id")
    y = vm["human_harmful"].values
    p = wide.loc[vm.index, jc].values
    before, after = (p >= 0.5).astype(int), (p >= new_th).astype(int)
    flipped = before != after
    to_correct = int((flipped & (after == y)).sum())
    to_wrong = int((flipped & (after != y)).sum())
    a, b2 = st.columns(2)
    a.markdown("**원래 설정 (0.5)**")
    a.dataframe(pd.Series(metrics(y, p, 0.5)).round(3).rename("값"))
    b2.markdown(f"**바꾼 설정 ({new_th:.2f})**")
    b2.dataframe(pd.Series(metrics(y, p, new_th)).round(3).rename("값"))
    m1, m2, m3 = st.columns(3)
    m1.metric("뒤집힌 판정", int(flipped.sum()))
    m2.metric("맞게 바뀜", to_correct)
    m3.metric("틀리게 바뀜", to_wrong)
    if flipped.any():
        ft = pd.DataFrame({"text_id": vm.index[flipped], "위반 확률": p[flipped].round(4),
                           "사람 라벨": np.where(y[flipped] == 1, "위반", "정상"),
                           "결과": np.where(after[flipped] == y[flipped], "맞게 바뀜", "틀리게 바뀜")})
        ev = st.dataframe(ft, hide_index=True, width="stretch", height=220,
                          on_select="rerun", selection_mode="single-row", key="p4_flips")
        if ev.selection.rows:
            select(ft.iloc[ev.selection.rows[0]]["text_id"])
    note = st.text_input("수정안 메모 (예: Qwen3Guard 임계값을 0.3으로 낮추면 놓친 위반을 더 잡음)")
    if st.button("수정안 저장"):
        st.session_state.proposals.append({"판정 모델": jc, "임계값": new_th, "맞게 바뀜": to_correct,
                                           "틀리게 바뀜": to_wrong, "메모": note})
    if st.session_state.proposals:
        pr = pd.DataFrame(st.session_state.proposals)
        st.dataframe(pr, hide_index=True, width="stretch")
        st.download_button("저장한 수정안 CSV 로 받기", pr.to_csv(index=False).encode("utf-8-sig"), "proposals.csv")
