"""Presentation only: visible summaries, stable comparison panels and reference links."""
from html import escape
from pathlib import Path
import math

import streamlit as st

SOURCE_NAMES = {"do_not_answer": "Do-Not-Answer", "jigsaw": "Jigsaw ACRC", "prosocial": "ProsocialDialog"}
MODEL_NAMES = {"team": "Team · T2", "shieldgemma": "ShieldGemma", "qwen3guard": "Qwen3Guard", "llama_guard": "Llama Guard"}
CAUSE_NAMES = {"retrieval": "규칙 찾기", "label_check": "라벨 확인 필요", "rule_text": "규칙 문장", "judge_model": "판정 모델", "unclassified": "미분류", "맞음/진단대상 아님": "맞음 / 진단대상 아님"}
CAUSE_COLORS = ["#d49353", "#9a82b0", "#568ea0", "#d27571", "#a5afbd", "#cdd8e2"]


def number(value):
    return f"{value:.3f}" if value is not None and math.isfinite(value) else "—"


def model_metrics(frame):
    cards=[]
    for row in frame.itertuples():
        values=''.join(f'<div class="comparison"><span>{label}</span><span></span><span class="after">{number(getattr(row,key))}</span><span></span></div>' for key,label in [("roc_auc","ROC-AUC"),("f1","F1"),("kappa_human","사람 κ")])
        cards.append(f'<div class="source-card"><div class="source-name">{escape(SOURCE_NAMES.get(row.source,row.source))}</div><span style="font-size:10px;color:#8896a7">{row.n:,}건 · 실제 위반 {row.n_pos:,}건</span>{values}</div>')
    if cards:
        st.markdown('<div class="source-comparisons">'+''.join(cards)+'</div>',unsafe_allow_html=True)


def mount_theme():
    st.html("<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>")


def brand():
    st.sidebar.markdown('<div class="workbench-brand"><div class="brand-icon">◇</div><div><div class="brand-name">ETHICS EXPLORER</div><div class="brand-sub">VISUAL ANALYTICS LAB</div></div></div>', unsafe_allow_html=True)


def header(version, temporary=False):
    status = "임시 데이터 · TEST 미평가" if temporary else "TEST 미평가"
    st.markdown(f'<div class="page-heading"><div><div class="eyebrow">PROJECT 04 / MODEL BEHAVIOR ANALYSIS</div><h1>AI 윤리 판정 분석실</h1><p class="description">규칙을 찾고, 판단을 비교하고, 오류의 원인을 따라 수정 효과를 검증합니다.</p></div><div class="badge-row"><span class="status-badge green">● DEV WORKSPACE</span><span class="status-badge">{escape(version)}</span><span class="status-badge amber">{status}</span></div></div>', unsafe_allow_html=True)


def overview(n, errors, disagreement, boundary):
    cards = [("분석 텍스트", n, "공통 필터가 적용된 텍스트", ""), ("선택 모델의 오류", errors, "사람 라벨과 다른 기본 판정", "alert"), ("모델 간 판단 불일치", disagreement, "같은 텍스트에 서로 다른 예측", ""), ("검색 경계 사례", boundary, "기준 τ ± δ에서 규칙이 바뀜", "")]
    html = ''.join(f'<div class="overview-card {kind}"><div class="label">{label}</div><div class="value">{value:,}</div><div class="foot">{foot}</div></div>' for label, value, foot, kind in cards)
    st.markdown('<div class="overview-grid">'+html+'</div>', unsafe_allow_html=True)


def case_summary(row, model, probability=None):
    label = "위반" if row.human_label else "정상"
    probability_text = f" · {MODEL_NAMES.get(model,model)} 위반 확률 {probability:.3f}" if probability is not None else ""
    st.markdown(f'<div class="case-strip"><span>선택 사례</span><strong>{escape(str(row.text_id))}</strong><span>· {escape(SOURCE_NAMES.get(row.source,row.source))}</span><span class="pill {"danger" if row.human_label else ""}">사람 라벨: {label}</span><span>정답 규칙 {escape(str(row.gold_rule_id) or "없음")}{escape(probability_text)}</span></div>', unsafe_allow_html=True)


def section(number, title, description):
    st.markdown(f'<div class="panel-heading"><div class="panel-kicker">WORKSPACE {number}</div><h2>{escape(title)}</h2><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def route_card(predicted, gold, divergence):
    pred = " → ".join(predicted) or "선택 없음"
    actual = " → ".join(gold) or "정답 경로 없음"
    st.markdown(f'<div class="path-card"><div class="small">SELECTED CASE / 검색 증거</div><div class="route pred">찾은 경로　{escape(pred)}</div><div class="route">정답 경로　{escape(actual)}</div><div class="small" style="margin-top:10px;margin-bottom:0">최초 분기: {escape(divergence)}</div></div>', unsafe_allow_html=True)


def scenario_cards(prop):
    cards=[]
    for source, before in prop.get("metrics_before",{}).items():
        after=prop["metrics_after"][source]
        rows=[]
        for key,label in [("roc_auc","ROC-AUC"),("f1","F1"),("kappa_human","사람 κ")]:
            b,a=before[key],after[key]
            delta=a-b
            delta_text=f"{delta:+.3f}" if math.isfinite(delta) else "—"
            rows.append(f'<div class="comparison"><span>{label}</span><span>{number(b)}</span><span class="after">→ {number(a)}</span><span class="{"up" if delta>=0 else "down"}">{delta_text}</span></div>')
        cards.append(f'<div class="source-card"><div class="source-name">{escape(SOURCE_NAMES.get(source,source))} <span style="color:#96a1b0;font-weight:400">· n={before["n"]:,}</span></div>'+''.join(rows)+'</div>')
    st.markdown('<div class="source-comparisons">'+''.join(cards)+'</div>', unsafe_allow_html=True)


def transition_totals(frame):
    labels=[("unchanged_correct","맞음 유지",""),("fixed","고쳐진 판정","fixed"),("broken","새로 틀린 판정","broken"),("unchanged_wrong","오류 유지","")]
    counts=frame.transition.value_counts()
    st.markdown('<div class="delta-grid">'+''.join(f'<div class="delta-card {kind}"><span>{label}</span><strong>{int(counts.get(key,0)):,}</strong></div>' for key,label,kind in labels)+'</div>', unsafe_allow_html=True)


def references():
    st.markdown('<div class="reference-footer">DESIGN REFERENCES · <a href="https://ieeevis.b-cdn.net/vis_2024/pdfs/v-full-1067.pdf" target="_blank">AdversaFlow — 연동 탐색과 변화 흐름</a> &nbsp; / &nbsp; <a href="https://arxiv.org/pdf/1710.06501" target="_blank">Blocks — 계층과 사례 탐색</a> &nbsp; / &nbsp; <a href="https://pair-code.github.io/what-if-tool/learn/tutorials/tour/" target="_blank">What-If Tool — 시나리오 비교</a><br>공개 연구의 분석·시각화 원칙을 이 과제에 맞게 적용했습니다. 판정은 저장된 dev 결과와 실제 재계산을 사용합니다.</div>', unsafe_allow_html=True)
