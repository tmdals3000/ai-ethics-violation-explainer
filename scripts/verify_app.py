"""현재 dev 자료로 화면 필터·초기화·차트 이벤트를 점검하는 로컬 실행 기록."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
scratch = Path(__file__).resolve().parents[1] / 'data/tmp'
scratch.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(scratch)
from streamlit.testing.v1 import AppTest
from src.common import load_config
from src import data

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/app.py'), default_timeout=90).run()


def widget(group, label):
    return next(w for w in group if w.label == label)


def check(label):
    errors = [e.message for e in app.exception]
    print(label, errors, flush=True)
    assert not errors


def event(key, selection):
    chart = next(e for e in app if getattr(e, 'key', None) == key)
    states = app._tree.get_widget_states()
    state = states.widgets.add()
    state.id = chart.proto.id
    state.string_value = json.dumps({'selection': selection})
    app._run(states)
    check(key)


check('initial')
widget(app.multiselect, '출처').set_value([]).run()
check('empty sources')
widget(app.multiselect, '출처').set_value(['jigsaw']).run()
check('Jigsaw only')
widget(app.selectbox, '조건').select('rule_only').run()
check('Jigsaw rule_only')
widget(app.multiselect, '출처').set_value(['prosocial']).run()
check('PD without gold rules')
widget(app.selectbox, '조건').select('base').run()
widget(app.toggle, '판정 임계값 직접 지정').set_value(True).run()
widget(app.slider, '새 임계값').set_value(.9).run()
widget(app.button, '원래 값으로 되돌리기').click().run()
check('reset')
assert not widget(app.toggle, '판정 임계값 직접 지정').value
assert widget(app.slider, 'P4 검색 τ').value == .2
model = widget(app.selectbox, '판정 모델')
assert 'team' in model.options, '현재 팀 프롬프트의 judge 실행을 먼저 완료하세요.'
model.select('team').run()
check('active team prompt')
widget(app.multiselect, '출처').set_value(['do_not_answer']).run()
cfg = load_config()
texts, _, _, sample = data.load(cfg)
ids = texts[(texts.source == 'do_not_answer') & texts.text_id.isin(sample.text_id)].text_id.tolist()
event('p2_scatter', {'pair_text': {'text_id': [ids[0]]}})
assert app.session_state.sel == ids[0]
event('p3_map', {'map_text': {'text_id': [ids[1]]}, 'brush': {}})
assert app.session_state.sel == ids[1]
event('p1_tree', {'rule_node': {'node_id': ['H01']}})
assert widget(app.selectbox, '노드').value == 'H01'
event('p3_causes', {'cause_pick': {'cause': ['retrieval']}})
assert widget(app.selectbox, '강조할 원인').value == 'retrieval'
print('ALL APP CHECKS PASSED', flush=True)
