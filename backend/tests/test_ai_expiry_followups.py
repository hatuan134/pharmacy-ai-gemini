"""Regression: a follow-up expiry analysis must reuse *intent*, not stale data.

All checks use local SQLite fixtures and mocked LLM; no Gemini/Groq requests or
production data are accessed.
"""
import json
from datetime import timedelta
from types import SimpleNamespace
from sqlalchemy import select
from app import ai
from app.config import today
from app.models import Batch
from app.schemas import AIChatRequest, ChatMessage

MANAGER = SimpleNamespace(role='manager', id=1)


def _request(question, history=None):
    return AIChatRequest(agent=True, message=question, history=history or [])


def _history():
    return [
        ChatMessage(role='user', content='Các lô sắp hết hạn trong 90 ngày.'),
        ChatMessage(role='assistant', content='Danh sách lô trước đó (không dùng như dữ liệu mới).'),
    ]


def test_two_demo_prompts_use_same_verified_expiry_tool():
    first = _request('Các lô sắp hết hạn trong 90 ngày.')
    second = _request(
        'Dựa trên dữ liệu các lô thuốc cận hạn trong hệ thống, hãy tổng hợp thành một báo cáo ngắn '
        'và đề xuất những biện pháp xử lý phù hợp để hạn chế thiệt hại cho nhà thuốc. Không tự tạo số liệu.',
        _history(),
    )
    for request in [first, second]:
        plan, _ = ai._fast_intent(request, MANAGER.role)
        assert [x.name for x in plan.calls] == ['expiry_alerts']
        assert plan.calls[0].days == 90
        assert ai._expiry_scope(plan.calls[0].query) == 'upcoming'


def test_three_continuation_phrasings_requery_same_lots():
    for question in ['Hãy phân tích các lô này',
                     'Đề xuất cách xử lý cho từng lô',
                     'Lô nào cần ưu tiên xử lý nhất?']:
        plan, _ = ai._fast_intent(_request(question, _history()), MANAGER.role)
        assert plan.calls[0].name == 'expiry_alerts'
        assert plan.calls[0].days == 90
        assert ai._expiry_scope(plan.calls[0].query) == 'upcoming'
    # Do not mistake an unrelated topic for a follow-up expiry question.
    successive = _history() + [
        ChatMessage(role='user', content='Hãy phân tích các lô này'),
        ChatMessage(role='assistant', content='Đã phân tích'),
        ChatMessage(role='user', content='Đề xuất cách xử lý cho từng lô'),
        ChatMessage(role='assistant', content='Gợi ý xử lý'),
    ]
    plan, _ = ai._fast_intent(_request('Lô nào cần ưu tiên xử lý nhất?', successive), 'manager')
    assert plan.calls[0].name == 'expiry_alerts'
    assert ai._expiry_scope(plan.calls[0].query) == 'upcoming'
    successive.extend([ChatMessage(role='user', content='Có bao nhiêu nhà cung cấp?'),
                       ChatMessage(role='assistant', content='Đã trả lời')])
    assert ai._expiry_context(_request('Lô nào cần ưu tiên xử lý nhất?', successive)) is None
    assert ai._expiry_context(_request('So sánh doanh thu tháng này', _history())) is None
    assert ai._expiry_context(_request('Đề xuất cách xử lý cho từng lô')) is None


def test_expired_and_upcoming_have_distinct_scopes_and_exact_counts(setup):
    client, Session, ids, _ = setup
    with Session() as db:
        original = db.get(Batch, ids['batch'])
        # Existing fixture has a lot expiring in 30 days.
        expired = Batch(medicine_id=original.medicine_id, supplier_id=original.supplier_id,
                        code='EXPIRED-FOR-TEST', received_date=today()-timedelta(days=90),
                        expiry_date=today()-timedelta(days=11), quantity=26,
                        purchase_price=1000, sale_price=2000)
        same_day = Batch(medicine_id=original.medicine_id, supplier_id=original.supplier_id,
                         code='TODAY-FOR-TEST', received_date=today()-timedelta(days=90),
                         expiry_date=today(), quantity=13,
                         purchase_price=1000, sale_price=2000)
        db.add_all([expired, same_day]); db.commit()
        up, _ = ai._tool_expiry(db, 90, 'Các lô sắp hết hạn trong 90 ngày')
        old, _ = ai._tool_expiry(db, 90, 'Các lô đã hết hạn')
        all_lots, _ = ai._tool_expiry(db, 90, 'Tất cả cảnh báo hạn sử dụng')
        assert [x['code'] for x in up] == ['B1']
        assert {x['code'] for x in old} == {'EXPIRED-FOR-TEST', 'TODAY-FOR-TEST'}
        assert len(all_lots) == 3
        count, _ = ai._tool_alert_counts(db, 'Có bao nhiêu lô thuốc sắp hết hạn trong 90 ngày?', 90)
        assert count['count'] == 1
        # Dashboard and main /api/alerts still show overdue lots, unchanged.
        assert len(client.get('/api/alerts').json()['expiry']) == 3


def test_followup_queries_fresh_db_even_when_first_response_is_stale(setup, monkeypatch):
    from app import agent, direct_tools
    client, Session, ids, _ = setup
    monkeypatch.setattr(agent, 'prepare', lambda *a: (_ for _ in ()).throw(AssertionError('Unexpected agent planner')))
    monkeypatch.setattr(direct_tools, 'select_native_tools', lambda *a: (_ for _ in ()).throw(AssertionError('Unexpected native selector')))
    follow = _request('Đề xuất cách xử lý cho từng lô', _history())
    with Session() as db:
        first = ai._prepare_chat(db, follow, MANAGER)
        assert first['used_tools'] == ['expiry_alerts']
        assert first['results'][0]['data'][0]['quantity'] == 10
        db.get(Batch, ids['batch']).quantity = 7
        db.commit()
        second = ai._prepare_chat(db, follow, MANAGER)
        assert second['results'][0]['data'][0]['quantity'] == 7
        # Extra evidence is added to the synthesize prompt, not to old history.
        prompt = ai._stream_prompt(follow, MANAGER, second['results'])
        assert 'VERIFIED_EXPIRY_RESULT' in prompt
        assert '1 lô (sắp hết hạn: 1' in prompt
        assert 'quantity": 7' in prompt


def test_stream_endpoint_reports_verified_expiry_tool_for_second_demo_question(setup, monkeypatch):
    client, _, _, tokens = setup
    monkeypatch.setattr(ai, '_provider_stream_events',
                        lambda request, user, results: iter([('model','test-provider'),('delta','Đã nhận dữ liệu lô.' )]))
    client.cookies.set('session', tokens['manager'])
    payload = _request('Hãy tổng hợp báo cáo các lô thuốc cận hạn và đề xuất xử lý.', _history()).model_dump()
    response = client.post('/api/ai/chat/stream', json=payload)
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    meta = next(x for x in events if x['type'] == 'meta')
    assert meta['used_tools'] == ['expiry_alerts']
    assert meta['presentation']['table']['rows'][0]['code'] == 'B1'
    assert events[-1]['status'] == 'ok'
