import json

from app import ai
from app.config import settings


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body, ensure_ascii=False)

    def json(self):
        return self._body


def test_gemini_retries_primary_then_switches_model(monkeypatch):
    monkeypatch.setattr(settings, 'gemini_fallback_models', 'gemini-3.8-flash,gemini-3.5-flash')
    monkeypatch.setattr(settings, 'gemini_retry_attempts', 3)
    monkeypatch.setattr(settings, 'gemini_retry_base_seconds', 0.01)
    monkeypatch.setattr(ai.time_module, 'sleep', lambda *_: None)

    seen_models = []
    responses = [
        FakeResponse(503, {'error': {'message': 'high demand'}}),
        FakeResponse(503, {'error': {'message': 'high demand'}}),
        FakeResponse(503, {'error': {'message': 'high demand'}}),
        FakeResponse(200, {'steps': []}),
    ]

    def fake_post(url, headers, json, timeout):
        seen_models.append(json['model'])
        return responses.pop(0)

    monkeypatch.setattr(ai.httpx, 'post', fake_post)
    data = ai._gemini_interaction({'model': 'gemini-3.5-flash-lite', 'input': 'hello'}, timeout=1)

    assert seen_models == [
        'gemini-3.5-flash-lite',
        'gemini-3.5-flash-lite',
        'gemini-3.5-flash-lite',
        'gemini-3.8-flash',
    ]
    assert data['_app_model_used'] == 'gemini-3.8-flash'


def test_fallback_is_explicitly_data_only_not_fake_ai_analysis():
    results = [
        {
            'tool': 'expiry_alerts',
            'data': [
                {
                    'medicine_name': 'Paracetamol 500mg',
                    'code': 'LO-001',
                    'quantity': 12,
                    'unit': 'Hộp',
                    'expiry_date': '2026-10-15',
                    'days_left': 17,
                },
                {
                    'medicine_name': 'Vitamin C',
                    'code': 'LO-002',
                    'quantity': 20,
                    'unit': 'Hộp',
                    'expiry_date': '2026-12-01',
                    'days_left': 64,
                },
            ],
        },
        {
            'tool': 'procedures',
            'data': [
                {
                    'id': 1,
                    'title': 'Xử lý thuốc hết hạn',
                    'content': 'Cách ly lô hết hạn, lập biên bản và xử lý theo quy trình nội bộ.',
                }
            ],
        },
    ]

    answer = ai._fallback_chat_answer(results)

    assert 'AI tạm thời không khả dụng' in answer
    assert 'Phần phân tích/tổng hợp bằng AI chưa được thực hiện' in answer
    assert 'Paracetamol 500mg' in answer
    assert 'Quy trình nội bộ đã truy xuất' not in answer
    assert 'Đề xuất:' not in answer
    assert 'FEFO' not in answer
    assert 'expiry_alerts:' not in answer


def test_backend_router_keeps_simple_lookup_direct_and_analysis_for_ai():
    direct = ai._fallback_chat_plan('Doanh thu tháng này là bao nhiêu?', 'manager')
    assert [call.name for call in direct.calls] == ['sales_summary']
    assert ai._question_needs_ai('Doanh thu tháng này là bao nhiêu?', direct) is False

    analysis = ai._fallback_chat_plan('Phân tích tồn kho và các rủi ro đáng chú ý', 'manager')
    assert any(call.name == 'stock_risk' for call in analysis.calls)
    assert ai._question_needs_ai('Phân tích tồn kho và các rủi ro đáng chú ý', analysis) is True


def test_external_lookup_is_detected_and_not_routed_to_public_tools():
    assert ai._external_lookup_requested('Hãy tìm thông tin thuốc trên Internet') is True
    plan = ai._fallback_chat_plan('Hãy tìm thông tin thuốc trên Internet', 'manager')
    assert all(call.name != 'public_drug_sources' for call in plan.calls)


def test_stream_direct_lookup_does_not_require_gemini(setup, monkeypatch):
    c, S, ids, tokens = setup
    monkeypatch.setattr(settings, 'gemini_api_key', '')
    c.cookies.set('session', tokens['manager'])

    response = c.post('/api/ai/chat/stream', json={
        'message': 'Doanh thu tháng này là bao nhiêu?',
        'history': [],
    })
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0]['type'] == 'meta'
    assert events[0]['route'] == 'direct'
    assert events[0]['used_ai'] is False
    assert any(e.get('type') == 'delta' and 'doanh thu' in e.get('text', '').lower() for e in events)
    assert events[-1]['status'] == 'direct'


def test_stream_analysis_fallback_is_transparent_when_ai_missing(setup, monkeypatch):
    c, S, ids, tokens = setup
    monkeypatch.setattr(settings, 'gemini_api_key', '')
    c.cookies.set('session', tokens['manager'])

    response = c.post('/api/ai/chat/stream', json={
        'message': 'Phân tích tồn kho và các rủi ro đáng chú ý',
        'history': [],
    })
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0]['route'] == 'ai'
    text = ''.join(e.get('text', '') for e in events if e.get('type') == 'delta')
    assert 'AI tạm thời không khả dụng' in text
    assert 'Phần phân tích/tổng hợp bằng AI chưa được thực hiện' in text
    assert events[-1]['status'] == 'fallback_not_configured'
