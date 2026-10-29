"""Regression checks for screen parity and the latest streamed AI workflow.

These use an isolated SQLite fixture and mocked language models.  They must
never read the user's real .env or require a live Gemini/Groq API key.
"""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import event, select

from app import ai
from app.models import AIMessage, Invoice, User
from app.screen_detail import invoice_records


def test_invoice_list_limit_is_explicit_but_exact_receipt_is_accessible(setup):
    """AI must not silently claim a truncated invoice list is all invoices."""
    _, Session, _, _ = setup
    with Session() as db:
        manager = db.scalars(select(User).where(User.role == 'manager')).one()
        for i in range(35):
            db.add(Invoice(
                user_id=manager.id, customer=f'Khách {i}',
                created_at=datetime.now(timezone.utc), total=1000 + i,
                status='paid', payment_method='cash',
                request_key=f'ai-regression-{uuid4()}', request_hash='test-only-hash',
            ))
        db.commit()
        cashier = SimpleNamespace(id=3, role='cashier')
        listed, _ = invoice_records(db, 'Danh sách hóa đơn', cashier)
        assert listed['total'] == 35
        assert listed['shown'] == 30
        assert listed['truncated'] is True
        assert len(listed['records']) == 30
        assert all('request_key' not in row and 'request_hash' not in row
                   for row in listed['records'])
        oldest = db.scalars(select(Invoice).order_by(Invoice.id)).first()
        exact, _ = invoice_records(db, f'Chi tiết hóa đơn số {oldest.id}', cashier)
        assert exact['total'] == 1
        assert exact['truncated'] is False
        assert exact['records'][0]['id'] == oldest.id


def test_unauthorized_report_is_blocked_even_with_tool_suggestion(setup):
    """A tool name proposed by a model must not confer any backend authority."""
    from app import direct_tools
    from app.schemas import AIChatRequest
    cashier = SimpleNamespace(id=3, role='cashier')
    candidate = [
        {'name': 'report_view', 'query': 'Báo cáo doanh thu'},
        {'name': 'staff_records', 'query': 'Danh sách nhân viên'},
        {'name': 'inventory_search', 'query': 'Paracetamol'},
    ]
    normalized = direct_tools._normalize(candidate, direct_tools._allowed('cashier'), 'x')
    assert [item['name'] for item in normalized] == ['inventory_search']
    _, Session, _, _ = setup
    with Session() as db:
        prepared = ai._prepare_chat(db, AIChatRequest(message='Báo cáo doanh thu toàn nhà thuốc'), cashier)
    assert prepared['terminal'] is True
    assert prepared['status'] == 'access_denied'


def test_stream_saves_small_deltas_in_batches_and_finalizes(setup, monkeypatch):
    """Avoid one database COMMIT per token; always store the final answer."""
    client, Session, _, _ = setup
    conversation_id = client.post('/api/ai/conversations', json={}).json()['id']
    def fake_stream(db, request, user):
        for _ in range(40):
            yield json.dumps({'type': 'delta', 'text': 'a'}) + '\n'
        yield json.dumps({'type': 'done', 'status': 'ok', 'model': 'mock-provider'}) + '\n'
    monkeypatch.setattr(ai, 'chat_stream', fake_stream)
    commits = []
    def count_commit(session):
        commits.append(1)
    event.listen(Session.class_, 'after_commit', count_commit)
    try:
        answer = client.post(f'/api/ai/session/{uuid4()}/stream', json={
            'message': 'Hỏi tồn kho', 'conversation_id': conversation_id,
        })
    finally:
        event.remove(Session.class_, 'after_commit', count_commit)
    assert answer.status_code == 200
    with Session() as db:
        message = db.scalars(select(AIMessage).where(
            AIMessage.conversation_id == conversation_id,
            AIMessage.role == 'assistant',
        )).one()
        assert message.status == 'complete'
        assert message.content == 'a' * 40
    # Old behavior committed 40 times; the new implementation batches them.
    assert len(commits) < 15


def test_groq_stream_404_tries_next_model_and_stops_on_success(monkeypatch):
    """Unavailable Groq model is skipped; text is produced by only one model."""
    from app import llm_providers
    from app.config import settings
    monkeypatch.setattr(settings, 'groq_api_key', 'test-only-key')
    monkeypatch.setattr(settings, 'groq_models', 'openai/gpt-oss-120b,openai/gpt-oss-20b')
    attempted = []

    class FakeStream:
        def __init__(self, status_code, payloads=()):
            self.status_code = status_code
            self.payloads = payloads
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def iter_lines(self):
            return iter(self.payloads)

    def fake_stream(method, url, **kwargs):
        attempted.append(kwargs['json']['model'])
        if len(attempted) == 1:
            return FakeStream(404)
        return FakeStream(200, [
            'data: {"choices":[{"delta":{"content":"Đã trả lời."}}]}',
            'data: [DONE]',
        ])

    monkeypatch.setattr(llm_providers.httpx, 'stream', fake_stream)
    result = list(llm_providers.groq_stream_events('Bạn là An Tâm AI', 'Xin chào'))
    assert attempted == ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']
    assert result == [('model', 'groq:openai/gpt-oss-20b'), ('delta', 'Đã trả lời.')]


def test_groq_stream_never_stitches_two_models_after_text(monkeypatch):
    """A partial reply must never be followed by another model's continuation."""
    import httpx
    from app import llm_providers
    from app.config import settings
    monkeypatch.setattr(settings, 'groq_api_key', 'test-only-key')
    monkeypatch.setattr(settings, 'groq_models', 'openai/gpt-oss-120b,openai/gpt-oss-20b')
    calls = []

    class InterruptedStream:
        status_code = 200
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Bắt đầu "}}]}'
            raise httpx.ReadTimeout('simulated read timeout')

    def fake_stream(method, url, **kwargs):
        calls.append(kwargs['json']['model'])
        return InterruptedStream()

    monkeypatch.setattr(llm_providers.httpx, 'stream', fake_stream)
    result = list(llm_providers.groq_stream_events('Bạn là An Tâm AI', 'Xin chào'))
    assert calls == ['openai/gpt-oss-120b']
    assert result == [('model', 'groq:openai/gpt-oss-120b'),
                      ('delta', 'Bắt đầu '), ('error', 'stream_interrupted')]


def test_groq_401_is_not_retried_with_other_models(monkeypatch):
    """An invalid API key is provider-wide, not a model-specific 404."""
    from app import llm_providers
    from app.config import settings
    monkeypatch.setattr(settings, 'groq_api_key', 'invalid-test-key')
    monkeypatch.setattr(settings, 'groq_models', 'openai/gpt-oss-120b,openai/gpt-oss-20b')
    calls = []

    class Unauthorized:
        status_code = 401
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    def fake_stream(method, url, **kwargs):
        calls.append(kwargs['json']['model'])
        return Unauthorized()

    monkeypatch.setattr(llm_providers.httpx, 'stream', fake_stream)
    assert list(llm_providers.groq_stream_events('system', 'prompt')) == [('error', 'auth')]
    assert calls == ['openai/gpt-oss-120b']
