import json
import pytest
from uuid import uuid4
from sqlalchemy import select
from app import ai
from app.models import AIMessage, AIConversation


@pytest.mark.parametrize('ending', ['complete', 'interrupted', 'stopped'])
def test_durable_stream_and_history(setup, monkeypatch, ending):
    from app.conversations import _running
    client, Session, ids, tokens = setup
    cid = client.post('/api/ai/conversations', json={}).json()['id']
    rid = str(uuid4())
    presentation = {'cards': [{'label': 'Tồn', 'value': 10}],
                    'table': {'columns': [{'key': 'name', 'label': 'Thuốc'}], 'rows': [{'name': 'Test'}]},
                    'actions': [{'page': 'batches', 'label': 'Xem kho'}]}

    def fake_stream(db, request, user):
        yield json.dumps({'type': 'meta', 'presentation': presentation})+'\n'
        yield json.dumps({'type': 'delta', 'text': 'Nội dung đã nhận'})+'\n'
        with Session() as check:
            msg = check.scalars(select(AIMessage).where(AIMessage.conversation_id == cid, AIMessage.role == 'assistant')).one()
            assert msg.content == 'Nội dung đã nhận'
            assert json.loads(msg.metadata_json)['presentation'] == presentation
        if ending == 'interrupted':
            raise ConnectionError('simulated network failure')
        if ending == 'stopped':
            _running[rid][1].set()
        yield json.dumps({'type': 'done', 'status': 'ok'})+'\n'

    monkeypatch.setattr(ai, 'chat_stream', fake_stream)
    response = client.post(f'/api/ai/session/{rid}/stream', json={'message': 'Xem kho', 'conversation_id': cid})
    assert response.status_code == 200
    for _ in range(2):
        msg = client.get(f'/api/ai/conversations/{cid}').json()['messages'][-1]
        assert msg['content'] == 'Nội dung đã nhận'
        assert msg['presentation'] == presentation
        assert msg['status'] == ending
    with Session() as check:
        assert check.get(AIConversation, cid).busy is False


def test_old_empty_history_is_not_fabricated(setup):
    client, Session, ids, tokens = setup
    cid = client.post('/api/ai/conversations', json={}).json()['id']
    with Session() as db:
        db.add(AIMessage(conversation_id=cid, role='assistant', content='', status='interrupted', metadata_json='invalid legacy json'))
        db.commit()
    msg = client.get(f'/api/ai/conversations/{cid}').json()['messages'][-1]
    assert msg['content'] == ''
    assert msg['status'] == 'interrupted'
    assert msg['presentation'] is None


def test_closed_request_session_and_disconnect(setup, monkeypatch):
    from app import conversations
    from app.models import User
    from app.schemas import AIChatRequest
    client, Session, ids, tokens = setup
    cid = client.post('/api/ai/conversations', json={}).json()['id']
    def fake_stream(db, request, user):
        yield json.dumps({'type': 'delta', 'text': 'Đã gửi trước khi mất mạng'})+'\n'
        yield json.dumps({'type': 'done', 'status': 'ok'})+'\n'
    monkeypatch.setattr(ai, 'chat_stream', fake_stream)
    monkeypatch.setattr(conversations, 'StreamingResponse', lambda content, **kwargs: content)
    with Session() as db:
        user = db.scalars(select(User).where(User.role == 'manager')).one()
        iterator = conversations.stream(uuid4(), AIChatRequest(message='Xem kho', conversation_id=cid), user, db)
    # Simulate FastAPI dependency cleanup before the response iterator starts.
    assert json.loads(next(iterator))['type'] == 'message'
    assert json.loads(next(iterator))['type'] == 'delta'
    iterator.close()
    msg = client.get(f'/api/ai/conversations/{cid}').json()['messages'][-1]
    assert msg['content'] == 'Đã gửi trước khi mất mạng'
    assert msg['status'] == 'interrupted'
    with Session() as db:
        assert db.get(AIConversation, cid).busy is False
