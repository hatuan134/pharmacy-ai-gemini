"""Verify the UI never relies on model-provided numbers/links and scopes saved chats."""
from uuid import uuid4
from app.ai_presentation import presentation
from app.auth import token
from app.models import User


def test_verified_sales_cards_and_role_guard():
    result = [{'tool': 'sales_summary', 'data': {
        'revenue': '250000.00', 'invoice_count': 2,
        'invoice_totals': [{'id': 10, 'total': '75000'}, {'id': 12, 'total': '175000'}]}}]
    manager = presentation(result, 'manager')
    assert manager['cards'][0]['value'] == 250000
    assert manager['cards'][1]['value'] == 2
    assert manager['chart'] is None  # no chart data is generated after V6
    assert any(action['page'] == 'reports' for action in manager['actions'])
    cashier = presentation(result, 'cashier')
    assert cashier['cards'] == []
    assert cashier['chart'] is None
    assert not any(a['page'] == 'reports' for a in cashier['actions'])



def test_stock_risk_still_offers_details_without_chart():
    result = [{'tool': 'stock_risk', 'data': [
        {'medicine': 'Thuốc A', 'stock': 9, 'sold_30d': 2, 'nearest_expiry_days': 18}
    ]}]
    data = presentation(result, 'pharmacist')
    assert data['chart'] is None
    assert data['table']['rows'][0]['medicine'] == 'Thuốc A'
    assert data['actions'][0]['page'] == 'batches'

def test_verified_inventory_renders_safe_columns_and_route():
    result = [{'tool': 'inventory_search', 'data': [{
        'medicine': 'Thuốc A', 'batch': 'LOT1', 'quantity': 9,
        'expiry_date': '2027-12-12', 'purchase_price': '34000', 'supplier': 'ABC'}]}]
    pharmacist = presentation(result, 'pharmacist')
    assert pharmacist['table']['rows'][0]['quantity'] == '9'
    assert 'purchase_price' not in pharmacist['table']['rows'][0]
    assert pharmacist['actions'][0]['page'] == 'batches'
    cashier = presentation(result, 'cashier')
    assert cashier['actions'][0]['page'] == 'medicines'


def test_role_scope_prevents_reading_history_after_role_change(setup):
    client, Session, ids, tokens = setup
    client.cookies.set('session', tokens['pharmacist'])
    cid = client.post('/api/ai/conversations', json={}).json()['id']
    client.cookies.set('session', tokens['manager'])
    with Session() as db:
        pharmacist = db.query(User).filter(User.role == 'pharmacist').one()
        uid = pharmacist.id
    response = client.patch(f'/api/users/{uid}', json={'role': 'cashier'})
    assert response.status_code == 200
    with Session() as db:
        user = db.get(User, uid)
        new_token = token(user)
    client.cookies.set('session', new_token)
    assert client.get('/api/ai/conversations').json() == []
    assert client.get(f'/api/ai/conversations/{cid}').status_code == 404
    assert client.post(f'/api/ai/session/{uuid4()}/stream', json={
        'conversation_id': cid, 'message': 'Xem thông tin cũ'}).status_code == 404
    newer = client.post('/api/ai/conversations', json={})
    assert newer.status_code == 201 and newer.json()['id'] != cid
    client.cookies.set('session', tokens['manager'])
    assert client.patch(f'/api/users/{uid}', json={'role': 'pharmacist'}).status_code == 200
    with Session() as db:
        back_token = token(db.get(User, uid))
    client.cookies.set('session', back_token)
    assert client.get(f'/api/ai/conversations/{cid}').status_code == 200
    assert all(c['id'] != newer.json()['id'] for c in client.get('/api/ai/conversations').json())


def test_chat_ui_meta_is_stored_for_reopen(setup, monkeypatch):
    import json
    from app import ai
    client, Session, ids, tokens = setup
    cid = client.post('/api/ai/conversations', json={}).json()['id']
    verified = {'cards': [{'label': 'Số thuốc', 'value': 1, 'format': 'number'}],
                'table': None, 'chart': None, 'actions': [{'page': 'medicines', 'label': 'Xem danh mục thuốc'}]}
    def fake_stream(db, request, user):
        yield json.dumps({'type':'meta','sources':[],'used_tools':['dynamic_query'],
                          'route':'ai','presentation':verified})+'\n'
        yield json.dumps({'type':'delta','text':'Trong hệ thống có 1 thuốc.'})+'\n'
        yield json.dumps({'type':'done','status':'ok','model':'gemini-test'})+'\n'
    monkeypatch.setattr(ai, 'chat_stream', fake_stream)
    response = client.post(f'/api/ai/session/{uuid4()}/stream', json={'message':'Có bao nhiêu thuốc?', 'conversation_id':cid})
    assert response.status_code == 200
    detail = client.get(f'/api/ai/conversations/{cid}').json()
    assert detail['messages'][-1]['presentation'] == verified


def test_migration_adds_role_to_legacy_conversations():
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.orm import Session
    from app.db import Base
    from app.migrations import apply
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE ai_conversations DROP COLUMN role_scope')
        connection.exec_driver_sql("INSERT INTO users (id, username, name, password_hash, role, active, token_version) VALUES (1, 'demo', 'Demo', 'hash', 'pharmacist', 1, 0)")
        connection.exec_driver_sql("INSERT INTO ai_conversations (id, user_id, title, created_at, updated_at, busy) VALUES (1, 1, 'Cũ', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)")
    with Session(engine) as db:
        apply(db)
        assert db.execute(text('SELECT role_scope FROM ai_conversations WHERE id=1')).scalar_one() == 'pharmacist'
        apply(db)
        assert db.execute(text('SELECT count(*) FROM ai_conversations')).scalar_one() == 1
    engine.dispose()


def test_real_stream_meta_contains_ui_from_verified_results(setup, monkeypatch):
    import json
    from app import ai
    client, Session, ids, tokens = setup
    def prepared(db, request, user):
        return {'terminal': False, 'message': None, 'status': None,
                'results': [{'tool': 'low_stock', 'data': [{'id': 12,'name': 'Thuốc thiếu hàng', 'available_stock': 1, 'min_stock': 10}]}],
                'sources': [], 'used_tools': ['low_stock'], 'needs_ai': False}
    monkeypatch.setattr(ai, '_prepare_chat', prepared)
    response = client.post('/api/ai/chat/stream', json={'message': 'Thuốc nào đang tồn thấp?'})
    assert response.status_code == 200
    packets = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    meta = next(p for p in packets if p['type'] == 'meta')
    assert meta['presentation']['table']['rows'][0]['name'] == 'Thuốc thiếu hàng'
    assert meta['presentation']['actions'][0]['page'] == 'alerts'
    assert all(p['type'] != 'model' for p in packets)
