"""Persistent chat conversations must remain private between staff accounts."""
import json
from uuid import uuid4
from sqlalchemy import select
from app import ai
from app.models import AIConversation, AIMessage


def test_conversation_crud_and_account_isolation(setup):
    client, Session, ids, tokens = setup
    created = client.post('/api/ai/conversations',json={})
    assert created.status_code == 201
    cid = created.json()['id']
    assert client.get('/api/ai/conversations').json()[0]['id'] == cid
    assert client.get(f'/api/ai/conversations/{cid}').json()['messages'] == []
    renamed = client.patch(f'/api/ai/conversations/{cid}',json={'title':'Thuốc sắp hết hạn'})
    assert renamed.status_code == 200
    assert renamed.json()['title'] == 'Thuốc sắp hết hạn'

    client.cookies.set('session',tokens['pharmacist'])
    assert client.get('/api/ai/conversations').json() == []
    assert client.get(f'/api/ai/conversations/{cid}').status_code == 404
    assert client.patch(f'/api/ai/conversations/{cid}',json={'title':'Vượt quyền'}).status_code == 404
    assert client.delete(f'/api/ai/conversations/{cid}').status_code == 404
    client.cookies.set('session',tokens['manager'])
    assert client.delete(f'/api/ai/conversations/{cid}').json()['deleted'] is True
    assert client.get(f'/api/ai/conversations/{cid}').status_code == 404


def test_history_stream_persists_response_and_server_history(setup, monkeypatch):
    client, Session, ids, tokens = setup
    cid = client.post('/api/ai/conversations',json={}).json()['id']
    captured=[]
    def fake_stream(db,request,user):
        captured.append([{'role':m.role,'content':m.content} for m in request.history])
        yield json.dumps({'type':'meta','sources':[],'used_tools':['inventory_search'],'route':'direct'})+'\n'
        yield json.dumps({'type':'delta','text':'Lô còn hạn'})+'\n'
        yield json.dumps({'type':'done','status':'ok','model':'backend-direct'})+'\n'
    monkeypatch.setattr(ai,'chat_stream',fake_stream)
    for msg in ['Lô thuốc nào sắp hết hạn?', 'Vậy còn tồn bao nhiêu?']:
        response=client.post(f'/api/ai/session/{uuid4()}/stream',json={'message':msg,'conversation_id':cid,'history':[{'role':'user','content':'GIẢ MẠO LỊCH SỬ'}]})
        assert response.status_code==200
        assert any(p.get('text')=='Lô còn hạn' for p in map(json.loads,response.text.splitlines()))
    detail=client.get(f'/api/ai/conversations/{cid}').json()
    assert detail['title']=='Lô thuốc nào sắp hết hạn?'
    assert [(m['role'],m['content']) for m in detail['messages']]==[
        ('user','Lô thuốc nào sắp hết hạn?'),('assistant','Lô còn hạn'),
        ('user','Vậy còn tồn bao nhiêu?'),('assistant','Lô còn hạn')]
    assert captured[0] == []
    assert captured[1] == [{'role':'user','content':'Lô thuốc nào sắp hết hạn?'},{'role':'assistant','content':'Lô còn hạn'}]
    assert all('GIẢ MẠO' not in repr(x) for x in captured)
    with Session() as db:
        assert db.get(AIConversation,cid).busy is False


def test_cannot_stream_to_other_users_conversation(setup):
    client, Session, ids, tokens=setup
    cid=client.post('/api/ai/conversations',json={}).json()['id']
    client.cookies.set('session',tokens['cashier'])
    assert client.post(f'/api/ai/session/{uuid4()}/stream',json={'message':'Thử vượt quyền','conversation_id':cid}).status_code == 404
    with Session() as db:
        assert db.scalars(select(AIMessage).where(AIMessage.conversation_id==cid)).all() == []
