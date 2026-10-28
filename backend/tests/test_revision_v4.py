import json
from uuid import uuid4
from sqlalchemy import select,func
from app import ai, agent
from app.models import Medicine,User,AIConversation,AIMessage
from app.schemas import AIChatRequest


def test_approval_api_removed_and_info_can_be_referenced(setup):
    c,S,ids,tokens=setup
    m=c.get('/api/medicines').json()[0]
    assert 'approval_revision' not in m
    assert c.post(f'/api/medicines/{m["id"]}/approval',json={'approved':True,'revision':'a'*64}).status_code==404
    with S() as db:
        medicine=db.get(Medicine,ids['medicine'])
        medicine.approved=False
        db.commit()
        data,source=ai._tool_medicine_search(db,'M1')
        assert data and source
        assert data[0]['information']


def test_no_approval_block_in_agent(setup,monkeypatch):
    _,S,ids,_=setup
    monkeypatch.setattr(agent,'generate_json',lambda *a,**kw:{'calls':[{'name':'inventory_search','query':'M1'}]})
    with S() as db:
        user=db.scalar(select(User).where(User.role=='manager'))
        m=db.get(Medicine,ids['medicine']);m.approved=False;db.commit()
        result=agent.prepare(db,AIChatRequest(message='Tóm tắt thông tin M1',agent=True),user)
        assert not result['terminal']
        assert any(x['tool']=='medicine_search' for x in result['results'])

def test_session_without_conversation_id_does_not_store_history(setup,monkeypatch):
    c,S,ids,_=setup
    from app.config import settings
    monkeypatch.setattr(settings,'gemini_api_key','')
    response=c.post(f'/api/ai/session/{uuid4()}/stream',json={'message':'Quy trình nhập lô?','history':[]})
    assert response.status_code==200
    events=[json.loads(line) for line in response.text.splitlines()]
    assert not any(x['type']=='title' for x in events)
    assert 'không khả dụng' in ''.join(x.get('text','') for x in events)
    with S() as db:
        for model in [AIMessage,AIConversation]:assert db.scalar(select(func.count()).select_from(model))==0
    assert c.get('/api/ai/conversations').status_code==200


def test_stop_is_owner_scoped_and_cleans_registry(setup,monkeypatch):
    from app.conversations import _running
    c,S,ids,tokens=setup
    def fake(db,request,user):
        yield json.dumps({'type':'delta','text':'first'})+'\n'
        for owner,event in _running.values():event.set()
        yield json.dumps({'type':'delta','text':'second'})+'\n'
    monkeypatch.setattr(ai,'chat_stream',fake)
    result=c.post(f'/api/ai/session/{uuid4()}/stream',json={'message':'test'})
    assert 'first' in result.text and 'second' not in result.text and not _running
    from threading import Event
    key=str(uuid4());event=Event();_running[key]=(999,event)
    try:
        c.post(f'/api/ai/session/{key}/stop',json={})
        assert not event.is_set()
    finally:_running.pop(key,None)


