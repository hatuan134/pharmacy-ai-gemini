"""Mocked API regression tests for Hybrid Fast Path / Gemini->Groq failover."""
import json
from types import SimpleNamespace
from app import ai, llm_providers, agent
from app.config import settings
from app.schemas import AIChatRequest, ChatMessage

MANAGER = SimpleNamespace(id=1, role='manager')
CASHIER = SimpleNamespace(id=2, role='cashier')


def test_fast_sales_uses_tool_without_advanced_agent(monkeypatch):
    monkeypatch.setattr(agent, 'prepare', lambda *_: (_ for _ in ()).throw(AssertionError('unneeded agent')))
    monkeypatch.setattr(ai, 'execute_chat_tools', lambda db, plan, user: (
        [{'tool':plan.calls[0].name,'data':{'revenue':'1000000','invoice_count':3}}], []))
    result = ai._prepare_chat(None, AIChatRequest(agent=True,message='Doanh thu hôm nay bao nhiêu?'), MANAGER)
    assert result['fast'] is True
    assert result['used_tools'] == ['sales_summary']
    assert result['needs_ai'] is True


def test_fast_followup_revenue_respects_history(monkeypatch):
    monkeypatch.setattr(ai, 'execute_chat_tools', lambda db, plan, user: (
        [{'tool':plan.calls[0].name,'data':{'revenue':'900000'}}], []))
    question = AIChatRequest(agent=True,message='Thế hôm qua?',history=[
        ChatMessage(role='user',content='Doanh thu hôm nay bao nhiêu?'),
        ChatMessage(role='assistant',content='Hôm nay 1.000.000 đồng.')])
    result = ai._prepare_chat(None, question, MANAGER)
    assert result['fast'] is True
    assert result['used_tools'] == ['sales_summary']


def test_role_protection_runs_before_fast_lookup(monkeypatch):
    monkeypatch.setattr(agent, 'prepare', lambda *_: (_ for _ in ()).throw(AssertionError('agent should not run')))
    denied = ai._prepare_chat(None, AIChatRequest(agent=True,message='Doanh thu hôm nay bao nhiêu?'), CASHIER)
    assert denied['terminal'] is True
    assert denied['results'] == []
    assert denied['status'] == 'access_denied'


def test_gemini_success_skips_groq(monkeypatch):
    monkeypatch.setattr(ai,'_gemini_stream_events',lambda *a: iter([('model','gemini:ok'),('delta','Đã trả lời')]))
    monkeypatch.setattr(llm_providers,'groq_stream_events',lambda *a: (_ for _ in ()).throw(AssertionError('Groq should not run')))
    events=list(ai._provider_stream_events(AIChatRequest(message='Xin chào'),MANAGER,[]))
    assert events[-1] == ('delta','Đã trả lời')


def test_gemini_network_error_then_groq_success(monkeypatch):
    monkeypatch.setattr(settings,'ai_enable_groq_fallback',True)
    monkeypatch.setattr(ai,'_gemini_stream_events',lambda *a: iter([('error','network')]))
    monkeypatch.setattr(llm_providers,'groq_stream_events',lambda *a: iter([('model','groq:openai/gpt-oss-120b'),('delta','Doanh thu là 1.000.000 ₫.')]))
    events=list(ai._provider_stream_events(AIChatRequest(message='Doanh thu hôm nay'),MANAGER,[]))
    assert ('model','groq:openai/gpt-oss-120b') in events
    assert ('delta','Doanh thu là 1.000.000 ₫.') in events
    assert not any(kind=='error' for kind,_ in events)


def test_stream_started_then_fails_never_switches_provider(monkeypatch):
    monkeypatch.setattr(settings,'ai_enable_groq_fallback',True)
    monkeypatch.setattr(ai,'_gemini_stream_events',lambda *a: iter([('delta','Đã bắt đầu'),('error','stream_interrupted')]))
    monkeypatch.setattr(llm_providers,'groq_stream_events',lambda *a: (_ for _ in ()).throw(AssertionError('must not mix providers')))
    events=list(ai._provider_stream_events(AIChatRequest(message='Hỏi'),MANAGER,[]))
    assert events == [('delta','Đã bắt đầu'),('error','stream_interrupted')]


def test_both_unavailable_never_reveals_raw_sql(monkeypatch):
    monkeypatch.setattr(settings,'ai_enable_groq_fallback',True)
    monkeypatch.setattr(ai,'_gemini_stream_events',lambda *a: iter([('error','network')]))
    monkeypatch.setattr(llm_providers,'groq_stream_events',lambda *a: iter([('error','unavailable')]))
    monkeypatch.setattr(ai,'_prepare_chat',lambda *a: {
        'terminal':False,'results':[{'tool':'dynamic_query','data':{'rows':[{'SUM(total)':1000000}]}}],
        'sources':[],'used_tools':['dynamic_query'],'needs_ai':True})
    events=[json.loads(raw) for raw in ai.chat_stream(None,AIChatRequest(message='Doanh thu?',agent=True),MANAGER)]
    answer=''.join(e.get('text','') for e in events if e['type']=='delta')
    assert 'SUM(' not in answer and '1000000' not in answer
    assert 'chưa thể kết nối' in answer
    assert events[-1]['status']=='interrupted'
    assert any(e['type']=='meta' and e.get('presentation') is None for e in events)


def test_groq_model_order(monkeypatch):
    monkeypatch.setattr(settings,'groq_api_key','test-key')
    monkeypatch.setattr(settings,'groq_models','openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b')
    assert llm_providers.groq_models()==['openai/gpt-oss-120b','qwen/qwen3.8-27b','openai/gpt-oss-20b']


def test_agent_planning_uses_groq_when_gemini_key_missing(monkeypatch):
    monkeypatch.setattr(settings,'gemini_api_key','')
    monkeypatch.setattr(llm_providers,'groq_json',lambda *a,**kw: {'calls':[{'name':'sales_summary','query':'doanh thu hôm nay'}]})
    assert agent.generate_json('allowed_tools',{'question':'doanh thu'})['calls'][0]['name']=='sales_summary'


def test_groq_native_function_call_plan(monkeypatch):
    monkeypatch.setattr(settings,'groq_api_key','mock-groq')
    monkeypatch.setattr(settings,'groq_models','openai/gpt-oss-120b')
    class FakeResponse:
        status_code=200
        def json(self):
            return {'choices':[{'message':{'tool_calls':[{'function':{
                'name':'dynamic_query','arguments':json.dumps({'query':'SELECT COUNT(*) FROM medicines'})}}]}}]}
    seen=[]
    def fake_post(*args,**kwargs):
        seen.append(kwargs['json'])
        return FakeResponse()
    monkeypatch.setattr(llm_providers.httpx,'post',fake_post)
    result=llm_providers.groq_json('Only tools',{
        'question':'Đếm thuốc', 'allowed_tools':['dynamic_query']})
    assert result['calls'][0]['name']=='dynamic_query'
    assert result['calls'][0]['query'].startswith('SELECT')
    assert seen[0]['tools'][0]['function']['name']=='dynamic_query'
    assert 'sales_summary' not in str(seen[0]['tools'])
