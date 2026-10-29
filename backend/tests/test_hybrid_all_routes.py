"""Regression coverage for expanded Fast Path, native tool routing and answer focus."""
import json
from types import SimpleNamespace
from app import ai, direct_tools, llm_providers
from app.config import settings
from app.schemas import AIChatRequest, ChatMessage
from app.ai_presentation import presentation

MANAGER=SimpleNamespace(id=1,role='manager')
PHARMACIST=SimpleNamespace(id=2,role='pharmacist')
CASHIER=SimpleNamespace(id=3,role='cashier')


def requested(message, role='manager'):
    plan=ai._fast_intent(AIChatRequest(message=message,agent=True),role)
    return [c.name for c in plan[0].calls] if plan else None


def test_fast_routes_cover_business_data():
    examples={
        'Thông tin thuốc Paracetamol': 'medicine_search',
        'Thuốc Paracetamol còn bao nhiêu hộp?': 'stock_balance',
        'Thuốc nào tồn kho thấp?': 'low_stock',
        'Những lô thuốc sắp hết hạn trong 20 ngày?': 'expiry_alerts',
        'Thuốc nào bán chậm?': 'stock_risk',
        'Cho tôi danh sách nhà cung cấp': 'system_data',
        'Danh sách nhóm thuốc': 'system_data',
        'Chi tiết hóa đơn 123': 'invoice_records',
        'Doanh thu hôm nay': 'sales_summary',
        'Có bao nhiêu loại thuốc trong nhà thuốc?': 'overview_counts',
        'Có bao nhiêu nhà cung cấp?': 'overview_counts',
        'Hôm nay có bao nhiêu hóa đơn?': 'sales_summary',
        'Có bao nhiêu thuốc sắp hết hạn trong 30 ngày?': 'alert_counts',
        'Thuốc Paracetamol nằm trong lô nào?': 'inventory_search',
    }
    for message, tool in examples.items():
        assert requested(message)==[tool],message


def test_chat_without_data_uses_no_tools():
    assert requested('Xin chào') == []
    assert requested('Hướng dẫn tôi nhập lô thuốc') == []
    assert requested('Bạn có thể trò chuyện với tôi không?') == []


def test_complex_questions_stay_in_advanced_agent():
    for message in ('Phân tích thuốc tồn kho nhiều và bán chậm',
                    'So sánh doanh thu hôm nay với hôm qua',
                    'Top 5 loại thuốc doanh thu cao nhất'):
        assert ai._needs_advanced_agent(AIChatRequest(message=message))


def test_fast_path_does_not_invoke_planner_or_native_model(monkeypatch):
    from app import agent
    monkeypatch.setattr(agent,'prepare',lambda *a: (_ for _ in ()).throw(AssertionError('advanced called')))
    monkeypatch.setattr(direct_tools,'select_native_tools',lambda *a: (_ for _ in ()).throw(AssertionError('native called')))
    monkeypatch.setattr(ai,'execute_chat_tools',lambda db,plan,user: ([{'tool':plan.calls[0].name,'data':{}}],[]))
    result=ai._prepare_chat(None,AIChatRequest(message='Có bao nhiêu loại thuốc?',agent=True),MANAGER)
    assert result['fast'] and result['used_tools']==['overview_counts']


def test_cashier_cannot_use_unauthorized_aggregate():
    result=ai._prepare_chat(None,AIChatRequest(message='Doanh thu hôm nay là bao nhiêu?',agent=True),CASHIER)
    assert result['terminal'] and result['status']=='access_denied'
    assert requested('Có bao nhiêu nhà cung cấp?',role='cashier') is None


def test_gemini_native_function_selection_and_scope(monkeypatch):
    monkeypatch.setattr(settings,'gemini_api_key','fake-gemini')
    class Response:
        status_code=200
        def json(self):
            return {'candidates':[{'content':{'parts':[
                {'functionCall':{'name':'inventory_search','args':{'query':'Paracetamol'}}},
                {'functionCall':{'name':'sales_summary','args':{'query':'Doanh thu'}}},
            ]}}]}
    seen=[]
    def fake_post(*args,**kwargs):
        seen.append(kwargs['json'])
        return Response()
    monkeypatch.setattr(direct_tools.httpx,'post',fake_post)
    result=direct_tools.select_native_tools(AIChatRequest(message='Kiểm tra thông tin hàng',agent=True),CASHIER)
    assert result==[{'name':'inventory_search','query':'Paracetamol','days':90}]
    assert 'sales_summary' not in [x['name'] for x in seen[0]['tools'][0]['functionDeclarations']]
    assert 'dynamic_query' not in [x['name'] for x in seen[0]['tools'][0]['functionDeclarations']]


def test_native_groq_failover_selects_business_tool(monkeypatch):
    monkeypatch.setattr(settings,'gemini_api_key','')
    monkeypatch.setattr(settings,'groq_api_key','fake-groq')
    monkeypatch.setattr(settings,'ai_enable_groq_fallback',True)
    def groq(*args,**kwargs):
        return {'calls':[{'name':'medicine_search','query':'Paracetamol'}]}
    monkeypatch.setattr(direct_tools,'groq_json',groq)
    result=direct_tools.select_native_tools(AIChatRequest(message='Bạn có thông tin gì về loại này?',agent=True),PHARMACIST)
    assert result and result[0]['name']=='medicine_search'


def test_native_malformed_sql_and_role_calls_dropped():
    calls=[{'name':'dynamic_query','query':'DELETE FROM invoices'},
           {'name':'sales_summary','query':'Doanh thu'},
           {'name':'inventory_search','query':'Vitamin'}]
    selected=direct_tools._normalize(calls,direct_tools._allowed('cashier'),'x')
    assert [x['name'] for x in selected]==['inventory_search']


def test_native_no_tools_for_data_question_goes_to_agent(monkeypatch):
    monkeypatch.setattr(settings,'gemini_api_key','fake')
    monkeypatch.setattr(direct_tools,'_gemini_calls',lambda *a: [])
    assert direct_tools.select_native_tools(AIChatRequest(message='Hệ thống còn bao nhiêu thuốc?',agent=True),MANAGER) is None


def test_native_chat_does_not_need_agent(monkeypatch):
    from app import agent
    monkeypatch.setattr(direct_tools,'select_native_tools',lambda *a: [])
    monkeypatch.setattr(agent,'prepare',lambda *a: (_ for _ in ()).throw(AssertionError('planner called')))
    prepared=ai._prepare_chat(None,AIChatRequest(message='Tôi đang sử dụng máy tính, bạn có giúp tôi không?',agent=True),MANAGER)
    assert not prepared['results'] and (prepared.get('route')=='conversation' or prepared.get('fast'))


def test_first_metric_no_extra_presentation():
    revenue=[{'tool':'sales_summary','data':{'revenue':'1000000','invoice_count':3}}]
    assert presentation(revenue,'manager','Doanh thu hôm nay?')['cards']==[]
    assert presentation(revenue,'manager','Hôm nay bao nhiêu hóa đơn?')['cards']==[]
    assert len(presentation(revenue,'manager','Phân tích doanh thu tháng này')['cards'])==2


def test_context_followup_only_if_last_subject_clear():
    request=AIChatRequest(message='Thế hôm qua?',history=[
        ChatMessage(role='user',content='Doanh thu tháng này là bao nhiêu?'),
        ChatMessage(role='assistant',content='1 triệu'),
    ])
    assert ai._fast_intent(request,'manager')[0].calls[0].name=='sales_summary'
    # No assumed revenue when recent history only concerns a medicine.
    request.history=[ChatMessage(role='user',content='Thuốc A còn bao nhiêu?')]
    assert ai._fast_intent(request,'manager') is None


def test_focus_prompt_is_scope_aware():
    req=AIChatRequest(message='Doanh thu hôm nay?')
    prompt=ai._stream_prompt(req,MANAGER,[{'tool':'sales_summary','data':{'revenue':'1000000'}}])
    assert 'ONE_NUMBER_REVENUE' in prompt
    assert 'TOOL_RESULTS' in prompt


def test_count_tool_uses_actual_database(setup):
    from app.models import Medicine
    client,Session,ids,tokens=setup
    with Session() as db:
        total=ai._tool_overview_counts(db,'Có bao nhiêu loại thuốc?', 'manager')[0]
        assert total['count']==db.query(Medicine).count()
        assert total['metric']=='Loại thuốc'
        forbidden=ai._tool_overview_counts(db,'Có bao nhiêu nhà cung cấp?', 'cashier')[0]
        assert forbidden=={}


def test_stock_balance_exact_and_expiry_counts(setup):
    from app.models import Medicine, Batch
    from app.config import today
    from sqlalchemy import select, func
    client,Session,ids,tokens=setup
    with Session() as db:
        med=db.scalars(select(Medicine).order_by(Medicine.id)).first()
        value, sources=ai._tool_stock_balance(db,f'Thuốc {med.code} còn bao nhiêu?', 'manager')
        assert value['matches']
        assert any(row['code']==med.code for row in value['matches'])
        for row in value['matches']:
            matching=db.scalars(select(Medicine).where(Medicine.code==row['code'])).first()
            expected=int(db.scalar(select(func.sum(Batch.quantity)).where(
                Batch.medicine_id==matching.id, Batch.quantity>0,
                Batch.expiry_date>today())) or 0)
            assert row['available_stock']==expected
        cnt,_=ai._tool_alert_counts(db,'Có bao nhiêu lô thuốc sắp hết hạn trong 90 ngày?',90)
        from app.services import alerts
        assert cnt['count']==len(alerts(db,90)['expiry'])
