"""AI sees the same role-visible screen data, without exposing unauthorized fields."""
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4
import pytest
from app import ai
from app.agent_query import schema
from app.ai_presentation import presentation
from app.screen_tools import report_view, dashboard_view
from app.schemas import AIChatRequest
from app.config import today

M = SimpleNamespace(id=1,role='manager')
P = SimpleNamespace(id=2,role='pharmacist')
C = SimpleNamespace(id=3,role='cashier')


def test_fast_report_and_med_detail_intents():
    for question, name in (
        ('Top 5 thuốc bán chạy trong 30 ngày', 'report_view'),
        ('Giá trị tồn kho hiện tại bao nhiêu?', 'report_view'),
        ('Xu hướng doanh thu từng ngày trong 7 ngày', 'report_view'),
        ('Tình hình tồn kho theo báo cáo', 'report_view'),
        ('Thuốc AT030 có ngưỡng cảnh báo tồn bao nhiêu?', 'medicine_search'),
        ('Nguồn thông tin thuốc AT030 là gì?', 'medicine_search'),
        ('Báo cáo hoạt động tháng này', 'report_view'),
    ):
        selection = ai._fast_intent(AIChatRequest(message=question,agent=True), 'manager')
        assert selection is not None, question
        assert [call.name for call in selection[0].calls] == [name], question


def test_role_scoped_tool_names_and_no_credentials():
    assert 'report_view' in ai._chat_allowed_tools('manager')
    for role in ('cashier', 'pharmacist'):
        assert 'report_view' not in ai._chat_allowed_tools(role)
        assert ai._role_access_denied_reason('Top 5 thuốc bán chạy trong tháng',role)
        assert ai._role_access_denied_reason('Giá trị tồn kho hiện tại',role)
        assert ai._role_access_denied_reason('Xem báo cáo hoạt động',role)
    assert 'dashboard_view' in ai._chat_allowed_tools('cashier')
    assert 'password_hash' not in schema('manager')['users']
    assert 'token_version' not in schema('manager')['users']
    assert 'information' in schema('manager')['medicines']
    assert 'source' in schema('pharmacist')['medicines']
    assert 'phone' in schema('pharmacist')['suppliers']
    assert 'movements' in schema('pharmacist')
    assert 'invoices' not in schema('pharmacist')


def test_manager_report_equal_ui_and_top_five(setup):
    client,Session,ids,_ = setup
    purchase=client.post('/api/invoices', json={
        'request_key':str(uuid4()),
        'items':[{'batch_id':ids['batch'],'quantity':2,'expected_sale_price':2000}]
    })
    assert purchase.status_code == 200
    api_report=client.get('/api/reports').json()
    with Session() as db:
        top,_=report_view(db,'Top 5 thuốc bán chạy trong 30 ngày',M)
        stock,_=report_view(db,'Giá trị tồn kho',M)
        trend,_=report_view(db,'Biểu đồ doanh thu 30 ngày',M)
    assert len(top['top_medicines']) == 1
    assert top['top_medicines'][0]['quantity']==api_report['top_medicines'][0]['quantity']
    assert Decimal(stock['stock_value'])==Decimal(str(api_report['stock_value']))
    assert len(trend['daily_revenue'])==len(api_report['series'])
    assert trend['period']['end']==str(today())
    assert (today()- __import__('datetime').date.fromisoformat(trend['period']['start'])).days==29
    assert 'top_medicines' not in stock  # focus: don't send unrelated metrics
    assert 'stock_value' not in top


def test_report_denied_even_when_tool_called_directly(setup):
    _,Session,_,_ = setup
    for user in (P,C):
        with Session() as db:
            with pytest.raises(ValueError,match='Chỉ Quản lý'):
                report_view(db,'Thuốc bán chạy',user)
            for item in ('report_view',):
                assert item not in ai._chat_allowed_tools(user.role)


def test_medicine_details_matches_edit_form(setup):
    client,Session,ids,_ = setup
    row=client.get('/api/medicines').json()[0]
    with Session() as db:
        details,_=ai._tool_medicine_search(db,'M1')
    assert details
    item=details[0]
    for key in ('code','name','min_stock','prescription_required','active','information','source'):
        assert item[key]==row[key]
    assert item['category']=='Test' and item['unit']=='Hộp'


def test_cashier_only_own_shift_dashboard(setup):
    client,Session,ids,tokens=setup
    # Manager's sale must not count toward cashier's shift.
    sale=client.post('/api/invoices',json={
        'request_key':str(uuid4()),
        'items':[{'batch_id':ids['batch'],'quantity':1,'expected_sale_price':2000}]
    })
    assert sale.status_code == 200
    with Session() as db:
        result,_=dashboard_view(db,'Doanh số ca hôm nay của tôi',C)
    assert Decimal(result['today_sales'])==0
    assert result['scope'].startswith('Chỉ')
    assert ai._role_access_denied_reason('Doanh thu hôm nay của nhà thuốc', 'cashier')
    assert ai._role_access_denied_reason('Doanh thu ca hôm nay của tôi','cashier') == ''
    plan = ai._fast_intent(AIChatRequest(message='Doanh thu ca hôm nay của tôi'), 'cashier')
    assert plan and [c.name for c in plan[0].calls] == ['dashboard_view']


def test_verified_presentation_is_role_scoped():
    sample=[{'tool':'report_view','data':{'top_medicines':[{'name':'X','quantity':1,'unit':'Hộp','revenue':'1000'}]}}]
    assert presentation(sample,'manager','Top 5 thuốc bán chạy')['table'] is not None
    assert presentation(sample,'cashier','Top 5 thuốc bán chạy')['table'] is None

def test_active_counts_match_visible_dashboard(setup):
    client, Session, ids, _ = setup
    with Session() as db:
        active,_=ai._tool_overview_counts(db,'Có bao nhiêu thuốc đang kinh doanh?','manager')
        assert active['count']==client.get('/api/dashboard').json()['medicine_count']
        # Cashier may count medicines, but never account names.
        assert 'users' not in schema('cashier')
        staff,_=ai._tool_overview_counts(db,'Có bao nhiêu tài khoản đang hoạt động?','manager')
        assert staff['count']==client.get('/api/dashboard').json()['staff_count']


def test_report_chart_and_medicine_source_not_shown_as_extra_cards(setup):
    client,Session,ids,_=setup
    with Session() as db:
        prepared=ai._prepare_chat(db, AIChatRequest(message='Giá trị tồn kho hiện tại bao nhiêu?',agent=True),M)
        assert prepared['fast'] and prepared['used_tools']==['report_view']
        assert list(prepared['results'][0]['data'])==['period','stock_value','stock_value_note']
        manager_ui=presentation(prepared['results'],'manager','Giá trị tồn kho hiện tại bao nhiêu?')
        assert manager_ui['table'] is None

def test_cashier_inventory_kpis_are_visible_without_own_revenue(setup):
    _,Session,_,_ = setup
    with Session() as db:
        batches,_=dashboard_view(db,'Có bao nhiêu lô đang bán?',C)
        low,_=dashboard_view(db,'Có bao nhiêu thuốc tồn thấp?',C)
    assert 'available_batches' in batches and 'today_sales' not in batches
    assert 'low_stock_count' in low and 'today_invoice_count' not in low

def test_catalog_lists_and_counts_same_as_ui(setup):
    client,Session,ids,tokens=setup
    with Session() as db:
        catalog,_=__import__('app.screen_tools',fromlist=['medicine_catalog']).medicine_catalog(
            db,'Danh sách thuốc đang kinh doanh trong nhóm Test',C)
    assert catalog['total']==1 and catalog['rows'][0]['code']=='M1'
    assert catalog['rows'][0]['category']=='Test'
    assert ai._fast_intent(AIChatRequest(message='Có bao nhiêu thuốc kê đơn đang kinh doanh?'), 'cashier')[0].calls[0].name=='medicine_catalog'
    assert ai._fast_intent(AIChatRequest(message='Những thuốc nào thuộc nhóm thuốc Test?'), 'cashier')[0].calls[0].name=='medicine_catalog'
    count_answer=[{'tool':'medicine_catalog','data':catalog}]
    assert presentation(count_answer,'cashier','Có bao nhiêu thuốc đang kinh doanh?')['table'] is None
    assert presentation(count_answer,'cashier','Danh sách thuốc đang kinh doanh')['table'] is not None


def test_specific_medicine_code_never_confused_with_generic_group_list():
    req=AIChatRequest(message='Thuốc AT030 thuộc nhóm thuốc nào?')
    assert ai._fast_intent(req,'manager')[0].calls[0].name=='medicine_search'
