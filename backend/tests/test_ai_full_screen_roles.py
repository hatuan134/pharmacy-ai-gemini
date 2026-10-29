"""Role-scoped parity checks for the screens not covered by earlier AI tools."""
from types import SimpleNamespace
from uuid import uuid4
from decimal import Decimal
import pytest
from app import ai, direct_tools
from app.ai_presentation import presentation
from app.screen_detail import invoice_records, batch_records, movement_records, supplier_records, staff_records
from app.schemas import AIChatRequest

MANAGER=SimpleNamespace(id=1,role='manager')
PHARMACIST=SimpleNamespace(id=2,role='pharmacist')
CASHIER=SimpleNamespace(id=3,role='cashier')


def _plan(message, role='manager'):
    value=ai._fast_intent(AIChatRequest(message=message,agent=True),role)
    return [c.name for c in value[0].calls] if value else None


def test_role_tool_map_matches_visible_screens():
    for role in ('manager','pharmacist','cashier'):
        assert 'invoice_records' in ai._chat_allowed_tools(role)
        assert 'invoice_records' in direct_tools._allowed(role)
    for name in ('batch_records','movement_records','supplier_records'):
        assert name in ai._chat_allowed_tools('manager')
        assert name in ai._chat_allowed_tools('pharmacist')
        assert name not in ai._chat_allowed_tools('cashier')
    assert 'staff_records' in ai._chat_allowed_tools('manager')
    assert 'staff_records' not in ai._chat_allowed_tools('pharmacist')
    assert 'staff_records' not in ai._chat_allowed_tools('cashier')
    for name in ('invoice_records','batch_records','movement_records','supplier_records','staff_records'):
        assert name in direct_tools.KNOWN


def test_screen_fast_routes():
    expected={
        'Chi tiết hóa đơn 123':'invoice_records',
        'Hóa đơn số 1 gồm các thuốc gì?':'invoice_records',
        'Danh sách hóa đơn đã hủy':'invoice_records',
        'Lịch sử biến động kho':'movement_records',
        'Lịch sử kiểm kê lô B1':'movement_records',
        'Lịch sử đổi giá':'movement_records',
        'Ngày nhập lô thuốc là khi nào?':'batch_records',
        'Danh sách lô nhập':'batch_records',
        'Số điện thoại nhà cung cấp Test supplier':'supplier_records',
        'Danh sách nhân viên đang hoạt động':'staff_records',
    }
    for text,tool in expected.items():
        assert _plan(text)==[tool],(text,_plan(text))
    assert _plan('Chi tiết hóa đơn 123',role='cashier')==['invoice_records']
    assert _plan('Lịch sử kiểm kê',role='cashier') is None
    assert _plan('Thông tin thuốc kê đơn nào?',role='cashier') is not None


def test_exact_invoice_info_is_allowed_but_aggregate_denied():
    assert ai._role_access_denied_reason('Tổng tiền hóa đơn số 14?', 'cashier') == ''
    assert ai._role_access_denied_reason('Tổng tiền hóa đơn số 14?', 'pharmacist') == ''
    assert ai._role_access_denied_reason('Tổng tiền tất cả hóa đơn hôm nay?', 'cashier')
    assert ai._role_access_denied_reason('Doanh thu cả nhà thuốc?', 'pharmacist')


def test_invoice_detail_matches_api_and_hides_purchase_cost(setup):
    client,Session,ids,tokens=setup
    response=client.post('/api/invoices', json={
        'request_key':str(uuid4()),
        'items':[{'batch_id':ids['batch'],'quantity':2,'expected_sale_price':2000}],
    })
    assert response.status_code==200
    ident=response.json()['id']
    with Session() as db:
        m,_=invoice_records(db,f'Chi tiết hóa đơn số {ident}',MANAGER)
        p,_=invoice_records(db,f'Chi tiết hóa đơn số {ident}',PHARMACIST)
        c,_=invoice_records(db,f'Chi tiết hóa đơn số {ident}',CASHIER)
        assert m['records'][0]['id']==ident
        assert Decimal(m['records'][0]['total'])==Decimal(str(response.json()['total']))
        assert m['records'][0]['items'][0]['quantity']==2
        assert 'purchase_price' in m['records'][0]['items'][0]
        for data in (p,c):
            assert data['records'][0]['items'][0]['quantity']==2
            assert 'purchase_price' not in data['records'][0]['items'][0]
            assert 'request_hash' not in repr(data)
            assert 'request_key' not in repr(data)
        # The underlying receipt API must not be more permissive than AI.
        from app.main import invoice as get_invoice
        raw_cashier=get_invoice(ident, user=CASHIER, db=db)
        assert 'purchase_price' not in raw_cashier['items'][0]
        assert 'request_key' not in raw_cashier and 'request_hash' not in raw_cashier
        raw_manager=get_invoice(ident, user=MANAGER, db=db)
        assert 'purchase_price' in raw_manager['items'][0]
        from app.main import invoices as list_invoices
        assert all('request_key' not in row and 'request_hash' not in row for row in list_invoices(user=CASHIER,db=db))
        empty,_=invoice_records(db,'Chi tiết hóa đơn số 999999',CASHIER)
        assert empty['total']==0 and empty['records']==[]


def test_batch_and_movement_match_ui_and_reject_cashier(setup):
    client,Session,ids,_=setup
    client.post('/api/invoices', json={
        'request_key':str(uuid4()),
        'items':[{'batch_id':ids['batch'],'quantity':1,'expected_sale_price':2000}],
    })
    with Session() as db:
        batches,_=batch_records(db,'Chi tiết lô B1',PHARMACIST)
        assert batches['total']==1
        row=batches['records'][0]
        assert row['batch']=='B1' and row['quantity']==9
        assert 'received_date' in row and 'purchase_price' in row
        moves,_=movement_records(db,'Lịch sử kho lô B1',PHARMACIST)
        assert moves['total_in_latest_1000']>=1
        assert moves['records'][0]['batch']=='B1'
        assert all('performed_by' in m and 'reason' in m for m in moves['records'])
        with pytest.raises(ValueError,match='Quản lý và Dược sĩ'):
            batch_records(db,'Chi tiết lô B1',CASHIER)
        with pytest.raises(ValueError,match='không được'):
            movement_records(db,'Lịch sử biến động kho',CASHIER)


def test_suppliers_staff_do_not_expose_passwords(setup):
    client,Session,ids,_=setup
    with Session() as db:
        suppliers,_=supplier_records(db,'Số điện thoại nhà cung cấp',PHARMACIST)
        assert suppliers['total']==1
        assert suppliers['records'][0]['name']=='Test supplier'
        staff,_=staff_records(db,'Danh sách nhân viên',MANAGER)
        assert staff['total']==3
        for row in staff['records']:
            assert set(row)=={'name','username','role','active'}
        with pytest.raises(ValueError,match='Chỉ Quản lý'):
            staff_records(db,'Danh sách nhân viên',PHARMACIST)
        with pytest.raises(ValueError,match='không được'):
            supplier_records(db,'Nhà cung cấp',CASHIER)


def test_presentation_follows_scope_for_new_tools():
    data=[{'tool':'staff_records','data':{'records':[{'name':'test','role':'manager'}]}}]
    assert presentation(data,'manager','Danh sách nhân viên')['table']
    assert presentation(data,'pharmacist','Danh sách nhân viên')['table'] is None
    data=[{'tool':'invoice_records','data':{'records':[{'id':1,'total':'1000','customer':'Khách lẻ','status':'paid'}]}}]
    assert presentation(data,'cashier','Chi tiết hóa đơn số 1')['table']
    assert 'invoice_records' in ai._chat_allowed_tools('cashier')
    assert presentation(data,'cashier','Tổng tiền hóa đơn số 1 là bao nhiêu?')['table'] is None
    supplier=[{'tool':'supplier_records','data':{'records':[{'name':'A','phone':'0123456789'}]}}]
    assert presentation(supplier,'manager','Số điện thoại nhà cung cấp A?')['table'] is None


def test_role_enforced_before_any_function_tool(setup):
    _,Session,_,_=setup
    with Session() as db:
        req=AIChatRequest(message='Danh sách nhân viên đang hoạt động',agent=True)
        result=ai._prepare_chat(db,req,CASHIER)
        assert result['terminal'] and result['status']=='access_denied'
        req=AIChatRequest(message='Lịch sử biến động kho',agent=True)
        result=ai._prepare_chat(db,req,CASHIER)
        assert result['terminal'] and result['status']=='access_denied'
