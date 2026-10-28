from sqlalchemy import select
import pytest

from app import ai, agent_query
from app.models import User
from app.schemas import AIChatRequest


def _user(db, role):
    return db.scalar(select(User).where(User.role == role))


def test_ai_tool_matrix_is_role_scoped():
    assert 'sales_summary' in ai._chat_allowed_tools('manager')
    assert 'sales_summary' not in ai._chat_allowed_tools('pharmacist')
    assert 'sales_summary' not in ai._chat_allowed_tools('cashier')
    assert 'stock_risk' in ai._chat_allowed_tools('pharmacist')
    assert 'stock_risk' not in ai._chat_allowed_tools('cashier')
    assert ai._dynamic_query_allowed('manager')
    assert ai._dynamic_query_allowed('pharmacist')
    assert not ai._dynamic_query_allowed('cashier')


def test_cashier_inventory_hides_purchase_price_and_supplier(setup):
    _, Session, _, _ = setup
    with Session() as db:
        cashier = _user(db, 'cashier')
        rows, _ = ai._tool_inventory_search(db, 'M1', cashier.role)
        assert rows
        assert 'sale_price' in rows[0]
        assert 'purchase_price' not in rows[0]
        assert 'supplier' not in rows[0]

        pharmacist = _user(db, 'pharmacist')
        rows, _ = ai._tool_inventory_search(db, 'M1', pharmacist.role)
        assert 'purchase_price' in rows[0]
        assert 'supplier' in rows[0]


def test_system_data_is_filtered_by_role(setup):
    _, Session, _, _ = setup
    with Session() as db:
        data, _ = ai._tool_system_data(db, 'nhà cung cấp', 'cashier')
        assert data == {}
        data, _ = ai._tool_system_data(db, 'nhà cung cấp', 'pharmacist')
        assert 'suppliers' in data

        data, _ = ai._tool_system_data(db, 'tài khoản và phân quyền', 'pharmacist')
        assert data == {}
        data, _ = ai._tool_system_data(db, 'tài khoản và phân quyền', 'manager')
        assert 'users' in data


def test_role_denials_are_returned_before_tools(setup):
    _, Session, _, _ = setup
    with Session() as db:
        cashier = _user(db, 'cashier')
        result = ai._prepare_chat(db, AIChatRequest(message='Giá nhập và nhà cung cấp của M1 là gì?', agent=True), cashier)
        assert result['terminal'] and result['status'] == 'access_denied'

        pharmacist = _user(db, 'pharmacist')
        result = ai._prepare_chat(db, AIChatRequest(message='Doanh thu tháng này là bao nhiêu?', agent=True), pharmacist)
        assert result['terminal'] and result['status'] == 'access_denied'

        manager = _user(db, 'manager')
        result = ai._role_access_denied_reason('Doanh thu tháng này là bao nhiêu?', manager.role)
        assert result == ''


def test_dynamic_query_schema_blocks_cross_role_data(setup):
    _, Session, _, _ = setup
    assert 'invoices' in agent_query.schema('manager')
    assert 'invoices' not in agent_query.schema('pharmacist')
    assert 'suppliers' in agent_query.schema('pharmacist')
    assert 'purchase_price' not in agent_query.schema('cashier')['batches']
    assert 'supplier_id' not in agent_query.schema('cashier')['batches']

    with Session() as db:
        with pytest.raises(ValueError):
            agent_query.run_query(db, 'SELECT SUM(total) FROM invoices', 'pharmacist')
        with pytest.raises(ValueError):
            agent_query.run_query(db, 'SELECT purchase_price FROM batches', 'cashier')

def test_all_roles_can_list_invoices_via_ai(setup):
    _, Session, _, _ = setup
    with Session() as db:
        for role in ('cashier', 'pharmacist'):
            assert ai._role_access_denied_reason('Cho tôi danh sách hóa đơn', role) == ''
            assert ai._role_access_denied_reason('Hóa đơn của nhân viên khác', role) == ''
            data, _ = ai._tool_system_data(db, 'hóa đơn', role)
            assert 'invoices' in data
            assert 'invoice_items' in data
            assert 'purchase_price' not in data['invoice_items']['rows'][0] if data['invoice_items']['rows'] else True
            assert ai._role_access_denied_reason('Doanh thu tổng nhà thuốc', role)
