"""Small, deterministic UI payloads derived exclusively from role-scoped tool results.

The language model cannot choose URLs, database fields or invent figures.
Keep the payload deliberately small; the full answer is generated independently.
"""
from decimal import Decimal, InvalidOperation

# Routes are client-side identifiers, never URLs supplied by the model.
ROLE_PAGES = {
    'manager': {'dashboard', 'pos', 'medicines', 'batches', 'invoices', 'alerts', 'suppliers', 'reports'},
    'pharmacist': {'dashboard', 'pos', 'medicines', 'batches', 'invoices', 'alerts'},
    'cashier': {'dashboard', 'pos', 'medicines', 'invoices'},
}

ROLE_TOOLS = {
    'manager': {'medicine_search','inventory_search','expiry_alerts','low_stock','stock_risk','sales_summary','system_data','dynamic_query'},
    'pharmacist': {'medicine_search','inventory_search','expiry_alerts','low_stock','stock_risk','system_data','dynamic_query'},
    'cashier': {'medicine_search','inventory_search','expiry_alerts','low_stock','system_data'},
}

TOOLS_TO_DESTINATIONS = {
    'medicine_search': ('medicines', 'Xem danh mục thuốc'),
    'inventory_search': ('batches', 'Xem lô và tồn kho'),
    'expiry_alerts': ('alerts', 'Xem cảnh báo hạn dùng'),
    'low_stock': ('alerts', 'Xem thuốc tồn kho thấp'),
    'stock_risk': ('batches', 'Xem chi tiết tồn kho'),
    'sales_summary': ('reports', 'Mở báo cáo doanh thu'),
}


def _num(value):
    try:
        result = Decimal(str(value))
        return float(result) if result.is_finite() else None
    except (ValueError, TypeError, InvalidOperation):
        return None


def _short(value, length=85):
    return str(value if value is not None else '—').replace('\n', ' ')[:length]


def _table(title, headers, data, limit=8):
    """Columns are field/label pairs checked at construction, not free-form model output."""
    return {
        'title': title,
        'columns': [{'key': k, 'label': label} for k, label in headers],
        'rows': [{k: _short(item.get(k)) for k, _ in headers} for item in data[:limit]],
        'shown': min(len(data), limit),
        'has_more': len(data) > limit,
    }


def _get_actions(results, role):
    allowed = ROLE_PAGES.get(role, set())
    actions = []
    for tool, data in results:
        if tool == 'system_data' and isinstance(data, dict):
            if 'invoices' in data or 'invoice_items' in data:
                route = ('invoices', 'Xem hóa đơn')
            elif 'suppliers' in data:
                route = ('suppliers', 'Xem nhà cung cấp')
            elif 'users' in data:
                route = ('dashboard', 'Về tổng quan')
            else:
                route = ('medicines', 'Xem danh mục thuốc')
        elif tool == 'dynamic_query':
            # Never use model-provided SQL/fields as a navigation destination.
            route = ('dashboard', 'Xem tổng quan')
        else:
            route = TOOLS_TO_DESTINATIONS.get(tool)
        if route:
            page, label = route
            if page not in allowed:
                page, label = ('medicines', 'Xem danh mục thuốc') if 'medicines' in allowed else ('dashboard', 'Về tổng quan')
            if all(a['page'] != page for a in actions):
                actions.append({'page': page, 'label': label})
    return actions[:3]


def presentation(results, role):
    """Return display-only data already filtered by each server-side tool's RBAC."""
    trusted = [(r.get('tool'), r.get('data')) for r in results if isinstance(r, dict) and not r.get('error') and r.get('tool') in ROLE_TOOLS.get(role, set())]
    # Keep chart=None for backward compatibility with persisted V5 chat metadata.
    ui = {'cards': [], 'table': None, 'chart': None, 'actions': _get_actions(trusted, role)}
    for tool, data in trusted:
        if tool == 'sales_summary' and role == 'manager' and isinstance(data, dict):
            revenue = _num(data.get('revenue'))
            invoices = data.get('invoice_count')
            if revenue is not None:
                ui['cards'].append({'label': 'Doanh thu đã thanh toán', 'value': revenue, 'format': 'currency'})
            if isinstance(invoices, int):
                ui['cards'].append({'label': 'Hóa đơn đã thanh toán', 'value': invoices, 'format': 'number'})
        elif tool == 'expiry_alerts' and isinstance(data, list):
            if ui['table'] is None and data:
                rows = [{**v, 'expiry_date': _short(v.get('expiry_date'), 10), 'quantity': f"{v.get('quantity', 0)} {v.get('unit', '')}"} for v in data if isinstance(v, dict)]
                ui['table'] = _table('Các lô cần chú ý', [('medicine_name', 'Thuốc'), ('code', 'Số lô'), ('expiry_date', 'Hạn dùng'), ('quantity', 'Tồn')], rows)
        elif tool == 'low_stock' and isinstance(data, list):
            if ui['table'] is None and data:
                ui['table'] = _table('Thuốc tồn kho thấp', [('name', 'Thuốc'), ('available_stock', 'Tồn khả dụng'), ('min_stock', 'Mức tối thiểu')], data)
        elif tool == 'inventory_search' and isinstance(data, list):
            if ui['table'] is None and data:
                ui['table'] = _table('Tồn kho theo lô', [('medicine', 'Thuốc'), ('batch', 'Số lô'), ('quantity', 'Tồn'), ('expiry_date', 'Hạn dùng')], data)
        elif tool == 'medicine_search' and isinstance(data, list):
            if ui['table'] is None and data:
                rows = [{**r, 'prescription_required': 'Kê đơn' if r.get('prescription_required') else 'Không kê đơn'} for r in data if isinstance(r, dict)]
                ui['table'] = _table('Danh sách thuốc tìm được', [('code', 'Mã'), ('name', 'Tên thuốc'), ('prescription_required', 'Loại')], rows)
        elif tool == 'stock_risk' and isinstance(data, list):
            if data and ui['table'] is None:
                ui['table'] = _table('Tồn kho và tốc độ bán', [('medicine', 'Thuốc'), ('stock', 'Tồn'), ('sold_30d', 'Đã bán 30 ngày'), ('nearest_expiry_days', 'Hạn gần nhất (ngày)')], data)
        elif tool == 'system_data' and isinstance(data, dict):
            if ui['table'] is None:
                groups = [('invoices', 'Hóa đơn', [('id', 'Mã'), ('customer', 'Khách hàng'), ('total', 'Tổng tiền'), ('status', 'Trạng thái')]),
                          ('suppliers', 'Nhà cung cấp', [('name', 'Tên'), ('phone', 'Điện thoại'), ('address', 'Địa chỉ')]),
                          ('categories', 'Danh mục', [('name', 'Tên nhóm')])]
                for key, title, headers in groups:
                    group = data.get(key)
                    if isinstance(group, dict) and isinstance(group.get('rows'), list) and group['rows']:
                        ui['table'] = _table(title, headers, group['rows'])
                        ui['table']['has_more'] = bool(group.get('truncated')) or ui['table']['has_more']
                        break
        elif tool == 'dynamic_query' and isinstance(data, dict):
            rows = data.get('rows')
            if ui['table'] is None and isinstance(rows, list) and rows and isinstance(rows[0], dict):
                # The read-only SQL projection has already denied forbidden tables/columns.
                headers = [(key, key.replace('_', ' ').capitalize()) for key in list(rows[0])[:6] if isinstance(key, str)]
                if headers:
                    ui['table'] = _table('Kết quả tra cứu', headers, rows)
                    ui['table']['has_more'] = bool(data.get('truncated')) or ui['table']['has_more']
    # Returning only whitelisted navigation/actions and capped verified data.
    return ui
