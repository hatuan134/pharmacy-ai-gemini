"""Role-scoped views for the assistant, backed by the SAME calculations as UI.

Only read-only, authenticated data; no credentials, hash or token fields are ever
provided to a language model. Tool output is limited to what the question asks.
"""
import re
from datetime import date, timedelta
from .config import today
from .agent_time import window
from .ai import plain


def report_view(db, question, user):
    """Re-use actual /api/reports computations, but only for managers.

    The UI itself restricts this screen to managers. No accidental exposure via
    model planning, Fast Path, or crafted function calls is possible here.
    """
    if user.role != 'manager':
        raise ValueError('Chỉ Quản lý được phép xem báo cáo hoạt động.')
    from .main import report_payload, report_insights
    start, end = window(question)
    # The report UI offers 7/30/90-day presets; accept short natural labels too.
    q = plain(question)
    n = re.search(r'(?<!\d)(7|30|90)\s*ngay\b', q)
    if n and not re.search(r'tu\s+\d|thang |hom qua|tuan |quy |ngay toi', q):
        end = today()
        start = end - timedelta(days=int(n.group(1))-1)
    if end < start or (end - start).days > 365:
        raise ValueError('Khoảng thời gian báo cáo phải hợp lệ và không quá 366 ngày.')
    stock_value = bool(re.search(r'gia tri ton|von ton|tong gia tri|gia tri hang ton', q))
    top = bool(re.search(r'ban chay|ban nhieu nhat|top\s*\d+|thuoc ban duoc nhieu', q))
    trend = bool(re.search(r'xu huong|bieu do|do thi|tung ngay|theo ngay|bien dong doanh thu', q))
    inventory = bool(re.search(r'tinh hinh ton kho|ton co the ban|lo con han|lo sap het han|lo da het han|thuoc duoi nguong|thuoc dang kinh doanh', q))
    revenue = bool(re.search(r'doanh thu|doanh so|hoa don|bao cao ban hang', q))
    # A broad 'báo cáo hoạt động' request asks for the screen overview.
    broad = not any((stock_value, top, trend, inventory, revenue))
    payload = report_payload(db, start, end)
    answer = {'period': {'start': str(start), 'end': str(end)}}
    if revenue or broad:
        answer['revenue'] = str(payload['revenue'])
        answer['invoice_count'] = int(payload['invoice_count'])
    if stock_value or broad:
        answer['stock_value'] = str(payload['stock_value'])
        answer['stock_value_note'] = 'Giá nhập hiện tại của tất cả lô còn tồn, bao gồm lô hết hạn.'
    # Keep top sellers and inventory identical to the /reports screen, NOT an
    # independent SQL formula that might disagree with the official report.
    if top or inventory or broad:
        insight = report_insights(db, start, end, payload)
        if top or broad:
            answer['top_medicines'] = [
                {'name': r['name'], 'quantity': r['quantity'], 'unit': r['unit'], 'revenue': str(r['revenue'])}
                for r in insight['top_medicines']
            ]
            answer['top_medicines_scope'] = 'Tối đa 5 thuốc bán chạy nhất từ hóa đơn đã thanh toán trong kỳ.'
        if inventory or broad:
            answer['inventory'] = {k:(str(v) if isinstance(v, date) else v)
                                   for k,v in insight['inventory'].items()}
    if trend or broad:
        answer['daily_revenue'] = [
            {'date': r['date'], 'revenue': str(r['revenue'])} for r in payload['series'][:366]
        ]
        answer['daily_revenue_scope'] = 'Chỉ những ngày có hóa đơn thanh toán; các ngày không bán được 0 ₫.'
    return answer, [{'id':'report:period', 'title':'Báo cáo hoạt động',
                     'reference':f'{start} → {end}, cùng phép tính với màn hình Báo cáo', 'kind':'internal'}]


def dashboard_view(db, question, user):
    """Mirror each user's actual dashboard scope, especially cashier own shift."""
    from .main import dashboard
    data = dashboard(user=user, db=db)
    q = plain(question)
    if user.role == 'cashier':
        own = {'today_sales': str(data['today_sales']),
               'today_invoice_count': int(data['today_invoice_count']),
               'available_batches': int(data['available_batches']),
               'low_stock_count': int(data['low_stock_count']),
               'scope': 'Chỉ doanh số và hóa đơn của tài khoản Thu ngân đang đăng nhập; các chỉ số lô/tồn là phạm vi kho được phép xem.'}
        # One metric requested => do not expose or narrate unrelated dashboard KPIs.
        if re.search(r'lo (dang ban|con hang)|lo con han',q):
            own = {'available_batches':own['available_batches'], 'scope':own['scope']}
        elif re.search(r'thuoc ton thap|thuoc duoi nguong|ton kho thap',q):
            own = {'low_stock_count':own['low_stock_count'], 'scope':own['scope']}
        elif re.search(r'hoa don', q) and not re.search(r'doanh so|doanh thu|tien', q):
            own = {'today_invoice_count': own['today_invoice_count'], 'scope':own['scope']}
        elif re.search(r'doanh so|doanh thu|tien', q) and not re.search(r'hoa don',q):
            own = {'today_sales': own['today_sales'], 'scope':own['scope']}
        return own, [{'id':'dashboard:own', 'title':'Tổng quan Thu ngân',
                      'reference':'Cùng số liệu với màn hình Tổng quan', 'kind':'internal'}]
    if user.role == 'pharmacist':
        result = {k: data[k] for k in ('available_units','available_batches','expired_batches','expiring_batches')}
        result['low_stock_count'] = len(data['alerts']['low_stock'])
        return result, [{'id':'dashboard:pharmacist','title':'Bàn làm việc Dược sĩ','reference':'Tổng quan theo quyền Dược sĩ','kind':'internal'}]
    return {'revenue':str(data['revenue']), 'invoice_count':data['invoice_count'],
            'staff_count':data['staff_count'], 'medicine_count':data['medicine_count'],
            'alert_count':len(data['alerts']['expiry'])}, [
                {'id':'dashboard:manager','title':'Trung tâm điều hành','reference':'Tổng quan theo quyền Quản lý','kind':'internal'}]


def medicine_catalog(db, question, user):
    """Medicine list/count filters matching the fields on the Medicine screen.

    Available to all three roles (same as GET /medicines); output never includes
    the internal approval/audit fields or full medicine description in bulk.
    """
    from sqlalchemy import select, func
    from .models import Medicine, Category, Unit
    if user.role not in {'manager','pharmacist','cashier'}:
        raise ValueError('Không có quyền xem danh mục thuốc.')
    q=plain(question)
    stmt=(select(Medicine,Category.name,Unit.name)
          .join(Category,Category.id==Medicine.category_id)
          .join(Unit,Unit.id==Medicine.unit_id))
    if re.search(r'khong ke don|khong can don', q):
        stmt=stmt.where(Medicine.prescription_required.is_(False))
    elif re.search(r'ke don|can don', q):
        stmt=stmt.where(Medicine.prescription_required.is_(True))
    if re.search(r'ngung kinh doanh|khong kinh doanh|da ngung', q):
        stmt=stmt.where(Medicine.active.is_(False))
    elif re.search(r'dang kinh doanh|con kinh doanh|hoat dong',q):
        stmt=stmt.where(Medicine.active.is_(True))
    categories=list(db.scalars(select(Category)))
    chosen=[c for c in categories if len(plain(c.name))>=3 and plain(c.name) in q]
    if chosen:
        stmt=stmt.where(Medicine.category_id==max(chosen,key=lambda c:len(c.name)).id)
    total=int(db.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
    rows=db.execute(stmt.order_by(Medicine.name,Medicine.id).limit(51)).all()
    data=[{'code':m.code,'name':m.name,'category':category,'unit':unit,
           'min_stock':m.min_stock,'prescription_required':m.prescription_required,
           'active':m.active} for m,category,unit in rows[:50]]
    return {'total':total,'shown':len(data),'truncated':total>len(data),'rows':data,
            'scope':'Danh sách được lọc từ dữ liệu Thuốc của hệ thống; chỉ hiển thị tối đa 50 dòng.'}, [
        {'id':'medicine:catalog','title':'Danh mục thuốc', 'reference':'Cùng các trường hiển thị tại màn hình Thuốc','kind':'internal'}]
