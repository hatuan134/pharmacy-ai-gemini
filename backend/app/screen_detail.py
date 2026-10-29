"""Focused, read-only screen records for the authenticated AI user.

These tools mirror visible screen fields, never expose request hashes, passwords,
API tokens, internal approval flags or purchase cost to cashiers. Lists are
bounded; exact record lookups bypass the list cap.
"""
import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import select, func
from .ai import plain
from .models import Batch, Invoice, InvoiceItem, Medicine, Movement, Supplier, User
from .services import batch_rows
from .config import today


def _source(section, reference):
    return [{'id': 'screen:' + section, 'title': section,
             'reference': reference, 'kind': 'internal'}]


def _invoice_id(question):
    q = plain(question)
    match = re.search(r'(?:hoa don|don hang|invoice)\s*(?:so|ma|#)?\s*#?\s*(\d+)\b', q)
    return int(match.group(1)) if match else None


def invoice_records(db, question, user):
    """Match /invoices and /invoices/{id} without leaking internal request keys.

    Cashier and pharmacist can inspect invoices because both UI and API expose
    invoice lists and receipts. Only manager sees purchase costs.
    """
    if user.role not in {'manager','pharmacist','cashier'}:
        raise ValueError('Không có quyền xem hóa đơn.')
    ident = _invoice_id(question)
    query = select(Invoice)
    if ident is not None:
        query = query.where(Invoice.id == ident)
    else:
        q=plain(question)
        if re.search(r'hom nay|hom qua|tuan nay|tuan truoc|thang nay|thang truoc|quy nay|\d+\s*ngay qua|\d+\s*ngay gan day|tu\s+\d{1,2}/\d{1,2}', q):
            from .agent_time import window
            first,last=window(question)
            if (last-first).days>365:
                raise ValueError('Chỉ tra cứu hóa đơn trong khoảng tối đa 366 ngày.')
            zone=ZoneInfo('Asia/Ho_Chi_Minh')
            start=datetime.combine(first,time.min,zone).astimezone(timezone.utc)
            end=datetime.combine(last+timedelta(days=1),time.min,zone).astimezone(timezone.utc)
            query=query.where(Invoice.created_at>=start,Invoice.created_at<end)
        if re.search(r'da huy|bi huy|huy bo',q):
            query=query.where(Invoice.status=='cancelled')
        elif re.search(r'da thanh toan|thanh cong',q):
            query=query.where(Invoice.status=='paid')
    total=int(db.scalar(select(func.count()).select_from(query.subquery())) or 0)
    records=list(db.scalars(query.order_by(Invoice.id.desc()).limit(31)))
    records=records[:30]
    seller_ids={r.user_id for r in records}
    sellers={u.id:u.name for u in db.scalars(select(User).where(User.id.in_(seller_ids)))} if seller_ids else {}
    data=[]
    for inv in records:
        item={
            'id': inv.id, 'created_at': str(inv.created_at), 'customer':inv.customer,
            'seller':sellers.get(inv.user_id, '—'), 'total':str(inv.total),
            'status':inv.status,'payment_method':inv.payment_method,
            'prescription_ref':inv.prescription_ref,'cancel_reason':inv.cancel_reason,
        }
        if ident is not None:
            lines=list(db.scalars(select(InvoiceItem).where(InvoiceItem.invoice_id==inv.id).order_by(InvoiceItem.id)))
            item['items']=[{
                'medicine_name':i.medicine_name,'batch_id':i.batch_id,
                'unit_name':i.unit_name,'quantity':i.quantity,'sale_price':str(i.sale_price),
                **({'purchase_price':str(i.purchase_price)} if user.role=='manager' else {})
            } for i in lines]
        data.append(item)
    return {'total':total,'shown':len(data),'truncated':total>len(data),
            'records':data,'scope':'Hóa đơn trong phạm vi màn hình Hóa đơn; chỉ hóa đơn cụ thể mới kèm các dòng chi tiết.'}, _source('Hóa đơn', 'Màn hình Hóa đơn và chi tiết hóa đơn')


def batch_records(db, question, user):
    if user.role not in {'manager','pharmacist'}:
        raise ValueError('Chỉ Quản lý và Dược sĩ được xem màn hình Lô & tồn kho.')
    q=plain(question)
    rows=batch_rows(db, available=False)
    codes=[r for r in rows if re.search(r'(?<![\w-])'+re.escape(plain(r['code']))+r'(?![\w-])',q)]
    if codes:
        rows=codes
    else:
        meds=[r for r in rows if len(plain(r['medicine_name']))>=4 and plain(r['medicine_name']) in q]
        if meds:
            rows=meds
        if re.search(r'con han|con hang',q):
            rows=[r for r in rows if r['days_left']>0 and r['quantity']>0]
        if re.search(r'da het han|het han',q) and not re.search(r'sap het han',q):
            rows=[r for r in rows if r['days_left']<=0]
    total=len(rows)
    rows=rows[:30]
    data=[{'medicine_name':r['medicine_name'],'medicine_code':r['medicine_code'],
           'batch':r['code'],'received_date':str(r['received_date']),
           'expiry_date':str(r['expiry_date']),'days_left':r['days_left'],
           'quantity':r['quantity'],'unit':r['unit'],
           'sale_price':str(r['sale_price']), 'purchase_price':str(r['purchase_price']),
           'supplier':r['supplier_name'],'supplier_id':r['supplier_id'],
           'prescription_required':r['prescription_required']}
          for r in rows]
    return {'total':total,'shown':len(data),'truncated':total>len(data),
            'records':data,'scope':'Dữ liệu Lô & tồn kho; danh sách tối đa 30 dòng, truy vấn cụ thể theo số lô để xem đầy đủ.'},_source('Lô & tồn kho','Cùng dữ liệu với màn hình Lô & tồn kho')


def movement_records(db, question, user):
    """Inventory movements use the same rows as /movements (latest 1000).

    Filter by exact batch code if present. Never return arbitrary user credentials.
    """
    if user.role not in {'manager','pharmacist'}:
        raise ValueError('Vai trò hiện tại không được tra cứu lịch sử kho.')
    q=plain(question)
    # UI /movements lists latest 1,000 records, so never claim to cover all-time.
    moves=list(db.scalars(select(Movement).order_by(Movement.id.desc()).limit(1000)))
    batch_ids={r.batch_id for r in moves}
    batches={b.id:b for b in db.scalars(select(Batch).where(Batch.id.in_(batch_ids)))} if batch_ids else {}
    user_ids={r.user_id for r in moves}
    users={u.id:u.name for u in db.scalars(select(User).where(User.id.in_(user_ids)))} if user_ids else {}
    matched=[b for b in batches.values() if re.search(r'(?<![\w-])'+re.escape(plain(b.code))+r'(?![\w-])',q)]
    if matched:
        ids={b.id for b in matched}
        moves=[r for r in moves if r.batch_id in ids]
    kind=None
    for pattern,canonical in ((r'kiem ke|dieu chinh', 'adjustment'),(r'doi gia|thay doi gia','price'),
                              (r'nhap lo|nhap kho','receipt'),(r'huy don|huy hoa don','cancel'),
                              (r'ban hang|xuat kho','sale')):
        if re.search(pattern,q):
            kind=canonical
            break
    if kind:
        moves=[r for r in moves if r.kind==kind]
    total=len(moves)
    rows=[{'id':r.id,'created_at':str(r.created_at),'batch':batches[r.batch_id].code if r.batch_id in batches else str(r.batch_id),
           'performed_by':users.get(r.user_id,'—'),'invoice_id':r.invoice_id,
           'kind':r.kind,'delta':r.delta,'balance':r.balance,'reason':r.reason}
          for r in moves[:30]]
    return {'total_in_latest_1000':total,'shown':len(rows),'truncated':total>len(rows),
            'records':rows,'scope':'Lịch sử kho tối đa 1.000 biến động mới nhất như màn hình Lịch sử; không phải toàn bộ lịch sử từ khi tạo hệ thống.'}, _source('Lịch sử kho','Cùng phạm vi 1.000 bản ghi mới nhất với /api/movements')


def supplier_records(db, question, user):
    # Dược sĩ được xem danh sách NCC qua /api/suppliers khi nhập lô, dù không
    # có mục NCC độc lập trên sidebar. Thu ngân không có quyền này.
    if user.role not in {'manager','pharmacist'}:
        raise ValueError('Vai trò hiện tại không được tra cứu nhà cung cấp.')
    q=plain(question)
    suppliers=list(db.scalars(select(Supplier).order_by(Supplier.id.desc())))
    specific=[s for s in suppliers if len(plain(s.name))>=4 and plain(s.name) in q]
    if specific:suppliers=specific
    elif re.search(r'dang hop tac|dang hoat dong',q):suppliers=[s for s in suppliers if s.active]
    elif re.search(r'ngung hop tac|khong hoat dong',q):suppliers=[s for s in suppliers if not s.active]
    total=len(suppliers)
    return {'total':total,'shown':min(total,30),'truncated':total>30,
            'records':[{'id':s.id,'name':s.name,'phone':s.phone,'address':s.address,'active':s.active}
                       for s in suppliers[:30]]},_source('Nhà cung cấp','Cùng các trường của màn hình Nhà cung cấp')


def staff_records(db, question, user):
    if user.role!='manager':
        raise ValueError('Chỉ Quản lý được tra cứu Phân quyền.')
    q=plain(question)
    staff=list(db.scalars(select(User).order_by(User.id)))
    if re.search(r'dang hoat dong|chua khoa',q):staff=[u for u in staff if u.active]
    elif re.search(r'da khoa|khong hoat dong',q):staff=[u for u in staff if not u.active]
    if re.search(r'thu ngan',q):staff=[u for u in staff if u.role=='cashier']
    elif re.search(r'duoc si',q):staff=[u for u in staff if u.role=='pharmacist']
    total=len(staff)
    return {'total':total,'shown':min(total,30),'truncated':total>30,
            'records':[{'name':u.name,'username':u.username,'role':u.role,'active':u.active} for u in staff[:30]],
            'scope':'Không có mật khẩu, phiên đăng nhập, token hoặc trường bảo mật.'},_source('Phân quyền','Dữ liệu tài khoản không nhạy cảm trong màn hình Phân quyền')
