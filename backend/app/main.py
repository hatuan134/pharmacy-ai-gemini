from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
from decimal import Decimal
from typing import Literal
from pathlib import Path
from urllib.parse import urlparse
import os
import json
import time as clock
from threading import Lock

from fastapi import FastAPI, Depends, HTTPException, Request, Response, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError

from .config import settings, today
from .db import get_db, Base, engine
from .models import *
from .schemas import *
from .auth import current_user, staff, ai_user, manager, roles, verify_password, hash_password, token
from .services import require, row, movement, batch_rows, alerts, create_sale, cancel_sale, invoice_detail, paid_sales_summary
from . import ai


def documented_header(x_requested_with: str = Header(default='pharmacy')):
    return x_requested_with


app = FastAPI(title='An Tâm · Quản lý nhà thuốc', version='2.0.0', dependencies=[Depends(documented_header)])
from .conversations import router as conversation_router
app.include_router(conversation_router)
from .chat_history import router as chat_history_router
app.include_router(chat_history_router)

origins = [x.strip() for x in settings.cors_origins.split(',')]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
    allow_headers=['Content-Type', 'X-Requested-With'],
)



def _same_origin(request: Request, origin: str) -> bool:
    try:
        parsed = urlparse(origin)
        return parsed.netloc.lower() == (request.headers.get('host') or '').lower()
    except ValueError:
        return False


@app.middleware('http')
async def csrf(request: Request, call_next):
    if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
        origin = request.headers.get('origin')
        origin_allowed = (not origin) or origin in origins or _same_origin(request, origin)
        if not origin_allowed or request.headers.get('x-requested-with') != 'pharmacy':
            return JSONResponse(status_code=403, content={'detail': 'Yêu cầu không hợp lệ. Vui lòng thao tác từ ứng dụng.'})
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.exception_handler(IntegrityError)
async def integrity_error(request, exc):
    return JSONResponse(status_code=409, content={'detail': 'Dữ liệu trùng hoặc đang được sử dụng. Kiểm tra mã, tên và các liên kết.'})


@app.get('/api/health')
def health(db=Depends(get_db)):
    db.execute(select(1))
    return {'status': 'ok', 'version': '2.0.0'}


_attempts = defaultdict(deque)
_attempt_lock = Lock()


def rate_limit(key, limit, seconds):
    with _attempt_lock:
        now = clock.monotonic()
        if len(_attempts) > 5000:
            for old in list(_attempts):
                if not _attempts[old] or _attempts[old][-1] < now - 3600:
                    del _attempts[old]
        queue = _attempts[key]
        while queue and queue[0] < now - seconds:
            queue.popleft()
        if len(queue) >= limit:
            raise HTTPException(429, 'Bạn thao tác quá nhanh. Vui lòng thử lại sau.')
        queue.append(now)


# ---------------- Authentication & staff ----------------
@app.post('/api/auth/login')
def login(data: Login, response: Response, request: Request, db=Depends(get_db)):
    rate_limit('login:' + (request.client.host if request.client else 'local'), 15, 300)
    user = db.scalar(select(User).where(User.username == data.username))
    dummy = '0' * 32 + ':' + '0' * 64
    valid = verify_password(data.password, user.password_hash if user else dummy)
    if not user or not valid or not user.active:
        raise HTTPException(401, 'Tên đăng nhập hoặc mật khẩu không đúng.')
    response.set_cookie('session', token(user), httponly=True, samesite='strict', secure=settings.cookie_secure, max_age=28800)
    db.commit()
    return row(user)


@app.get('/api/auth/me')
def me(user=Depends(current_user)):
    return row(user)


@app.post('/api/auth/logout')
def logout(response: Response, user=Depends(current_user), db=Depends(get_db)):
    user.token_version += 1
    db.commit()
    response.delete_cookie('session')
    return {'message': 'Đã đăng xuất.'}


@app.post('/api/auth/password')
def password(data: Password, response: Response, user=Depends(current_user), db=Depends(get_db)):
    if not verify_password(data.old_password, user.password_hash):
        raise HTTPException(400, 'Mật khẩu cũ không đúng.')
    user.password_hash = hash_password(data.new_password)
    user.token_version += 1
    db.commit()
    response.set_cookie('session', token(user), httponly=True, samesite='strict', secure=settings.cookie_secure, max_age=28800)
    return {'message': 'Đã đổi mật khẩu.'}


@app.get('/api/users')
def users(user=Depends(manager), db=Depends(get_db)):
    return [row(u) for u in db.scalars(select(User).order_by(User.id))]


@app.post('/api/users')
def user_create(data: UserIn, user=Depends(manager), db=Depends(get_db)):
    item = User(**data.model_dump(exclude={'password'}), password_hash=hash_password(data.password))
    db.add(item)
    db.flush()
    db.commit()
    return row(item)


@app.patch('/api/users/{ident}')
def user_state(ident: int, data: UserState, user=Depends(manager), db=Depends(get_db)):
    item = require(db, User, ident)
    if item.id == user.id and (data.active is False or data.role not in (None, 'manager')):
        raise HTTPException(409, 'Không thể tự khóa hoặc hạ quyền tài khoản đang đăng nhập.')
    if data.active is not None:
        item.active = data.active
    if data.role is not None:
        item.role = data.role
    item.token_version += 1
    db.commit()
    return row(item)


# ---------------- Catalog with real role separation ----------------
def catalog(path, model, schema, create_permission=staff, update_permission=staff, delete_permission=manager, read_permission=current_user):
    def listing(user=Depends(read_permission), db=Depends(get_db)):
        return [row(x) for x in db.scalars(select(model).order_by(model.id.desc()))]

    def validate(db, values, existing=None):
        if model is Medicine:
            require(db, Category, values['category_id'])
            require(db, Unit, values['unit_id'])
            if existing and existing.unit_id != values['unit_id'] and db.scalar(select(Batch.id).where(Batch.medicine_id == existing.id).limit(1)):
                raise HTTPException(409, 'Thuốc đã có lô: không thể đổi đơn vị tính. Hãy tạo mã thuốc mới.')

    def create(data: schema, user=Depends(create_permission), db=Depends(get_db)):
        values = data.model_dump()
        validate(db, values)
        obj = model(**values)
        db.add(obj)
        db.flush()
        db.commit()
        return row(obj)

    def update(ident: int, data: schema, user=Depends(update_permission), db=Depends(get_db)):
        obj = db.scalar(select(model).where(model.id == ident).with_for_update())
        if not obj: raise HTTPException(404, 'Không tìm thấy bản ghi.')
        before = row(obj)
        values = data.model_dump()
        validate(db, values, obj)
        for key, value in values.items():
            setattr(obj, key, value)
        db.commit()
        return row(obj)

    def delete(ident: int, user=Depends(delete_permission), db=Depends(get_db)):
        obj = require(db, model, ident)
        snapshot = row(obj)
        db.delete(obj)
        db.commit()
        return {'message': 'Đã xóa.'}

    app.add_api_route('/api/' + path, listing, methods=['GET'], name=path + '_list')
    app.add_api_route('/api/' + path, create, methods=['POST'], name=path + '_create')
    app.add_api_route('/api/' + path + '/{ident}', update, methods=['PUT'], name=path + '_update')
    app.add_api_route('/api/' + path + '/{ident}', delete, methods=['DELETE'], name=path + '_delete')


# Manager controls business catalogs/pricing; pharmacists curate medicine information.
catalog('categories', Category, NameIn, manager, manager, manager)
catalog('units', Unit, NameIn, manager, manager, manager)
catalog('suppliers', Supplier, SupplierIn, manager, manager, manager, staff)
catalog('medicines', Medicine, MedicineIn, staff, staff, manager)


# ---------------- Inventory / batches ----------------
@app.get('/api/batches')
def batches(q: str = '', category_id: int | None = None, expiry_before: date | None = None, available: bool = False, user=Depends(current_user), db=Depends(get_db)):
    rows = batch_rows(db, q, category_id, expiry_before, available)
    # Cashier does not need purchase price or supplier purchasing details.
    if user.role == 'cashier':
        for item in rows:
            item.pop('purchase_price', None)
            item.pop('supplier_id', None)
            item.pop('supplier_name', None)
    return rows


@app.post('/api/batches')
def batch_create(data: BatchIn, user=Depends(staff), db=Depends(get_db)):
    medicine = require(db, Medicine, data.medicine_id)
    supplier = require(db, Supplier, data.supplier_id)
    if not medicine.active or not supplier.active:
        raise HTTPException(422, 'Thuốc hoặc nhà cung cấp đã ngừng hoạt động.')
    if data.expiry_date <= today() or data.expiry_date <= data.received_date or data.received_date > today():
        raise HTTPException(422, 'Ngày nhập không được ở tương lai; hạn dùng phải sau ngày nhập và sau hôm nay.')
    if user.role == 'pharmacist':
        # Pharmacists enter lots, not price policies.  Reuse the most recently
        # configured sale price for the medicine; the first lot needs a manager.
        reference = db.scalar(select(Batch).where(Batch.medicine_id == data.medicine_id)
                              .order_by(Batch.id.desc()).limit(1).with_for_update())
        if reference is None:
            raise HTTPException(403, 'Thuốc chưa có giá bán do Quản lý thiết lập. Hãy nhờ Quản lý nhập lô đầu tiên.')
        if data.sale_price != reference.sale_price:
            raise HTTPException(403, 'Dược sĩ không được tự thay đổi giá bán. Hãy dùng giá của lô gần nhất hoặc nhờ Quản lý cập nhật.')
    batch = Batch(**data.model_dump())
    db.add(batch)
    db.flush()
    movement(db, batch, user, batch.quantity, 'receipt', 'Nhập lô mới')
    db.commit()
    return row(batch)


@app.patch('/api/batches/{ident}/price')
def batch_price(ident: int, data: BatchPrice, user=Depends(manager), db=Depends(get_db)):
    batch = db.scalar(select(Batch).where(Batch.id == ident).with_for_update())
    if not batch:
        raise HTTPException(404, 'Không tìm thấy lô.')
    old_price = batch.sale_price
    batch.sale_price = data.sale_price
    movement(db, batch, user, 0, 'price', f'Giá bán: {old_price} → {data.sale_price}')
    db.commit()
    return row(batch)


@app.post('/api/batches/{ident}/adjust')
def adjust(ident: int, data: Adjustment, user=Depends(manager), db=Depends(get_db)):
    batch = db.scalar(select(Batch).where(Batch.id == ident).with_for_update())
    if not batch:
        raise HTTPException(404, 'Không tìm thấy lô.')
    if batch.quantity != data.expected_quantity:
        raise HTTPException(409, 'Tồn kho đã thay đổi. Tải lại rồi kiểm kê lại.')
    before = batch.quantity
    delta = data.quantity - batch.quantity
    batch.quantity = data.quantity
    movement(db, batch, user, delta, 'adjustment', data.reason)
    db.commit()
    return row(batch)


@app.get('/api/movements')
def movements(user=Depends(staff), db=Depends(get_db)):
    return [row(x) for x in db.scalars(select(Movement).order_by(Movement.id.desc()).limit(1000))]


# ---------------- Alerts, sales, reports ----------------
@app.get('/api/alerts')
def alert_list(days: int = Query(90, ge=1, le=365), user=Depends(current_user), db=Depends(get_db)):
    return alerts(db, days)


def _public_invoice(detail, user):
    # Return only display fields, without backend idempotency fingerprints.
    result = {k:v for k,v in detail.items() if k not in ('request_key','request_hash')}
    if 'items' in result and user.role != 'manager':
        result['items'] = [{k:v for k,v in line.items() if k!='purchase_price'}
                           for line in result['items']]
    return result


@app.post('/api/invoices')
def sale(data: Sale, user=Depends(current_user), db=Depends(get_db)):
    return _public_invoice(create_sale(db, data, user), user)


@app.get('/api/invoices')
def invoices(user=Depends(current_user), db=Depends(get_db)):
    stmt = select(Invoice).order_by(Invoice.id.desc()).limit(1000)
    return [_public_invoice(row(i), user) for i in db.scalars(stmt)]


@app.get('/api/invoices/{ident}')
def invoice(ident: int, user=Depends(current_user), db=Depends(get_db)):
    item = require(db, Invoice, ident)
    return _public_invoice(invoice_detail(db, item), user)


@app.post('/api/invoices/{ident}/cancel')
def cancel(ident: int, data: Cancel, user=Depends(manager), db=Depends(get_db)):
    return cancel_sale(db, ident, data.reason, user)


def report_payload(db, start: date, end: date):
    sales = paid_sales_summary(db, start, end)
    zone = ZoneInfo('Asia/Ho_Chi_Minh')
    series = defaultdict(Decimal)
    for inv in sales['invoices']:
        timestamp = inv['created_at'].replace(tzinfo=timezone.utc) if inv['created_at'].tzinfo is None else inv['created_at']
        series[timestamp.astimezone(zone).date().isoformat()] += inv['total']
    stock = batch_rows(db)
    return {
        'start': start,
        'end': end,
        'revenue': sales['revenue'],
        'invoice_count': sales['invoice_count'],
        'stock_value': sum((b['quantity'] * b['purchase_price'] for b in stock), Decimal(0)),
        'total_units': sum(b['quantity'] for b in stock),
        'medicine_count': len(list(db.scalars(select(Medicine).where(Medicine.active.is_(True))))),
        'series': [{'date': k, 'revenue': v} for k, v in sorted(series.items())],
        'alerts': alerts(db),
    }



def report_insights(db, start: date, end: date, report: dict):
    """Manager-only report extras, calculated from paid invoices and live stock.

    Sold quantities/revenue are aggregated from historical invoice items, not
    from the mutable batch sale price; canceled invoices never contribute.
    """
    zone = ZoneInfo('Asia/Ho_Chi_Minh')
    lower = datetime.combine(start, time.min, zone).astimezone(timezone.utc)
    upper = datetime.combine(end + timedelta(days=1), time.min, zone).astimezone(timezone.utc)
    best = db.execute(
        select(
            Medicine.id,
            Medicine.name,
            Unit.name,
            func.sum(InvoiceItem.quantity).label('sold_quantity'),
            func.sum(InvoiceItem.quantity * InvoiceItem.sale_price).label('sold_revenue'),
        )
        .select_from(InvoiceItem)
        .join(Invoice, Invoice.id == InvoiceItem.invoice_id)
        .join(Batch, Batch.id == InvoiceItem.batch_id)
        .join(Medicine, Medicine.id == Batch.medicine_id)
        .join(Unit, Unit.id == Medicine.unit_id)
        .where(Invoice.status == 'paid', Invoice.created_at >= lower, Invoice.created_at < upper)
        .group_by(Medicine.id, Medicine.name, Unit.name)
        .order_by(func.sum(InvoiceItem.quantity).desc(), Medicine.name.asc())
        .limit(5)
    ).all()
    stock = batch_rows(db, available=True)
    warning = report['alerts']
    return {
        'top_medicines': [
            {'medicine_id': med_id, 'name': name, 'unit': unit,
             'quantity': int(qty), 'revenue': revenue}
            for med_id, name, unit, qty, revenue in best
        ],
        'inventory': {
            'as_of': today(),
            'medicine_count': report['medicine_count'],
            'available_batches': len(stock),
            'available_units': sum(b['quantity'] for b in stock),
            'expired_batches': sum(b['days_left'] <= 0 for b in warning['expiry']),
            'expiring_batches': sum(0 < b['days_left'] <= 90 for b in warning['expiry']),
            'low_stock_count': len(warning['low_stock']),
        },
    }


@app.get('/api/reports')
def reports(start: date | None = None, end: date | None = None, user=Depends(manager), db=Depends(get_db)):
    start_date, end_date = start or today().replace(day=1), end or today()
    report = report_payload(db, start_date, end_date)
    report.update(report_insights(db, start_date, end_date, report))
    return report


@app.get('/api/dashboard')
def dashboard(user=Depends(current_user), db=Depends(get_db)):
    warning = alerts(db)
    if user.role == 'manager':
        data = report_payload(db, today().replace(day=1), today())
        data.update({
            'role': 'manager',
            'staff_count': db.scalar(select(func.count(User.id)).where(User.active.is_(True))) or 0,
        })
        return data
    if user.role == 'pharmacist':
        available = batch_rows(db, available=True)
        return {
            'role': 'pharmacist',
            'alerts': warning,
            'available_units': sum(x['quantity'] for x in available),
            'available_batches': len(available),
            'expired_batches': sum(1 for x in warning['expiry'] if x['days_left'] <= 0),
            'expiring_batches': sum(1 for x in warning['expiry'] if 0 < x['days_left'] <= 90),
        }
    zone = ZoneInfo('Asia/Ho_Chi_Minh')
    lower = datetime.combine(today(), time.min, zone).astimezone(timezone.utc)
    upper = datetime.combine(today() + timedelta(days=1), time.min, zone).astimezone(timezone.utc)
    own = list(db.scalars(select(Invoice).where(Invoice.user_id == user.id, Invoice.status == 'paid', Invoice.created_at >= lower, Invoice.created_at < upper)))
    return {
        'role': 'cashier',
        'today_invoice_count': len(own),
        'today_sales': sum((x.total for x in own), Decimal(0)),
        'available_batches': len(batch_rows(db, available=True)),
        'low_stock_count': len(warning['low_stock']),
        'alerts': {'expiry': warning['expiry'][:5], 'low_stock': warning['low_stock'][:5]},
    }


# ---------------- AI ----------------
# Backward-compatible endpoint kept for old screens/tests.
@app.post('/api/ai/ask')
def ask(data: AIRequest, user=Depends(staff), db=Depends(get_db)):
    rate_limit('ai:' + str(user.id), 10, 60)
    return ai.answer(db, data, user)


@app.post('/api/ai/chat')
def chat(data: AIChatRequest, user=Depends(ai_user), db=Depends(get_db)):
    rate_limit('ai-chat:' + str(user.id), 15, 60)
    return ai.chat(db, data, user)


@app.post('/api/ai/chat/stream')
def chat_stream(data: AIChatRequest, user=Depends(ai_user), db=Depends(get_db)):
    rate_limit('ai-chat-stream:' + str(user.id), 15, 60)
    return StreamingResponse(
        ai.chat_stream(db, data, user),
        media_type='application/x-ndjson; charset=utf-8',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
        },
    )


# ---------------- Production frontend ----------------
# In local development Vite serves React on :5173 and proxies /api to FastAPI.
# In Render/Docker, FRONTEND_DIST points to the compiled Vite dist directory so
# the same FastAPI service serves both the UI and API from one HTTPS origin.
_frontend_env = os.getenv('FRONTEND_DIST', '').strip()
_frontend_dist = Path(_frontend_env).resolve() if _frontend_env else None
if _frontend_dist and (_frontend_dist / 'index.html').is_file():
    assets = _frontend_dist / 'assets'
    if assets.is_dir():
        app.mount('/assets', StaticFiles(directory=str(assets)), name='frontend-assets')

    @app.get('/{full_path:path}', include_in_schema=False)
    def frontend_spa(full_path: str):
        # API paths must never fall through to index.html.
        if full_path == 'api' or full_path.startswith('api/'):
            raise HTTPException(404, 'API endpoint không tồn tại.')
        candidate = (_frontend_dist / full_path).resolve()
        try:
            candidate.relative_to(_frontend_dist)
        except ValueError:
            candidate = _frontend_dist / 'index.html'
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_frontend_dist / 'index.html'))
