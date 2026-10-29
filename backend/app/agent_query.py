"""Execute generated SELECTs on a disposable, bounded projection of business data.

Never execute model SQL on the production connection. SQLite authorizer is an
independent enforcement layer: only reads + five aggregate functions are legal.
"""
import re
import sqlite3
import time
from decimal import Decimal
from contextlib import contextmanager
from datetime import datetime, timezone
from sqlalchemy import select
from . import models as m

TABLES = {
 'medicines': (m.Medicine, 'id code name category_id unit_id min_stock prescription_required active information source'),
 'categories': (m.Category, 'id name'), 'units': (m.Unit, 'id name'),
 'suppliers': (m.Supplier, 'id name phone address active'),
 'batches': (m.Batch, 'id medicine_id supplier_id code received_date expiry_date quantity purchase_price sale_price'),
 'invoices': (m.Invoice, 'id user_id customer created_at total status payment_method prescription_ref cancel_reason'),
 'invoice_items': (m.InvoiceItem, 'id invoice_id batch_id medicine_name unit_name quantity sale_price purchase_price'),
 'users': (m.User, 'id username name role active'),
 'movements': (m.Movement, 'id batch_id user_id invoice_id delta balance kind reason created_at'),
}

ROLE_SCHEMA = {
    'manager': {name: fields.split() for name, (_, fields) in TABLES.items()},
    # Dược sĩ được phân tích động dữ liệu thuốc/kho/nhà cung cấp, nhưng không có
    # bảng hóa đơn để suy ra doanh thu tổng hợp.
    'pharmacist': {
        'medicines': TABLES['medicines'][1].split(),
        'categories': TABLES['categories'][1].split(),
        'units': TABLES['units'][1].split(),
        'suppliers': TABLES['suppliers'][1].split(),
        'batches': TABLES['batches'][1].split(),
        'movements': TABLES['movements'][1].split(),
    },
    # Thu ngân không được cấp dynamic_query; schema này vẫn thu hẹp để tạo lớp
    # bảo vệ thứ hai nếu hàm bị gọi trực tiếp trong tương lai.
    'cashier': {
        'medicines': TABLES['medicines'][1].split(),
        'categories': TABLES['categories'][1].split(),
        'units': TABLES['units'][1].split(),
        'batches': ['id', 'medicine_id', 'code', 'expiry_date', 'quantity', 'sale_price'],
    },
}

def schema(role):
    return {name: list(cols) for name, cols in ROLE_SCHEMA.get(role, {}).items()}

@contextmanager
def snapshot_reader(db):
    if db.get_bind().dialect.name == 'postgresql':
        with db.get_bind().connect().execution_options(isolation_level='REPEATABLE READ') as conn:
            with conn.begin():
                conn.exec_driver_sql('SET TRANSACTION READ ONLY')
                conn.exec_driver_sql("SET LOCAL statement_timeout = '3000ms'")
                yield conn
    else:
        yield db


def run_query(db, sql, role):
    if len(sql) > 6000 or not re.match(r'^\s*SELECT\b', sql, re.I):
        raise ValueError('Chỉ chấp nhận một câu SELECT.')
    if re.search(r';|--|/\*|\b(INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|ATTACH|DETACH|WITH|UNION|INTERSECT|EXCEPT|INTO)\b', sql, re.I):
        raise ValueError('Truy vấn chứa cú pháp không được phép.')
    con = sqlite3.connect(':memory:')
    con.row_factory = sqlite3.Row
    try:
        # A transaction snapshot from PostgreSQL; capped inputs are rejected, never
        # silently used to produce misleading aggregates.
        allowed = schema(role)
        if not allowed:
            raise ValueError('Vai trò không có quyền phân tích dữ liệu động.')
        with snapshot_reader(db) as reader:
            for name, cols in allowed.items():
                model = TABLES[name][0]
                rows = reader.execute(select(*(getattr(model, c) for c in cols)).limit(10001)).all()
                if len(rows) > 10000:
                    raise ValueError('Dữ liệu vượt giới hạn phân tích 10.000 dòng/bảng; hãy dùng báo cáo chuyên biệt.')
                types = ['NUMERIC' if c in {'quantity','purchase_price','sale_price','total','min_stock'} or c.endswith('_id') or c == 'id' else 'TEXT' for c in cols]
                con.execute('CREATE TABLE "'+name+'" (' + ','.join('"'+c+'" '+t for c,t in zip(cols,types))+')')
                def value(v):
                    if isinstance(v, Decimal): return str(v)
                    if isinstance(v, datetime): return v.astimezone(timezone.utc).isoformat() if v.tzinfo else v.replace(tzinfo=timezone.utc).isoformat()
                    if hasattr(v, 'isoformat'): return v.isoformat()
                    return v
                con.executemany('INSERT INTO "'+name+'" VALUES ('+','.join('?' for _ in cols)+')', [tuple(value(v) for v in row) for row in rows])
        con.commit()
        con.execute('PRAGMA query_only=ON')
        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT: return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 in allowed and (arg2 in allowed[arg1] or arg2 == ''): return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or '').lower() in {'sum','avg','count','min','max'}: return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        con.set_authorizer(authorize)
        deadline = time.monotonic() + 2
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        rows = con.execute(sql).fetchmany(201)
        return {'rows': [dict(r) for r in rows[:200]], 'truncated': len(rows)>200,
                'scope': 'Bản chiếu dữ liệu nội bộ tại thời điểm hỏi; tối đa 200 dòng kết quả.',
                'precision': 'Tiền lưu dạng số trên bản chiếu; tổng hợp có thể làm tròn ở mức hiển thị.'}
    except sqlite3.Error as exc:
        raise ValueError('Truy vấn bị từ chối, quá thời gian hoặc không đúng schema.') from exc
    finally:
        con.close()
