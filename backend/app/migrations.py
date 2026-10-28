"""Các migration dữ liệu nhỏ, chạy idempotent khi ứng dụng khởi động."""
import re
from sqlalchemy import select
from .models import SchemaMigration, Supplier


def _insert_for(db):
    if db.get_bind().dialect.name == 'postgresql':
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert


def _collapse_repeated_name(value: str) -> str:
    """Sửa lỗi nhập/paste tên bị lặp nguyên chuỗi hai lần, ví dụ ABCABC -> ABC."""
    name = (value or '').strip()
    if len(name) >= 4 and len(name) % 2 == 0:
        half = len(name) // 2
        if name[:half] == name[half:]:
            return name[:half].strip()
    return name


def apply(db):
    insert = _insert_for(db)

    # Dọn dữ liệu nhà cung cấp đã nhập sai trước khi bổ sung validation số điện thoại.
    # Chỉ sửa các trường hợp rõ ràng: phone có chữ; tên bị lặp đúng 2 lần; địa chỉ trùng tên.
    result = db.execute(
        insert(SchemaMigration)
        .values(name='v5_supplier_phone_validation_cleanup')
        .on_conflict_do_nothing()
    )
    if result.rowcount:
        suppliers = list(db.scalars(select(Supplier).order_by(Supplier.id)))
        occupied = {s.name.strip(): s.id for s in suppliers if s.name}
        for supplier in suppliers:
            old_name = (supplier.name or '').strip()
            new_name = _collapse_repeated_name(old_name)
            if new_name != old_name and (new_name not in occupied or occupied[new_name] == supplier.id):
                occupied.pop(old_name, None)
                supplier.name = new_name
                occupied[new_name] = supplier.id

            phone = (supplier.phone or '').strip()
            if phone and not re.fullmatch(r'\d{9,11}', phone):
                supplier.phone = ''

            address = (supplier.address or '').strip()
            normalized_name = (supplier.name or '').strip()
            if address and address in {old_name, normalized_name}:
                supplier.address = ''

    db.commit()
