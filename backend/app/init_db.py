"""Create initial schema and first manager; never reset existing data."""
import os, getpass
from sqlalchemy import select, update
from .db import Base, engine, SessionLocal
from .models import User, AIConversation
from .auth import hash_password

def main():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        # Streaming jobs cannot survive a Render restart: clear stale busy flags.
        db.execute(update(AIConversation).where(AIConversation.busy.is_(True)).values(busy=False))
        db.commit()
        from .migrations import apply
        apply(db)
        if db.scalar(select(User.id).limit(1)):
            print('CSDL đã có tài khoản. Không thay đổi dữ liệu.')
            return
        password = os.getenv('ADMIN_PASSWORD') or getpass.getpass('Đặt mật khẩu quản lý (ít nhất 10 ký tự): ')
        if len(password) < 10: raise SystemExit('Mật khẩu quá ngắn.')
        db.add(User(username='admin', name='Quản lý nhà thuốc', role='manager', password_hash=hash_password(password)))
        db.commit()
        print('Đã tạo CSDL và tài khoản admin. Đăng nhập bằng mật khẩu vừa đặt.')

if __name__ == '__main__': main()
