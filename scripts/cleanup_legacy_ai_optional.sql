-- TÙY CHỌN, CHỈ CHẠY SAU KHI CÓ BACKUP POSTGRESQL ĐÃ KIỂM TRA.
-- KHÔNG chạy bằng cách copy file patch và KHÔNG chạy trên DB production khi chưa xem lại.
-- Các bảng này không còn được ứng dụng giai đoạn 1 sử dụng.
-- Nếu cần giữ lịch sử phục vụ báo cáo/kiểm toán, KHÔNG drop.
BEGIN;
DROP TABLE IF EXISTS procedures;
DROP TABLE IF EXISTS ai_logs;
DROP TABLE IF EXISTS approval_requests;
DROP TABLE IF EXISTS audit_logs;
COMMIT;
-- Không xóa bảng inventory_movements: đó là lịch sử kho bắt buộc giữ lại.
-- Không xóa ai_conversations/ai_messages: dự kiến dùng lại ở giai đoạn 4.
-- KHÔNG DROP medicines.approved / medicines.approved_by ở GĐ1: ORM giữ để tương thích.
