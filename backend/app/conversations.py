"""Gemini streaming, cancellation, and persistent account-scoped chat history."""
import json
from datetime import datetime, timezone
from threading import Event, Lock
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update
from .auth import ai_user
from .db import get_db
from .models import AIConversation, AIMessage
from .schemas import AIChatRequest, ChatMessage
from .chat_history import owner_conversation
from . import ai

router = APIRouter(prefix='/api/ai/session')
_running = {}
_lock = Lock()

@router.post('/{request_id}/stop')
def stop(request_id: UUID, user=Depends(ai_user)):
    with _lock:
        entry = _running.get(str(request_id))
        if entry and entry[0] == user.id:
            entry[1].set()
    return {'status': 'stopping'}

@router.post('/{request_id}/stream')
def stream(request_id: UUID, data: AIChatRequest, user=Depends(ai_user), db=Depends(get_db)):
    from .main import rate_limit
    rate_limit('ai-session:' + str(user.id), 15, 60)
    key = str(request_id)
    event = Event()
    with _lock:
        if key in _running:
            raise HTTPException(409, 'Yêu cầu đang được xử lý.')
        _running[key] = (user.id, event)

    conversation_id = data.conversation_id
    assistant = None
    if conversation_id is not None:
        try:
            owner_conversation(db, conversation_id, user.id, user.role)
            # Atomically claim the conversation so two requests cannot append in parallel.
            claimed = db.execute(
                update(AIConversation)
                .where(AIConversation.id == conversation_id, AIConversation.user_id == user.id,
                       AIConversation.role_scope == user.role,
                       AIConversation.busy.is_(False))
                .values(busy=True)
            )
            if claimed.rowcount != 1:
                raise HTTPException(409, 'Cuộc trò chuyện đang được xử lý. Vui lòng thử lại.')
            recent = list(db.scalars(
                select(AIMessage).where(AIMessage.conversation_id == conversation_id,
                                       AIMessage.status == 'complete')
                .order_by(AIMessage.id.desc()).limit(12)
            ))
            data.history = [ChatMessage(role=m.role, content=m.content[:4000]) for m in reversed(recent)
                            if m.role in ('user', 'assistant') and m.content.strip()]
            conversation = owner_conversation(db, conversation_id, user.id, user.role)
            if conversation.title == 'Cuộc trò chuyện mới':
                conversation.title = data.message.strip()[:100]
            conversation.updated_at = datetime.now(timezone.utc)
            db.add(AIMessage(conversation_id=conversation_id, role='user', content=data.message, status='complete'))
            assistant = AIMessage(conversation_id=conversation_id, role='assistant', content='', status='interrupted')
            db.add(assistant)
            db.commit()
            db.refresh(assistant)
        except Exception:
            db.rollback()
            with _lock:
                _running.pop(key, None)
            raise

    data.agent = True
    data._cancelled = event.is_set

    def packets():
        gen = None
        content = []
        metadata = {'sources': [], 'tools': [], 'route': 'direct', 'model': '', 'presentation': None}
        status = 'interrupted'
        try:
            if assistant:
                yield json.dumps({'type':'message', 'id':assistant.id}) + '\n'
            gen = ai.chat_stream(db, data, user)
            for raw in gen:
                if event.is_set():
                    status = 'stopped'
                    yield json.dumps({'type':'done', 'status':'stopped'}) + '\n'
                    break
                try:
                    packet = json.loads(raw)
                except (TypeError, ValueError):
                    yield raw
                    continue
                if packet.get('type') == 'delta':
                    content.append(str(packet.get('text') or ''))
                elif packet.get('type') == 'meta':
                    metadata.update(sources=packet.get('sources') or [], tools=packet.get('used_tools') or [],
                                    route=packet.get('route') or 'direct', presentation=packet.get('presentation'))
                elif packet.get('type') == 'model':
                    metadata['model'] = packet.get('model') or ''
                elif packet.get('type') == 'done':
                    status = 'complete' if packet.get('status') != 'stopped' else 'stopped'
                    metadata['model'] = packet.get('model') or metadata['model']
                elif packet.get('type') == 'error':
                    status = 'interrupted'
                yield raw
        except GeneratorExit:
            raise
        except Exception:
            status = 'interrupted'
            yield json.dumps({'type':'error','text':'Phản hồi bị gián đoạn. Bạn có thể thử lại.'},ensure_ascii=False) + '\n'
        finally:
            if gen:
                gen.close()
            try:
                if assistant:
                    assistant.content = ''.join(content)[:20000]
                    assistant.status = status
                    assistant.metadata_json = json.dumps(metadata, ensure_ascii=False, default=str)
                    conversation = owner_conversation(db, conversation_id, user.id, user.role)
                    conversation.busy = False
                    conversation.updated_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception:
                db.rollback()
            finally:
                db.close()
                with _lock:
                    _running.pop(key, None)

    return StreamingResponse(packets(), media_type='application/x-ndjson', headers={
        'X-Accel-Buffering':'no', 'Cache-Control':'no-store'
    })
