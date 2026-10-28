"""Persistent private AI conversations. Authorization always happens in SQL queries."""
import json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from .auth import ai_user
from .db import get_db
from .models import AIConversation, AIMessage

router = APIRouter(prefix='/api/ai/conversations', tags=['AI conversations'])

class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=100)

def owner_conversation(db, cid, user_id, role):
    conversation = db.scalar(select(AIConversation).where(
        AIConversation.id == cid, AIConversation.user_id == user_id,
        AIConversation.role_scope == role))
    if not conversation:
        raise HTTPException(404, 'Không tìm thấy cuộc trò chuyện.')
    return conversation

def summary(item):
    return {
        'id': item.id, 'title': item.title,
        'created_at': item.created_at.isoformat() if item.created_at else None,
        'updated_at': item.updated_at.isoformat() if item.updated_at else None,
    }

def message_dict(item):
    try:
        meta = json.loads(item.metadata_json or '{}')
        if not isinstance(meta, dict):
            meta = {}
    except (TypeError, ValueError):
        meta = {}
    return {
        'id': item.id, 'role': item.role, 'content': item.content,
        'status': item.status, 'feedback': item.feedback,
        'created_at': item.created_at.isoformat() if item.created_at else None,
        'sources': meta.get('sources', []), 'tools': meta.get('tools', []),
        'route': meta.get('route'), 'model': meta.get('model'),
        'presentation': meta.get('presentation'),
    }

@router.get('')
def list_conversations(user=Depends(ai_user), db=Depends(get_db)):
    items = db.scalars(
        select(AIConversation).where(AIConversation.user_id == user.id, AIConversation.role_scope == user.role)
        .order_by(AIConversation.updated_at.desc(), AIConversation.id.desc()).limit(100)
    ).all()
    return [summary(item) for item in items]

@router.post('', status_code=201)
def create_conversation(user=Depends(ai_user), db=Depends(get_db)):
    item = AIConversation(user_id=user.id, role_scope=user.role, title='Cuộc trò chuyện mới')
    db.add(item)
    db.commit()
    db.refresh(item)
    return summary(item)

@router.get('/{conversation_id}')
def get_conversation(conversation_id:int, user=Depends(ai_user), db=Depends(get_db)):
    item = owner_conversation(db, conversation_id, user.id, user.role)
    msgs = db.scalars(select(AIMessage).where(AIMessage.conversation_id == item.id).order_by(AIMessage.id.asc())).all()
    return {**summary(item), 'messages': [message_dict(msg) for msg in msgs]}

@router.patch('/{conversation_id}')
def rename_conversation(conversation_id:int, payload:ConversationUpdate, user=Depends(ai_user), db=Depends(get_db)):
    item = owner_conversation(db, conversation_id, user.id, user.role)
    item.title = payload.title.strip()
    if not item.title:
        raise HTTPException(422, 'Tên cuộc trò chuyện không được để trống.')
    item.updated_at = datetime.now(timezone.utc)
    db.commit()
    return summary(item)

@router.delete('/{conversation_id}')
def delete_conversation(conversation_id:int, user=Depends(ai_user), db=Depends(get_db)):
    item = owner_conversation(db, conversation_id, user.id, user.role)
    if item.busy:
        raise HTTPException(409, 'Không thể xóa khi AI đang trả lời.')
    db.execute(delete(AIMessage).where(AIMessage.conversation_id == item.id))
    db.delete(item)
    db.commit()
    return {'deleted': True}
