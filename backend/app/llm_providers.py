"""Optional Groq fallback for An Tam's role-scoped AI tools.

The model never receives credentials or database connections. Only the caller's
already-authorized, capped tool results are supplied. No SDK is required.
"""
import json
import logging
from time import monotonic
import httpx
from .config import settings

log = logging.getLogger(__name__)
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'


def groq_models():
    if not (settings.groq_api_key or '').strip():
        return []
    return list(dict.fromkeys(m.strip() for m in settings.groq_models.split(',') if m.strip()))[:3]


def _message(system, prompt):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]


def _tool_schemas(allowed):
    descriptions = {
        'medicine_search':'Tra thông tin thuốc trong danh mục đã cấp quyền',
        'inventory_search':'Tra thuốc, tồn theo lô, hạn dùng, giá theo vai trò',
        'expiry_alerts':'Danh sách lô sắp hết hạn trong số ngày chỉ định',
        'low_stock':'Thuốc tồn kho dưới ngưỡng',
        'stock_risk':'Tốc độ bán và nguy cơ tồn kho theo thuốc',
        'sales_summary':'Tổng doanh thu đã thanh toán theo kỳ (chỉ Quản lý)',
        'system_data':'Thông tin danh mục, hóa đơn và nhà cung cấp theo quyền',
        'overview_counts':'Đếm chính xác số thuốc, lô, nhóm thuốc, đơn vị, nhà cung cấp (theo quyền)',
        'stock_balance':'Tổng tồn khả dụng chính xác của một thuốc hoặc mã thuốc',
        'alert_counts':'Đếm chính xác số lô hết/sắp hết hạn hoặc thuốc tồn thấp',
        'medicine_catalog':'Danh mục thuốc lọc theo nhóm, kê đơn, tình trạng kinh doanh và đếm chính xác (đúng quyền người dùng)',
        'report_view':'Báo cáo hoạt động chính thức theo kỳ: top 5 thuốc bán chạy, giá trị tồn kho, số lô, xu hướng doanh thu (chỉ Quản lý)',
        'dashboard_view':'Chỉ số Tổng quan theo đúng quyền và phạm vi tài khoản đăng nhập; thu ngân chỉ được hỏi doanh số ca của mình',
        'invoice_records':'Chi tiết một hóa đơn theo số/mã, khách hàng, thanh toán, người bán, từng mặt hàng; hoặc danh sách hóa đơn theo quyền.',
        'batch_records':'Chi tiết lô nhập: ngày nhập, hạn dùng, số lượng, giá nhập/bán và nhà cung cấp; chỉ Quản lý/Dược sĩ.',
        'movement_records':'Lịch sử biến động kho/kiểm kê/nhập lô/đổi giá, người thao tác và lý do; chỉ Quản lý/Dược sĩ.',
        'supplier_records':'Tên, điện thoại, địa chỉ và trạng thái nhà cung cấp; chỉ Quản lý/Dược sĩ.',
        'staff_records':'Danh sách nhân viên, vai trò và trạng thái hoạt động, không chứa mật khẩu; chỉ Quản lý.',
        'dynamic_query':'Truy vấn SQL SELECT trên schema được cung cấp và giới hạn theo vai trò',
    }
    return [{'type':'function','function':{'name':name,'description':desc,
             'parameters':{'type':'object','properties':{
                 'query':{'type':'string', 'description':('Câu SELECT chỉ đọc' if name == 'dynamic_query' else 'Tên, mã hoặc khoảng thời gian cần tra cứu')},
                 'days':{'type':'integer','description':'Số ngày cảnh báo, mặc định 90'}},
                 'required':['query'] if name=='dynamic_query' else []}}}
            for name,desc in descriptions.items() if name in set(allowed)]


def groq_json(system, payload, deadline=None):
    """Native tool selection when allowed_tools is present, bounded at backend.

    Model tool calls are *suggestions*: agent.Plan validates the names and
    execute_chat_tools / run_query re-check role and SQL safety.
    """
    if not groq_models():
        raise ValueError('groq_not_configured')
    tools = _tool_schemas(payload.get('allowed_tools') or [])
    for model in groq_models():
        remaining = (deadline - monotonic()) if deadline is not None else 12.0
        if remaining <= 0:
            break
        request_data = {
            'model': model, 'messages': _message(system + '\nNếu đã đủ kết quả, không gọi thêm công cụ và trả JSON {"calls":[]}.',
                json.dumps(payload, ensure_ascii=False, default=str)),
            'temperature': 0.1, 'max_completion_tokens': 1600,
        }
        if model.startswith(('openai/gpt-oss-', 'qwen/qwen3.8-')):
            request_data['reasoning_effort']='low'
        if tools:
            request_data.update({'tools': tools, 'tool_choice':'auto', 'parallel_tool_calls': False})
        else:
            request_data['response_format'] = {'type':'json_object'}
        try:
            response = httpx.post(GROQ_URL,
                headers={'Authorization': f'Bearer {settings.groq_api_key}'},
                json=request_data,
                timeout=httpx.Timeout(min(remaining, 12), connect=min(3, remaining)))
            if response.status_code in (401,403):
                break
            if response.status_code >= 400:
                log.warning('Groq planning HTTP %s on %s', response.status_code, model)
                continue
            message = response.json()['choices'][0]['message']
            requested = message.get('tool_calls') or []
            if requested:
                calls=[]
                for call in requested[:4]:
                    fn=call.get('function') or {}
                    name=fn.get('name')
                    if name not in (payload.get('allowed_tools') or []):
                        continue
                    try:
                        args=json.loads(fn.get('arguments') or '{}')
                    except (ValueError,TypeError):
                        continue
                    if isinstance(args,dict):
                        calls.append({'name':name,'query':str(args.get('query') or '')[:6000],
                                      'days':args.get('days') if isinstance(args.get('days'),int) else 90})
                if calls:
                    return {'calls':calls}
            content = message.get('content') or ''
            if isinstance(content,str) and content.strip():
                return json.loads(content)
        except (httpx.RequestError, ValueError, KeyError, IndexError, TypeError) as exc:
            log.warning('Groq planning failed on %s: %s', model, type(exc).__name__)
    raise ValueError('groq_planner_unavailable')


def groq_stream_events(system, prompt, cancelled=lambda: False):
    """Return ('model'|'delta'|'error', value). Never switch model after text."""
    models = groq_models()
    if not models:
        yield ('error', 'not_configured')
        return
    for model in models:
        if cancelled():
            return
        emitted = False
        try:
            with httpx.stream(
                'POST', GROQ_URL,
                headers={'Authorization': f'Bearer {settings.groq_api_key}'},
                json={'model': model, 'messages': _message(system, prompt),
                      'temperature': 0.2, 'max_completion_tokens': 1600,
                      'reasoning_effort': ('low' if 'ANSWER_TARGET: ONE_NUMBER_' in prompt else 'medium'),
                      'stream': True},
                timeout=httpx.Timeout(connect=3, read=settings.ai_stream_read_timeout_seconds,
                                      write=6, pool=3),
            ) as response:
                if response.status_code >= 400:
                    log.warning('Groq streaming HTTP %s on %s', response.status_code, model)
                    if response.status_code in (401, 403):
                        yield ('error', 'auth')
                        return
                    continue
                yield ('model', 'groq:' + model)
                for line in response.iter_lines():
                    if cancelled():
                        return
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='replace')
                    if not line.startswith('data:'):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == '[DONE]':
                        continue
                    try:
                        data = json.loads(raw)
                        delta = data.get('choices', [{}])[0].get('delta') or {}
                        chunk = delta.get('content') or ''
                    except (ValueError, IndexError, TypeError):
                        continue
                    if isinstance(chunk, str) and chunk:
                        emitted = True
                        yield ('delta', chunk)
                if emitted:
                    return
        except httpx.RequestError as exc:
            log.warning('Groq streaming network issue on %s: %s', model, type(exc).__name__)
            if emitted:
                yield ('error', 'stream_interrupted')
                return
        if emitted:
            return
    yield ('error', 'unavailable')


def groq_complete(system, prompt):
    """Non-streaming Groq synthesis for legacy HTTP callers."""
    if not groq_models():
        raise ValueError('groq_not_configured')
    for model in groq_models():
        try:
            response = httpx.post(GROQ_URL,
                headers={'Authorization': f'Bearer {settings.groq_api_key}'},
                json={'model':model,'messages':_message(system,prompt),
                      'temperature':0.2,'max_completion_tokens':1600},
                timeout=httpx.Timeout(15, connect=3))
            if response.status_code in (401,403):
                break
            if response.status_code >=400:
                continue
            content = response.json()['choices'][0]['message'].get('content') or ''
            if isinstance(content,str) and content.strip():
                return content.strip(), 'groq:' + model
        except (httpx.RequestError, ValueError, KeyError, IndexError, TypeError):
            continue
    raise ValueError('groq_unavailable')
