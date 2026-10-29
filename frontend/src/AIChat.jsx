import React,{useEffect,useRef,useState} from 'react';
import {Bot,Send,Square,Sparkles,History,Plus,Maximize2,Minimize2,X,ArrowUpRight,MessageSquare,Search,Trash2,Pencil,PanelLeftClose,ChevronRight,ChevronLeft,GripVertical,Table2,ExternalLink} from 'lucide-react';
import {api,send,streamApi} from './api';

const roleInfo={
  manager:{scope:'Hỗ trợ tra cứu thuốc, quản lý tồn kho, hóa đơn và phân tích hoạt động kinh doanh.',suggestions:['Doanh thu tháng này là bao nhiêu?','Những thuốc nào sắp hết hạn trong 90 ngày?','Thuốc nào đang tồn kho thấp?','Phân tích tình hình kinh doanh nhà thuốc']},
  pharmacist:{scope:'Hỗ trợ thông tin thuốc, lô nhập, tồn kho, hóa đơn và hướng dẫn nghiệp vụ.',suggestions:['Thuốc nào sắp hết hạn trong 90 ngày?','Tình hình tồn kho Paracetamol?','Hướng dẫn nhập lô thuốc mới','Tìm những thuốc có tồn kho thấp']},
  cashier:{scope:'Hỗ trợ tra cứu thuốc, giá bán, hạn dùng, hóa đơn và thao tác bán hàng.',suggestions:['Tra cứu thông tin Paracetamol','Thuốc nào tồn kho thấp?','Hướng dẫn quy trình bán thuốc tại quầy','Tra cứu hóa đơn gần đây']}
};
// Suggestions follow the current screen, then fall back to role-specific questions.
const pageQuestions={
  dashboard:['Nhà thuốc hôm nay có những cảnh báo gì?','Có những thuốc nào cần kiểm tra tồn kho?'],
  pos:['Làm sao để bán thuốc theo FEFO?','Tra cứu tồn kho thuốc đang bán','Tìm những hóa đơn mới nhất'],
  medicines:['Hệ thống có bao nhiêu loại thuốc?','Tìm thông tin thuốc theo tên hoặc mã'],
  batches:['Những lô nào sắp hết hạn trong 90 ngày?','Thuốc nào còn nhiều hàng nhưng bán chậm?'],
  alerts:['Những lô nào đã hết hạn?','Danh sách thuốc tồn kho thấp'],
  invoices:['Tìm hóa đơn gần đây','Hướng dẫn tra cứu và in lại hóa đơn'],
  reports:['Doanh thu tháng này là bao nhiêu?','Phân tích tình hình kinh doanh trong 30 ngày'],
  suppliers:['Nhà thuốc hiện có những nhà cung cấp nào?','Tra cứu nhà cung cấp thuốc'],
  settings:['Hướng dẫn quản lý danh mục thuốc','Các bước thay đổi mật khẩu']
};
const pageRouteNames={dashboard:'Tổng quan',pos:'Bán hàng',medicines:'Thuốc',batches:'Lô & tồn kho',alerts:'Cảnh báo',invoices:'Hóa đơn',reports:'Báo cáo',suppliers:'Nhà cung cấp',users:'Phân quyền'};
const allowedRoutes={
  manager:['dashboard','pos','medicines','batches','alerts','invoices','reports','suppliers','users'],
  pharmacist:['dashboard','pos','medicines','batches','alerts','invoices'],
  cashier:['dashboard','pos','medicines','invoices']
};
const fmtNum=n=>new Intl.NumberFormat('vi-VN',{maximumFractionDigits:0}).format(Number(n)||0);
function VerifiedPresentation({value,role,onNavigate}){
  if(!value||typeof value!=='object')return null;
  const cards=Array.isArray(value.cards)?value.cards.slice(0,3):[];
  const table=value.table;
  const actions=(Array.isArray(value.actions)?value.actions:[]).filter(a=>allowedRoutes[role]?.includes(a.page)&&pageRouteNames[a.page]);
  if(!cards.length&&!table&&!actions.length)return null;
  return <div className="verified-results">
    {cards.length>0&&<div className="verified-cards">{cards.map((c,i)=><div className="verified-card" key={i}><span>{c.label}</span><strong>{c.format==='currency'?fmtNum(c.value)+' ₫':fmtNum(c.value)}</strong></div>)}</div>}
    {table&&Array.isArray(table.columns)&&Array.isArray(table.rows)&&table.rows.length>0&&<details className="verified-table"><summary><Table2 size={16}/>{table.title||'Xem dữ liệu chi tiết'}<span>{table.rows.length} dòng</span></summary><div className="verified-table-scroll"><table><thead><tr>{table.columns.map(c=><th key={c.key}>{c.label}</th>)}</tr></thead><tbody>{table.rows.map((r,i)=><tr key={i}>{table.columns.map(c=><td key={c.key}>{String(r?.[c.key]??'—')}</td>)}</tr>)}</tbody></table></div>{table.has_more&&<p className="verified-more">Chỉ hiển thị một phần danh sách. Mở mục tương ứng để xem đầy đủ.</p>}</details>}
    {actions.length>0&&<nav className="verified-actions" aria-label="Đi đến chức năng liên quan">{actions.map(a=><button type="button" key={a.page} onClick={()=>onNavigate?.(a.page)}><ExternalLink size={15}/>{a.label}</button>)}</nav>}
  </div>;
}
const summaryDate=value=>{if(!value)return '';const d=new Date(value);return Number.isNaN(d.getTime())?'':d.toLocaleDateString('vi-VN',{day:'2-digit',month:'2-digit'});};
function cleanAssistantText(value){return String(value??'').replace(/^\s*\*\s+/gm,'- ').replace(/\*(?!\*)/g,'').replace(/^\s*Tình hình các lô\s*:?\s*$/gmi,'### Các lô hiện tại');}

export default function AIChat({user,MarkdownMessage,page,onNavigate}){
  const floating=true;
  const [opened,setOpened]=useState(false),[expanded,setExpanded]=useState(false),[historyOpen,setHistoryOpen]=useState(false);
  const [launcherCompact,setLauncherCompact]=useState(()=>{try{return window.localStorage.getItem('antam-ai-compact')==='true'}catch{return false}});
  const [launcherBottom,setLauncherBottom]=useState(()=>{try{const saved=Number(window.localStorage.getItem('antam-ai-bottom'));return Number.isFinite(saved)&&saved>=12?Math.min(saved,Math.max(12,window.innerHeight-80)):24}catch{return 24}});
  const drag=useRef(null);
  const [dragging,setDragging]=useState(false);
  const launcherBottomRef=useRef(launcherBottom);
  launcherBottomRef.current=launcherBottom;
  const [conversations,setConversations]=useState([]),[conversationId,setConversationId]=useState(null),[messages,setMessages]=useState([]);
  const [question,setQuestion]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState(''),[activity,setActivity]=useState('');
  const [filter,setFilter]=useState(''),[loading,setLoading]=useState(true);
  const controller=useRef(null),active=useRef(null),lock=useRef(false),end=useRef(null),messagesRef=useRef([]),idRef=useRef(null),mounted=useRef(true),loadToken=useRef(0);
  const suggestions=Array.from(new Set([...(pageQuestions[page]||[]),...(roleInfo[user.role]||roleInfo.cashier).suggestions])).slice(0,4);
  const scope=(roleInfo[user.role]||roleInfo.cashier).scope;
  function assignMessages(next){messagesRef.current=next;setMessages(next)}
  function mutateMessages(fn){setMessages(old=>{const next=fn(old);messagesRef.current=next;return next})}
  async function loadList(){const list=await api('/ai/conversations');setConversations(list);return list;}
  useEffect(()=>{
    mounted.current=true;
    const ticket=++loadToken.current;
    // Every login starts a blank chat. Saved conversations remain in History.
    idRef.current=null;setConversationId(null);assignMessages([]);setConversations([]);
    setQuestion('');setError('');setLoading(true);setHistoryOpen(false);
    (async()=>{try{const list=await api('/ai/conversations');if(mounted.current&&ticket===loadToken.current)setConversations(list)}catch(e){if(mounted.current&&ticket===loadToken.current)setError('Không tải được lịch sử: '+e.message)}finally{if(mounted.current&&ticket===loadToken.current)setLoading(false)}})();
    return()=>{mounted.current=false;++loadToken.current;controller.current?.abort();if(active.current)send(`/ai/session/${active.current}/stop`,'POST',{}).catch(()=>{})};
  },[user.id]);
  useEffect(()=>{if(opened||!floating)end.current?.scrollIntoView?.({block:'end'})},[messages,opened,activity,floating]);
  useEffect(()=>{try{window.localStorage.setItem('antam-ai-compact',String(launcherCompact))}catch{}},[launcherCompact]);
  useEffect(()=>{try{window.localStorage.setItem('antam-ai-bottom',String(Math.round(launcherBottom)))}catch{}},[launcherBottom]);
  useEffect(()=>{
    const clamp=()=>setLauncherBottom(b=>Math.min(Math.max(12,b),Math.max(12,window.innerHeight-75)));
    window.addEventListener('resize',clamp);
    return()=>window.removeEventListener('resize',clamp);
  },[]);
  useEffect(()=>{
    if(!opened)return;
    const key=e=>{if(e.key==='Escape'){if(expanded){setExpanded(false)}else{setOpened(false)}}};
    window.addEventListener('keydown',key);
    return()=>window.removeEventListener('keydown',key);
  },[opened,expanded]);
  function launch(){if(drag.current?.moved){drag.current=null;return;}drag.current=null;setOpened(true);setHistoryOpen(false)}
  function startDrag(e){if(e.button!==0||e.target.closest('.assistant-collapse-toggle'))return;
    drag.current={pointerId:e.pointerId,startY:e.clientY,startBottom:launcherBottomRef.current,moved:false};
    e.target.closest('.assistant-launcher')?.setPointerCapture?.(e.pointerId);
  }
  function moveDrag(e){const state=drag.current;if(!state||state.pointerId!==e.pointerId)return;
    if(Math.abs(e.clientY-state.startY)>5)state.moved=true;
    if(state.moved){setDragging(true);const next=state.startBottom+state.startY-e.clientY;setLauncherBottom(Math.max(12,Math.min(next,Math.max(12,window.innerHeight-75))));}
  }
  function stopDrag(e){setDragging(false);if(drag.current?.pointerId===e.pointerId){
    // Keep the movement flag until the click event has been ignored.
    if(!drag.current.moved||e.type==='pointercancel')drag.current=null;
    e.target.closest('.assistant-launcher')?.releasePointerCapture?.(e.pointerId);
  }}
  function keyboardDrag(e){if(e.key!=='ArrowUp'&&e.key!=='ArrowDown')return;
    e.preventDefault();setLauncherBottom(b=>Math.max(12,Math.min(b+(e.key==='ArrowUp'?24:-24),Math.max(12,window.innerHeight-75))));
  }
  async function openConversation(id){
    if(lock.current)return;
    const ticket=++loadToken.current;setError('');setLoading(true);
    try{const convo=await api(`/ai/conversations/${id}`);if(ticket!==loadToken.current)return;
      idRef.current=convo.id;setConversationId(convo.id);assignMessages(convo.messages||[]);setQuestion('');
      if(floating)setHistoryOpen(false);
    }catch(e){setError(e.message)}finally{if(ticket===loadToken.current)setLoading(false)}
  }
  async function fresh(){
    if(lock.current)return;
    ++loadToken.current;setError('');
    try{const convo=await send('/ai/conversations','POST',{});idRef.current=convo.id;setConversationId(convo.id);assignMessages([]);setQuestion('');setConversations(old=>[convo,...old]);if(floating)setHistoryOpen(false)}
    catch(e){setError(e.message)}
  }
  async function rename(convo){
    if(lock.current)return;
    const title=window.prompt('Tên cuộc trò chuyện:',convo.title);
    if(title===null||!title.trim()||title.trim()===convo.title)return;
    try{const updated=await send(`/ai/conversations/${convo.id}`,'PATCH',{title:title.trim()});setConversations(old=>old.map(x=>x.id===convo.id?updated:x))}catch(e){setError(e.message)}
  }
  async function remove(convo){
    if(lock.current)return;
    if(!window.confirm(`Xóa cuộc trò chuyện “${convo.title}”? Hành động này không thể hoàn tác.`))return;
    try{await api(`/ai/conversations/${convo.id}`,{method:'DELETE'});setConversations(old=>old.filter(x=>x.id!==convo.id));if(idRef.current===convo.id){idRef.current=null;setConversationId(null);assignMessages([]);setQuestion('')}}catch(e){setError(e.message)}
  }
  async function ask(e,text,base){
    e?.preventDefault();const q=(text??question).trim();if(!q||lock.current||loading)return;
    lock.current=true;setBusy(true);setError('');setActivity('Đang kết nối trợ lý…');
    const requestId=crypto.randomUUID();const signal=new AbortController();controller.current=signal;active.current=requestId;
    const previous=base??messagesRef.current;
    const temp=`pending-${requestId}`;
    try{
      let cid=idRef.current;
      if(!cid){const convo=await send('/ai/conversations','POST',{});cid=convo.id;idRef.current=cid;setConversationId(cid);setConversations(old=>[convo,...old]);}
      setQuestion('');if(floating)setHistoryOpen(false);
      assignMessages([...previous,{id:`client-${requestId}`,role:'user',content:q},{id:temp,role:'assistant',content:'',streaming:true}]);
      const update=fn=>mutateMessages(old=>old.map(m=>(m.id===temp||m.clientId===temp)?fn({...m,clientId:temp}):m));
      await streamApi(`/ai/session/${requestId}/stream`,{method:'POST',signal:signal.signal,body:JSON.stringify({message:q,conversation_id:cid,agent:true})},packet=>{
        if(!mounted.current)return;
        if(packet.type==='message')update(m=>({...m,id:packet.id}));
        if(packet.type==='activity')setActivity(packet.text);
        if(packet.type==='meta'){update(m=>({...m,presentation:packet.presentation||null}));setActivity('Đang tổng hợp kết quả…')}
        if(packet.type==='model')update(m=>({...m,model:packet.model}));
        if(packet.type==='delta'){setActivity('');update(m=>({...m,content:m.content+packet.text}))}
        if(packet.type==='error'){setError(packet.text);update(m=>({...m,status:'interrupted'}))}
        if(packet.type==='done')update(m=>({...m,streaming:false,status:packet.status==='stopped'?'stopped':packet.status==='interrupted'?'interrupted':'complete',model:packet.model||m.model}));
      });
      try{await loadList()}catch(e){setError('Không tải được danh sách lịch sử: '+e.message)}
    }catch(err){if(err.name!=='AbortError')setError(err.message);mutateMessages(old=>old.map(m=>(m.id===temp||m.clientId===temp)?{...m,streaming:false,status:m.status==='complete'?'complete':err.name==='AbortError'?'stopped':'interrupted'}:m))}
    finally{active.current=null;controller.current=null;lock.current=false;if(mounted.current){mutateMessages(old=>old.map(m=>(m.id===temp||m.clientId===temp)&&m.streaming?{...m,streaming:false,status:m.status||'interrupted'}:m));setBusy(false);setActivity('')}}
  }
  async function stop(){const id=active.current;if(id)try{await send(`/ai/session/${id}/stop`,'POST',{})}catch{}controller.current?.abort()}
  function goToRelatedPage(pageId){if(!allowedRoutes[user.role]?.includes(pageId))return;setOpened(false);setExpanded(false);setHistoryOpen(false);onNavigate?.(pageId)}
  const filtered=conversations.filter(c=>(c.title||'').toLocaleLowerCase('vi').includes(filter.toLocaleLowerCase('vi')));
  const visible=opened;
  return <div className={'antam-assistant assistant-floating'+(expanded&&opened?' expanded':'')}>
    {!opened&&<div className={'assistant-launcher-wrap'+(launcherCompact?' is-compact':'')+(dragging?' is-dragging':'')} style={{bottom:launcherBottom}} onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={stopDrag} onPointerCancel={stopDrag}>
      <button className="assistant-launcher" type="button" onClick={launch} onKeyDown={keyboardDrag} title="Mở An Tâm AI · kéo lên/xuống để đổi vị trí" aria-label="Mở An Tâm AI (dùng mũi tên lên/xuống để di chuyển)" aria-expanded={false}><img src="/an-tam-mark.png" alt=""/><span>An Tâm AI</span><GripVertical className="assistant-drag-icon" size={16} aria-hidden="true"/></button>
      <button className="assistant-collapse-toggle" type="button" onClick={()=>setLauncherCompact(x=>!x)} title={launcherCompact?'Mở rộng nút AI':'Thu gọn nút AI vào mép phải'} aria-label={launcherCompact?'Mở rộng nút AI':'Thu gọn nút AI vào mép phải'}>{launcherCompact?<ChevronLeft size={16}/>:<ChevronRight size={16}/>}</button>
    </div>}
    {visible&&<><div className="assistant-backdrop" aria-hidden="true" onClick={()=>{if(expanded)setExpanded(false)}}/><section className="assistant-shell" role={expanded?'dialog':'region'} aria-modal={expanded?'true':undefined} aria-label="Trợ lý An Tâm AI">
      <header className="assistant-head">
        <span className="assistant-identity"><img src="/an-tam-mark.png" alt=""/><span><b>An Tâm <em>AI</em></b><small><span className="online-dot"/> Trợ lý nhà thuốc · {user.role==='manager'?'Quản lý':user.role==='pharmacist'?'Dược sĩ':'Thu ngân'}</small></span></span>
        <div className="assistant-tools">
          <button type="button" title="Lịch sử trò chuyện" aria-label="Lịch sử trò chuyện" onClick={()=>setHistoryOpen(x=>!x)}><History size={19}/></button>
          <button type="button" title="Cuộc trò chuyện mới" aria-label="Cuộc trò chuyện mới" disabled={busy} onClick={fresh}><Plus size={20}/></button>
          <button type="button" title={expanded?'Thu nhỏ cửa sổ':'Phóng to cửa sổ'} aria-label={expanded?'Thu nhỏ cửa sổ':'Phóng to cửa sổ'} aria-pressed={expanded} onClick={()=>{setExpanded(x=>!x);setHistoryOpen(x=>!expanded?true:false)}}>{expanded?<Minimize2 size={18}/>:<Maximize2 size={18}/>}</button>
          <button type="button" title="Đóng chatbot" aria-label="Đóng chatbot" onClick={()=>{setExpanded(false);setOpened(false)}}><X size={19}/></button>
        </div>
      </header>
      <div className="assistant-content">
        <aside className={'assistant-history '+(historyOpen?'is-open':'')}>
          <div className="assistant-history-top"><b>Lịch sử trò chuyện</b><button aria-label="Ẩn lịch sử" title="Ẩn lịch sử" onClick={()=>setHistoryOpen(false)}><PanelLeftClose size={17}/></button></div>
          <button type="button" className="assistant-new" disabled={busy} onClick={fresh}><Plus size={18}/> Cuộc trò chuyện mới</button>
          <div className="assistant-find"><Search size={16}/><input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Tìm theo tiêu đề..." aria-label="Tìm lịch sử chat"/></div>
          <div className="assistant-history-list">{filtered.length?filtered.map(c=><div key={c.id} className={'assistant-history-item '+(conversationId===c.id?'selected':'')}>
            <button className="assistant-history-choice" onClick={()=>openConversation(c.id)} disabled={busy}><MessageSquare size={16}/><span><b>{c.title}</b><small>{summaryDate(c.updated_at)}</small></span></button>
            <button title="Đổi tên" aria-label={'Đổi tên '+c.title} onClick={()=>rename(c)} disabled={busy}><Pencil size={14}/></button>
            <button title="Xóa cuộc trò chuyện" aria-label={'Xóa '+c.title} onClick={()=>remove(c)} disabled={busy}><Trash2 size={14}/></button>
          </div>):<p className="assistant-no-history">{filter?'Không tìm thấy cuộc trò chuyện.':'Chưa có lịch sử. Hãy bắt đầu một câu hỏi mới.'}</p>}</div>
          <p className="assistant-private">Lịch sử chỉ hiển thị cho tài khoản của bạn.</p>
        </aside>
        <div className={'assistant-chat '+(historyOpen&&floating?'history-view':'')}>
          <div className="assistant-chat-body" aria-live="polite">
            {loading&&<p className="assistant-loading">Đang tải lịch sử trò chuyện…</p>}
            {!loading&&messages.length===0&&<div className="assistant-welcome"><img src="/an-tam-mark.png" alt=""/><span className="assistant-kicker">TRỢ LÝ NHÀ THUỐC AN TÂM</span><h2>Bạn muốn tìm hiểu điều gì hôm nay?</h2><p>{scope}</p><div className="assistant-prompts">{suggestions.map(s=><button key={s} disabled={busy} onClick={()=>ask(null,s)}>{s}<ArrowUpRight size={17}/></button>)}</div></div>}
            {messages.map((m,i)=><div key={m.clientId||m.id||i} className={'chat-row '+m.role}><div className="chat-bubble">
              {m.role==='assistant'&&<div className="bubble-top"><div className="bubble-name"><Sparkles size={16}/> An Tâm AI</div></div>}
              <div className="chat-text"><MarkdownMessage text={m.role==='assistant'?cleanAssistantText(m.content):m.content}/></div>
              {m.streaming&&!m.content&&<p className="agent-working">{activity||'Đang chuẩn bị phản hồi…'}</p>}
              {m.role==='assistant'&&!m.streaming&&<VerifiedPresentation value={m.presentation} role={user.role} onNavigate={goToRelatedPage}/>}
              {['stopped','interrupted'].includes(m.status)&&!m.streaming&&<small className="response-stopped">{m.status==='stopped'?'Đã dừng trả lời.':'Phản hồi chưa hoàn tất.'}</small>}
            </div></div>)}<div ref={end}/>
          </div>
          {error&&<div className="error assistant-error" role="alert">{error}</div>}
          <form className="chat-composer" onSubmit={ask}><div className="assistant-composer-shell"><label className="assistant-input-box"><span className="sr-only">Nội dung tin nhắn</span><textarea rows={1} maxLength={2000} placeholder="Nhắn tin cho An Tâm AI…" value={question} onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();ask(e)}}}/></label>{busy?<button type="button" className="stop-button" onClick={stop} aria-label="Dừng trả lời"><Square size={16}/>Dừng</button>:<button className="send-button" type="submit" disabled={loading||!question.trim()} aria-label="Gửi" title="Gửi tin nhắn"><Send size={20}/></button>}</div></form>
          <div className="chat-disclaimer">AI cung cấp thông tin tham khảo; không thay thế tư vấn của dược sĩ hoặc bác sĩ.</div>
        </div>
      </div>
    </section></>}
  </div>
}
