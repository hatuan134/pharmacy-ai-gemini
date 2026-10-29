// @vitest-environment jsdom
import React from 'react';
import {afterEach,beforeEach,describe,it,expect,vi} from 'vitest';
import {render,screen,waitFor,cleanup,fireEvent,within} from '@testing-library/react';
import {Login,Dashboard,Catalog,POS,Batches,Invoices,Alerts,Reports,AI,App,Staff,SettingsPage} from './main';
import {api,send,streamApi,localToday,money,date} from './api';
vi.mock('./api',async()=>{const real=await vi.importActual('./api');return{...real,api:vi.fn(),send:vi.fn(),streamApi:vi.fn()}});
const user={id:1,name:'Quản lý kiểm thử',role:'manager'};
const batch={id:1,medicine_id:1,medicine_name:'Thuốc kiểm thử A',medicine_code:'T001',code:'LOT-001',quantity:10,unit:'Hộp',days_left:30,expiry_date:'2027-01-01',sale_price:'20000',purchase_price:'10000',supplier_name:'NCC',prescription_required:false};
const medicine={id:1,name:'Thuốc kiểm thử A',code:'T001',approved:true,active:true,min_stock:5,category_id:1,unit_id:1,information:'Thông tin',source:'Nguồn kiểm thử'};
const report={role:'manager',revenue:'0',invoice_count:0,medicine_count:1,stock_value:'100000',series:[],staff_count:3,pending_approvals:0,alerts:{expiry:[batch],low_stock:[]}};
const data={'/medicines':[medicine],'/categories':[{id:1,name:'Nhóm 1'}],'/units':[{id:1,name:'Hộp'}],'/suppliers':[{id:1,name:'NCC'}],'/batches':[batch],'/invoices':[], '/procedures':[], '/users':[user], '/reports':report, '/dashboard':report, '/alerts':report.alerts, '/approval-requests':[], '/audit-logs':[]};
beforeEach(()=>{window.localStorage.clear();api.mockReset();send.mockReset();send.mockResolvedValue({id:1,title:'Cuộc trò chuyện mới'});streamApi.mockReset();api.mockImplementation(path=>Promise.resolve(data[path.split('?')[0]]??[]));});
afterEach(cleanup);
function renderChat(){render(<AI user={user}/>);fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));}
it('dashboard renders loaded values and navigation action',async()=>{const go=vi.fn();render(<Dashboard user={user} go={go}/>);await screen.findByText('Doanh thu tháng');fireEvent.click(screen.getByRole('button',{name:'Tạo đơn bán'}));expect(go).toHaveBeenCalledWith('pos')});
it('medicine page loads, opens edit and sends only writable fields',async()=>{render(<Catalog type="medicines" user={user}/>);await screen.findByText('Thuốc kiểm thử A');fireEvent.click(screen.getByRole('button',{name:'Sửa'}));fireEvent.change(screen.getByLabelText('Tên thuốc'),{target:{value:'Tên mới'}});send.mockResolvedValue({});fireEvent.click(screen.getByRole('button',{name:'Lưu thông tin'}));await waitFor(()=>expect(send).toHaveBeenCalled());expect(send.mock.calls[0][2].name).toBe('Tên mới');expect(send.mock.calls[0][2]).not.toHaveProperty('approved');expect(send.mock.calls[0][2]).not.toHaveProperty('id')});
it('POS adds a lot and submits captured price',async()=>{render(<POS user={user}/>);const product=await screen.findByRole('button',{name:/Thuốc kiểm thử A/});fireEvent.click(product);send.mockResolvedValue({id:1,customer:'Khách lẻ',total:'20000',items:[],status:'paid'});fireEvent.click(screen.getByRole('button',{name:'Xác nhận thanh toán'}));await screen.findByRole('dialog');expect(send.mock.calls[0][2].items).toEqual([{batch_id:1,quantity:1,expected_sale_price:'20000'}]);});
it('POS keeps request key on ambiguous network retry',async()=>{render(<POS user={user}/>);fireEvent.click(await screen.findByRole('button',{name:/Thuốc kiểm thử A/}));send.mockRejectedValue(new Error('Mất kết nối'));fireEvent.click(screen.getByRole('button',{name:'Xác nhận thanh toán'}));fireEvent.click(await screen.findByRole('button',{name:'Thử lại thanh toán'}));await waitFor(()=>expect(send).toHaveBeenCalledTimes(2));expect(send.mock.calls[0][2].request_key).toBe(send.mock.calls[1][2].request_key)});
it('cashier cannot select prescription product',async()=>{api.mockResolvedValue([{...batch,prescription_required:true}]);render(<POS user={{...user,role:'cashier'}}/>);const product=await screen.findByRole('button',{name:/Thuốc kiểm thử A/});expect(product.disabled).toBe(true)});
it('batch page renders quantity and opens adjustment',async()=>{render(<Batches user={user}/>);await screen.findByText('LOT-001');fireEvent.click(screen.getByRole('button',{name:'Kiểm kê'}));expect(screen.getByLabelText('Số lượng thực tế').value).toBe('10')});
it('invoice page has usable empty state',async()=>{render(<Invoices user={user}/>);await screen.findByText('Không có dữ liệu')});
it('alerts show expiring batch',async()=>{render(<Alerts/>);await screen.findByText('LOT-001');expect(screen.getByText('Còn 30 ngày')).toBeTruthy()});
it('reports handle no revenue',async()=>{render(<Reports/>);await screen.findByText('Chưa có doanh thu trong khoảng thời gian này');expect(screen.queryByRole('img',{name:/Biểu đồ đường/})).toBeNull()});
it('AI chatbot shows provider error and stays usable',async()=>{renderChat();const box=screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…');fireEvent.change(box,{target:{value:'Kiểm kê như thế nào?'}});streamApi.mockRejectedValue(new Error('Chưa cấu hình GEMINI_API_KEY'));await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));fireEvent.click(screen.getByRole('button',{name:'Gửi'}));await screen.findByRole('alert');expect(screen.getByText('Chưa cấu hình GEMINI_API_KEY')).toBeTruthy();expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(true)});
it('AI chatbot renders answer as text, not injected HTML',async()=>{renderChat();const box=screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…');fireEvent.change(box,{target:{value:'Kiểm kê?'}});send.mockResolvedValue({id:1,title:'Cuộc trò chuyện mới'});streamApi.mockImplementation(async(path,options,event)=>{event({type:'message',id:2});event({type:'delta',text:'<script>alert(1)</script>'});event({type:'done',status:'ok'})});await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));fireEvent.click(screen.getByRole('button',{name:'Gửi'}));await screen.findByText('<script>alert(1)</script>');expect(document.querySelector('script')).toBeNull()});
it('staff list and settings render',async()=>{const view=render(<Staff/>);await screen.findByText('Quản lý kiểm thử');view.unmount();render(<SettingsPage user={user}/>);expect(screen.getByLabelText('Mật khẩu hiện tại')).toBeTruthy()});
it('failed data loading offers retry instead of a blank page',async()=>{api.mockRejectedValueOnce(new Error('Mất kết nối máy chủ'));render(<Catalog type="suppliers" user={user}/>);await screen.findByRole('alert');fireEvent.click(screen.getByRole('button',{name:'Thử lại'}));await screen.findByText('NCC')});

it('AI stops active generation using AbortSignal',async()=>{
  send.mockResolvedValue({id:1,title:'Cuộc trò chuyện mới'});
  streamApi.mockImplementation((path,options,event)=>new Promise((resolve,reject)=>{event({type:'message',id:2});options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')))}));
  renderChat();
  fireEvent.change(screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…'),{target:{value:'Tồn kho?'}});
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));fireEvent.click(screen.getByRole('button',{name:'Gửi'}));
  await waitFor(()=>expect(streamApi).toHaveBeenCalled());
  fireEvent.click(screen.getByRole('button',{name:'Dừng trả lời'}));
  await screen.findByText('Đã dừng trả lời.');
  expect(send).toHaveBeenCalledWith(expect.stringMatching(/^\/ai\/session\/.+\/stop$/),'POST',{});
});
it('AI renders a Markdown table safely',async()=>{
  send.mockResolvedValue({id:1,title:'Cuộc trò chuyện mới'});
  streamApi.mockImplementation(async(path,options,event)=>{event({type:'message',id:2});event({type:'delta',text:'| Thuốc | Tồn |\n| --- | --- |\n| **Para** | 150 |'});event({type:'done',status:'ok'})});
  renderChat();fireEvent.change(screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…'),{target:{value:'Tồn?'}});await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));fireEvent.click(screen.getByRole('button',{name:'Gửi'}));
  await screen.findByRole('table');expect(screen.getByRole('cell',{name:'150'})).toBeTruthy();
});


it('AI header omits chat persistence note and hides stray asterisks',async()=>{
  streamApi.mockImplementation(async(path,options,event)=>{
    event({type:'message',id:2});
    event({type:'delta',text:'Quy trình\n* Bước một\nNội dung bị dư dấu*'});
    event({type:'done',status:'ok'});
  });
  renderChat();
  expect(screen.getByText(/Trợ lý nhà thuốc/)).toBeTruthy();
  expect(screen.queryByText(/Nội dung chat không được lưu lại/)).toBeNull();
  fireEvent.change(screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…'),{target:{value:'Quy trình?'}});
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));fireEvent.click(screen.getByRole('button',{name:'Gửi'}));
  await screen.findByText('Bước một');
  expect(screen.getByText('Nội dung bị dư dấu').textContent).not.toContain('*');
});

it('password eyes toggle independently and never submit',()=>{
  render(<SettingsPage user={user}/>);
  const old=screen.getByLabelText('Mật khẩu hiện tại'),next=screen.getByLabelText('Mật khẩu mới');
  expect(old.type).toBe('password');
  fireEvent.click(screen.getByRole('button',{name:'Hiện mật khẩu hiện tại'}));
  expect(old.type).toBe('text');expect(next.type).toBe('password');expect(send).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Ẩn mật khẩu hiện tại'}));expect(old.type).toBe('password');
});
it('login has password eye',()=>{render(<Login onLogin={vi.fn()}/>);fireEvent.click(screen.getByRole('button',{name:'Hiện mật khẩu'}));expect(screen.getByLabelText('Mật khẩu').type).toBe('text')});
it('medicine editor has no manual AI approval action',async()=>{
  render(<Catalog type="medicines" user={user}/>);
  await screen.findByText('Thuốc kiểm thử A');
  expect(screen.queryByTitle('Kiểm tra nguồn AI')).toBeNull();
  expect(screen.queryByText('Chưa duyệt')).toBeNull();
});
it('pharmacist can view batches but cannot adjust stock or sale price',async()=>{
  render(<Batches user={{...user,role:'pharmacist'}}/>);
  await screen.findByText('LOT-001');
  expect(screen.queryByRole('button',{name:'Kiểm kê'})).toBeNull();
  expect(screen.queryByRole('button',{name:'Đổi giá'})).toBeNull();
});
it('chat shows persistent conversation history and creates new sessions',async()=>{
  api.mockImplementation(path=>Promise.resolve(path==='/ai/conversations'?[{id:3,title:'Tra cứu tồn kho',updated_at:'2026-10-28T11:00:00Z'}]:path==='/ai/conversations/3'?{id:3,messages:[{id:21,role:'user',content:'Tồn bao nhiêu?'},{id:22,role:'assistant',content:'Còn 10 hộp',status:'complete'}]}:data[path.split('?')[0]]??[]));
  send.mockResolvedValue({id:4,title:'Cuộc trò chuyện mới'});
  renderChat();
  fireEvent.click(screen.getByRole('button',{name:'Lịch sử trò chuyện'}));
  await screen.findByText('Tra cứu tồn kho');
  expect(screen.queryByText('Còn 10 hộp')).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:/Tra cứu tồn kho/}));
  expect(await screen.findByText('Còn 10 hộp')).toBeTruthy();
  fireEvent.click(screen.getAllByRole('button',{name:'Cuộc trò chuyện mới'})[0]);
  await waitFor(()=>expect(send).toHaveBeenCalledWith('/ai/conversations','POST',{}));
  expect(screen.queryByText('Còn 10 hộp')).toBeNull();
});


it('AI is only a floating assistant: expand does not navigate to another page',async()=>{
  renderChat();
  expect(screen.queryByRole('button',{name:'Mở trang AI'})).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:'Phóng to cửa sổ'}));
  expect(screen.getByRole('dialog',{name:'Trợ lý An Tâm AI'})).toBeTruthy();
  expect(screen.getByRole('button',{name:'Thu nhỏ cửa sổ'})).toBeTruthy();
  fireEvent.click(screen.getByRole('button',{name:'Thu nhỏ cửa sổ'}));
  expect(screen.getByRole('region',{name:'Trợ lý An Tâm AI'})).toBeTruthy();
  fireEvent.click(screen.getByRole('button',{name:'Đóng chatbot'}));
  expect(screen.getByRole('button',{name:/Mở An Tâm AI/})).toBeTruthy();
});

it('AI floating launcher can collapse and move using arrow keys',()=>{
  render(<AI user={user}/>);
  const launcher=screen.getByRole('button',{name:/Mở An Tâm AI/});
  const floating=launcher.closest('.assistant-launcher-wrap');
  const previous=parseInt(floating.style.bottom,10);
  fireEvent.keyDown(launcher,{key:'ArrowUp'});
  expect(parseInt(floating.style.bottom,10)).toBeGreaterThan(previous);
  fireEvent.click(screen.getByRole('button',{name:'Thu gọn nút AI vào mép phải'}));
  expect(floating.className).toContain('is-compact');
  fireEvent.click(screen.getByRole('button',{name:'Mở rộng nút AI'}));
  expect(floating.className).not.toContain('is-compact');
});

it('sidebar has no dedicated AI route or navigation item',async()=>{
  api.mockImplementation(path=>Promise.resolve(path==='/auth/me'?user:data[path.split('?')[0]]??[]));
  render(<App/>);
  await screen.findByText('Trung tâm điều hành');
  expect(screen.queryByRole('navigation',{name:'An Tâm AI'})).toBeNull();
  expect(screen.queryByRole('button',{name:'An Tâm AI'})).toBeNull();
  expect(screen.getByRole('button',{name:/Mở An Tâm AI/})).toBeTruthy();
});

it('catalog shows 10 aligned rows per page and resets when searching',async()=>{
  const many=Array.from({length:24},(_,i)=>({...medicine,id:i+1,code:`T${i+1}`,name:`Thuốc số ${String(i+1).padStart(2,'0')}`}));
  api.mockImplementation(path=>Promise.resolve(path==='/medicines'?many:data[path.split('?')[0]]??[]));
  render(<Catalog type="medicines" user={user}/>);
  await screen.findByText('Thuốc số 01');
  const table=screen.getByRole('table');
  expect(table.querySelector('colgroup').children.length).toBe(table.querySelectorAll('thead th').length);
  expect(table.querySelectorAll('tbody tr').length).toBe(10);
  expect(screen.getByText('Hiển thị 1–10 / 24 dòng')).toBeTruthy();
  fireEvent.click(screen.getByRole('button',{name:'Trang 2'}));
  expect(screen.getByText('Thuốc số 11')).toBeTruthy();
  fireEvent.change(screen.getByLabelText('Tìm kiếm danh mục'),{target:{value:'Thuốc số 24'}});
  expect(screen.getByText('Thuốc số 24')).toBeTruthy();
  expect(screen.queryByText('Thuốc số 11')).toBeNull();
});

it('POS shows at most 6 distinct medicines per page and uses earliest lot',async()=>{
  const lots=Array.from({length:14},(_,i)=>({...batch,id:i+1,medicine_id:i+1,medicine_code:`M${i+1}`,medicine_name:`Thuốc POS ${String(i+1).padStart(2,'0')}`,code:`LOT-${i+1}`}));
  // A later batch for medicine 1 must not replace the earlier FEFO representative card.
  lots.push({...batch,id:90,medicine_id:1,medicine_name:'Thuốc POS 01',code:'LOT-LATER',expiry_date:'2028-01-01'});
  api.mockImplementation(path=>Promise.resolve(path.startsWith('/batches')?lots:data[path.split('?')[0]]??[]));
  render(<POS user={user}/>);
  await screen.findByText('Thuốc POS 01');
  expect(document.querySelectorAll('.product-grid .product').length).toBe(6);
  expect(screen.queryByText('LOT-LATER')).toBeNull();
  expect(screen.getByText('Hiển thị 1–6 / 14 thuốc')).toBeTruthy();
  fireEvent.click(screen.getByRole('button',{name:'Trang 2'}));
  expect(screen.getByText('Thuốc POS 07')).toBeTruthy();
  fireEvent.change(screen.getByPlaceholderText('Tìm tên thuốc, mã thuốc hoặc số lô…'),{target:{value:'Thuốc POS 14'}});
  expect(screen.getByText('Thuốc POS 14')).toBeTruthy();
  expect(screen.queryByText('Thuốc POS 07')).toBeNull();
});

it('floating AI launcher label is An Tâm AI without Hỏi',()=>{
  render(<AI user={user}/>);
  const btn=screen.getByRole('button',{name:/Mở An Tâm AI/});
  expect(btn.textContent).toContain('An Tâm AI');
  expect(btn.textContent).not.toContain('Hỏi');
});

it('reports show daily trend with real revenue and zero for days without sales',async()=>{
  const end=localToday();
  const day=i=>new Date(Date.parse(end+'T00:00:00Z')-i*86400000).toISOString().slice(0,10);
  const series=Array.from({length:24},(_,i)=>({date:day(i),revenue:String(100+i)}));
  api.mockImplementation(path=>Promise.resolve(path.startsWith('/reports')?{...report,series}:data[path.split('?')[0]]??[]));
  render(<Reports/>);
  const chart=await screen.findByRole('img',{name:/Biểu đồ đường thể hiện doanh thu/});
  expect(screen.getByText('24 ngày có doanh thu')).toBeTruthy();
  const points=within(chart).getAllByRole('img');
  expect(points).toHaveLength(30);
  expect(within(chart).getByRole('img',{name:`${date(day(0))}: ${money(100)}`})).toBeTruthy();
  const zero=within(chart).getByRole('img',{name:`${date(day(29))}: ${money(0)}`});
  fireEvent.focus(zero);
  expect(document.querySelector('.report-trend-caption [aria-live]').textContent).toBe(`${date(day(29))}: ${money(0)}`);
  expect(screen.queryByRole('button',{name:'Trang 3'})).toBeNull();
});

it('settings show both catalogs and keep edits scoped to the selected catalog',async()=>{
  render(<SettingsPage user={user}/>);
  fireEvent.click(screen.getByRole('button',{name:'Danh mục'}));
  const categories=screen.getByRole('region',{name:'Nhóm thuốc'});
  const units=screen.getByRole('region',{name:'Đơn vị tính'});
  await within(categories).findByText('Nhóm 1');
  await within(units).findByText('Hộp');
  fireEvent.change(within(categories).getByRole('textbox'),{target:{value:'không trùng'}});
  expect(within(categories).queryByText('Nhóm 1')).toBeNull();
  expect(within(units).getByText('Hộp')).toBeTruthy();
  fireEvent.click(within(units).getByRole('button',{name:'Sửa'}));
  const dialog=screen.getByRole('dialog');
  fireEvent.change(within(dialog).getByRole('textbox'),{target:{value:'Viên'}});
  fireEvent.click(within(dialog).getByRole('button',{name:'Lưu thông tin'}));
  await waitFor(()=>expect(send).toHaveBeenCalledWith('/units/1','PUT',{name:'Viên'}));
});

it('non-managers cannot access catalog settings',()=>{
  render(<SettingsPage user={{...user,role:'pharmacist'}}/>);
  expect(screen.queryByRole('button',{name:'Danh mục'})).toBeNull();
  expect(screen.getByLabelText('Mật khẩu hiện tại')).toBeTruthy();
});

it('settings catalogs paginate 10 rows independently and show pagination for a single page',async()=>{
  const categories=Array.from({length:23},(_,i)=>({id:i+1,name:`Nhóm ${String(i+1).padStart(2,'0')}`}));
  const units=Array.from({length:12},(_,i)=>({id:i+1,name:`Đơn vị ${String(i+1).padStart(2,'0')}`}));
  api.mockImplementation(path=>Promise.resolve(path==='/categories'?categories:path==='/units'?units:data[path.split('?')[0]]??[]));
  render(<SettingsPage user={user}/>);
  fireEvent.click(screen.getByRole('button',{name:'Danh mục'}));
  const left=screen.getByRole('region',{name:'Nhóm thuốc'}),right=screen.getByRole('region',{name:'Đơn vị tính'});
  await within(left).findByText('Nhóm 01');await within(right).findByText('Đơn vị 01');
  expect(within(left).getAllByRole('row')).toHaveLength(11);
  expect(within(right).getAllByRole('row')).toHaveLength(11);
  expect(within(left).getByText(/10 dòng\/trang/)).toBeTruthy();
  fireEvent.click(within(left).getByRole('button',{name:'Trang 2'}));
  expect(within(left).getByText('Nhóm 11')).toBeTruthy();
  expect(within(right).getByText('Đơn vị 01')).toBeTruthy();
  expect(within(right).queryByText('Đơn vị 11')).toBeNull();
  fireEvent.click(within(right).getByRole('button',{name:'Trang 2'}));
  expect(within(right).getByText('Đơn vị 11')).toBeTruthy();
});

it('POS keeps one quantity editor per medicine and splits 10 units by FEFO',async()=>{
  const lots=[{...batch,id:1,quantity:4,code:'LOT-EARLY',sale_price:'20000'}, {...batch,id:2,quantity:8,code:'LOT-LATE',expiry_date:'2028-03-01',sale_price:'25000'}];
  api.mockResolvedValue(lots);
  render(<POS user={user}/>);
  fireEvent.click(await screen.findByRole('button',{name:/Thuốc kiểm thử A/}));
  const input=screen.getByRole('spinbutton',{name:'Số lượng Thuốc kiểm thử A'});
  fireEvent.change(input,{target:{value:'10'}});
  expect(input.value).toBe('10');
  fireEvent.blur(input);
  expect(screen.getByRole('spinbutton',{name:'Số lượng Thuốc kiểm thử A'}).value).toBe('10');
  expect(screen.getByText(/4 Hộp ×/)).toBeTruthy();
  expect(screen.getByText(/6 Hộp ×/)).toBeTruthy();
  send.mockResolvedValue({id:11,customer:'Khách lẻ',total:'230000',items:[],status:'paid'});
  fireEvent.click(screen.getByRole('button',{name:'Xác nhận thanh toán'}));
  await waitFor(()=>expect(send).toHaveBeenCalled());
  expect(send.mock.calls[0][2].items).toEqual([{batch_id:1,quantity:4,expected_sale_price:'20000'},{batch_id:2,quantity:6,expected_sale_price:'25000'}]);
});

it('POS rejects overstock and clears stale alert when item removed',async()=>{
  render(<POS user={user}/>);
  fireEvent.click(await screen.findByRole('button',{name:/Thuốc kiểm thử A/}));
  const input=screen.getByRole('spinbutton',{name:'Số lượng Thuốc kiểm thử A'});
  fireEvent.change(input,{target:{value:'999'}});fireEvent.blur(input);
  expect(screen.getByText(/chỉ còn 10 Hộp khả dụng/)).toBeTruthy();
  expect(screen.getByRole('spinbutton',{name:'Số lượng Thuốc kiểm thử A'}).value).toBe('1');
  fireEvent.click(screen.getByRole('button',{name:'Bỏ Thuốc kiểm thử A khỏi giỏ'}));
  expect(screen.queryByText(/chỉ còn 10 Hộp khả dụng/)).toBeNull();
});
