// @vitest-environment jsdom
import React from 'react';
import {it,expect,vi,beforeEach,afterEach} from 'vitest';
import {render,screen,waitFor,fireEvent,cleanup} from '@testing-library/react';
import AIChat from './AIChat';
import {api,send,streamApi} from './api';
vi.mock('./api',()=>({api:vi.fn(),send:vi.fn(),streamApi:vi.fn()}));
const manager={id:1,name:'Quản lý',role:'manager'};
const Mark=({text})=><div>{text}</div>;
beforeEach(()=>{api.mockReset().mockResolvedValue([]);send.mockReset().mockResolvedValue({id:1,title:'Cuộc trò chuyện mới'});streamApi.mockReset()});
afterEach(cleanup);

it('keeps history accessible without auto-restoring saved conversation on login',async()=>{
  const c={id:7,title:'Kho',updated_at:'2026-10-28T08:00:00Z'};
  api.mockImplementation(async(path)=>path==='/ai/conversations'?[c]:{id:7,messages:[{
    id:99,role:'assistant',content:'Nội dung đã lưu',status:'complete',presentation:{
      cards:[{label:'Tổng tồn đã lưu',value:10}],
      table:{title:'Bảng đã lưu',columns:[{key:'name',label:'Thuốc'}],rows:[{name:'Mặt hàng đã lưu'}]},
      actions:[{page:'batches',label:'Mở kho đã lưu'}]}}]});
  const props={user:manager,MarkdownMessage:Mark,page:'batches'};
  let view=render(<AIChat {...props}/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  expect(await screen.findByText('Bạn muốn tìm hiểu điều gì hôm nay?')).toBeTruthy();
  expect(screen.queryByText('Nội dung đã lưu')).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:'Lịch sử trò chuyện'}));
  await screen.findByRole('button',{name:/Kho/});
  fireEvent.click(screen.getByRole('button',{name:/Kho/}));
  await screen.findByText('Nội dung đã lưu');
  fireEvent.click(screen.getByRole('button',{name:'Đóng chatbot'}));
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  expect(screen.getByText('Nội dung đã lưu')).toBeTruthy();
  view.unmount();
  view=render(<AIChat {...props}/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  expect(await screen.findByText('Bạn muốn tìm hiểu điều gì hôm nay?')).toBeTruthy();
  expect(screen.queryByText('Nội dung đã lưu')).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:'Lịch sử trò chuyện'}));
  await screen.findByRole('button',{name:/Kho/});
  fireEvent.click(screen.getByRole('button',{name:/Kho/}));
  await screen.findByText('Nội dung đã lưu');
  expect(screen.getByText('Tổng tồn đã lưu')).toBeTruthy();
  expect(screen.getByText('Mặt hàng đã lưu')).toBeTruthy();
  expect(screen.getByText('Bảng đã lưu').closest('details').open).toBe(false);
  expect(screen.getByRole('button',{name:'Mở kho đã lưu'})).toBeTruthy();
  expect(screen.queryByText('Phản hồi chưa hoàn tất.')).toBeNull();
});

it('does not mark a completed answer interrupted when the history list fails',async()=>{
  api.mockResolvedValueOnce([]).mockRejectedValue(new Error('network'));
  streamApi.mockImplementation(async(_url,_options,packet)=>{
    packet({type:'delta',text:'Câu trả lời hoàn tất'});
    packet({type:'done',status:'ok'});
  });
  render(<AIChat user={manager} MarkdownMessage={Mark} page='dashboard'/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  fireEvent.change(screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…'),{target:{value:'Kiểm tra'}});
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button',{name:'Gửi'}));
  await screen.findByText('Câu trả lời hoàn tất');
  await screen.findByRole('alert');
  expect(screen.queryByText('Phản hồi chưa hoàn tất.')).toBeNull();
});

it('shows page-specific suggestions',async()=>{
  render(<AIChat user={manager} MarkdownMessage={Mark} page="batches"/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  expect(await screen.findByRole('button',{name:/Những lô nào sắp hết hạn trong 90 ngày/})).toBeTruthy();
});

it('renders trusted cards and an in-app navigation action without technical badges',async()=>{
  const navigate=vi.fn();
  streamApi.mockImplementation(async (_url,_options,onPacket)=>{
    onPacket({type:'message',id:2});
    onPacket({type:'meta',used_tools:['sales_summary'],route:'ai',presentation:{
      cards:[{label:'Doanh thu đã thanh toán',value:250000,format:'currency'}],
      actions:[{page:'reports',label:'Mở báo cáo doanh thu'}],table:null,chart:null}});
    onPacket({type:'delta',text:'Tháng này nhà thuốc có doanh thu 250.000 ₫.'});
    onPacket({type:'done',status:'ok',model:'gemini-test'});
  });
  render(<AIChat user={manager} MarkdownMessage={Mark} page="dashboard" onNavigate={navigate}/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  fireEvent.change(screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…'),{target:{value:'Doanh thu?' }});
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button',{name:'Gửi'}));
  expect(await screen.findByText('Doanh thu đã thanh toán')).toBeTruthy();
  expect(screen.queryByText('Phân tích AI')).toBeNull();
  expect(screen.queryByText(/Đã tra cứu/)).toBeNull();
  expect(screen.queryByText(/Nguồn dữ liệu/)).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:'Mở báo cáo doanh thu'}));
  expect(navigate).toHaveBeenCalledWith('reports');
});

it('never offers a management report link to a cashier',async()=>{
  const cashier={id:3,name:'Thu ngân',role:'cashier'};
  streamApi.mockImplementation(async(_url,_opts,packet)=>{
    packet({type:'message',id:3});
    packet({type:'meta',presentation:{cards:[],table:null,chart:null,actions:[{page:'reports',label:'Mở báo cáo doanh thu'}]}});
    packet({type:'delta',text:'Bạn không có quyền xem doanh thu toàn hệ thống.'});
    packet({type:'done',status:'ok'});
  });
  render(<AIChat user={cashier} MarkdownMessage={Mark} page="pos" onNavigate={vi.fn()}/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  fireEvent.change(screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…'),{target:{value:'Doanh thu tháng này?'}});
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button',{name:'Gửi'}));
  await screen.findByText('Bạn không có quyền xem doanh thu toàn hệ thống.');
  expect(screen.queryByRole('button',{name:'Mở báo cáo doanh thu'})).toBeNull();
});


it('does not display chart or copy/regenerate actions in streamed answers', async()=>{
  const chart={title:'Tồn khả dụng một số thuốc',unit:'',bars:[{label:'Thuốc A',value:9}]};
  const presentation={cards:[],table:null,chart,actions:[{page:'batches',label:'Xem chi tiết tồn kho'}]};
  streamApi.mockImplementation(async (_url,_options,onPacket)=>{
    onPacket({type:'message',id:18});
    onPacket({type:'meta',presentation});
    onPacket({type:'delta',text:'Thuốc A hiện có 9 đơn vị.'});
    onPacket({type:'done',status:'ok',model:'gemini-test'});
  });
  render(<AIChat user={manager} MarkdownMessage={Mark} page='batches' onNavigate={vi.fn()}/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  fireEvent.change(screen.getByPlaceholderText('Nhắn tin cho An Tâm AI…'),{target:{value:'Tồn kho Thuốc A?' }});
  await waitFor(()=>expect(screen.getByRole('button',{name:'Gửi'}).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button',{name:'Gửi'}));
  expect(await screen.findByText('Thuốc A hiện có 9 đơn vị.')).toBeTruthy();
  expect(screen.queryByText('Tồn khả dụng một số thuốc')).toBeNull();
  expect(screen.queryByText('Sao chép')).toBeNull();
  expect(screen.queryByText('Tạo lại')).toBeNull();
  expect(screen.getByRole('button',{name:'Xem chi tiết tồn kho'})).toBeTruthy();
});


it('does not show a saved V5 chart after reopening conversation',async()=>{
  const c={id:7,title:'Tồn kho',updated_at:'2026-10-28T08:00:00Z'};
  api.mockImplementation(async(path)=>path==='/ai/conversations'?[c]:{id:7,messages:[{id:99,role:'assistant',content:'Đã tra cứu 9 đơn vị.',presentation:{cards:[],table:null,chart:{title:'Biểu đồ cũ',bars:[{label:'A',value:9}]},actions:[]}}]});
  render(<AIChat user={manager} MarkdownMessage={Mark} page='batches'/>);
  fireEvent.click(screen.getByRole('button',{name:/Mở An Tâm AI/}));
  expect(await screen.findByText('Bạn muốn tìm hiểu điều gì hôm nay?')).toBeTruthy();
  expect(screen.queryByText('Đã tra cứu 9 đơn vị.')).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:'Lịch sử trò chuyện'}));
  await screen.findByRole('button',{name:/Tồn kho/});
  fireEvent.click(screen.getByRole('button',{name:/Tồn kho/}));
  expect(await screen.findByText('Đã tra cứu 9 đơn vị.')).toBeTruthy();
  expect(screen.queryByText('Biểu đồ cũ')).toBeNull();
  expect(screen.queryByText('Sao chép')).toBeNull();
  expect(screen.queryByText('Tạo lại')).toBeNull();
});
