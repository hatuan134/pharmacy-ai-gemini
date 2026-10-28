"""Stage 1 regression coverage: locked pricing, FIFO/FEFO, invoice visibility."""
from datetime import timedelta
from uuid import uuid4
from sqlalchemy import select
from app.models import Batch, InvoiceItem, Movement
from app.config import today


def item(batch_id, qty, price):
    return {'batch_id': batch_id, 'quantity': qty, 'expected_sale_price': price}


def new_batch(ids, code, expiry_days, quantity=10, price='2300'):
    return {'medicine_id':ids['medicine'], 'supplier_id':ids['supplier'],
            'code':code, 'received_date':str(today()),
            'expiry_date':str(today()+timedelta(days=expiry_days)),
            'quantity': quantity, 'purchase_price':'1000', 'sale_price':price}


def test_price_and_stock_adjustments_manager_only(setup):
    c,S,ids,tokens=setup
    c.cookies.set('session',tokens['pharmacist'])
    assert c.patch(f'/api/batches/{ids["batch"]}/price',json={'sale_price':2500}).status_code==403
    assert c.post(f'/api/batches/{ids["batch"]}/adjust',json={'quantity':9,'expected_quantity':10,'reason':'Đối soát tồn kho'}).status_code==403
    with S() as db:
        assert db.get(Batch,ids['batch']).quantity==10
        assert db.get(Batch,ids['batch']).sale_price==2000


def test_pharmacist_new_batch_uses_existing_price_only(setup):
    c,S,ids,tokens=setup
    c.cookies.set('session',tokens['pharmacist'])
    assert c.post('/api/batches',json=new_batch(ids,'MORE',90,price='2400')).status_code==403
    accepted=c.post('/api/batches',json=new_batch(ids,'MORE',90,price='2000'))
    assert accepted.status_code==200, accepted.text


def test_fefo_non_nearest_lot_rejected_and_multilot_sale(setup):
    c,S,ids,_=setup
    # Original lot B1 expires in 30 days, new lot expires in 15 days.
    newer=c.post('/api/batches',json=new_batch(ids,'EARLY',15,quantity=4,price='2300')).json()
    rejected=c.post('/api/invoices',json={'items':[item(ids['batch'],3,2000)],'request_key':str(uuid4())})
    assert rejected.status_code==409 and 'FEFO' in rejected.text
    order={'items':[item(newer['id'],4,'2300'),item(ids['batch'],3,'2000')], 'request_key':str(uuid4())}
    response=c.post('/api/invoices',json=order)
    assert response.status_code==200,response.text
    assert float(response.json()['total'])==15200
    with S() as db:
        assert db.get(Batch,newer['id']).quantity==0
        assert db.get(Batch,ids['batch']).quantity==7
        lines=list(db.scalars(select(InvoiceItem).where(InvoiceItem.invoice_id==response.json()['id'])))
        assert sorted((i.batch_id,i.quantity) for i in lines)==sorted([(newer['id'],4),(ids['batch'],3)])


def test_all_roles_see_all_invoices_but_only_manager_cancels(setup):
    c,S,ids,tokens=setup
    receipt=c.post('/api/invoices',json={'items':[item(ids['batch'],1,2000)],'request_key':str(uuid4())}).json()
    for role in ('pharmacist','cashier'):
        c.cookies.set('session',tokens[role])
        assert any(r['id']==receipt['id'] for r in c.get('/api/invoices').json())
        assert c.get('/api/invoices/'+str(receipt['id'])).status_code==200
        assert c.post('/api/invoices/'+str(receipt['id'])+'/cancel',json={'reason':'Hàng hoàn trả nguyên vẹn'}).status_code==403
        assert c.get('/api/reports').status_code==403
