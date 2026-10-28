"""V12 business report regression tests; uses SQLite test fixture, no external APIs."""
from datetime import timedelta
from decimal import Decimal
from uuid import uuid4
from app.config import today
from app.models import Invoice, InvoiceItem, Batch


def _sell(client, batch_id, quantity=2, unit_price=2000):
    return client.post('/api/invoices', json={
        'request_key': str(uuid4()),
        'items': [{'batch_id': batch_id, 'quantity': quantity, 'expected_sale_price': unit_price}],
    })


def test_reports_v12_empty_and_inventory_snapshot(setup):
    client, _, _, tokens = setup
    result = client.get('/api/reports').json()
    assert result['invoice_count'] == 0
    assert result['top_medicines'] == []
    assert result['inventory']['available_units'] == 10
    assert result['inventory']['available_batches'] == 1
    assert result['inventory']['expiring_batches'] == 1
    assert result['inventory']['expired_batches'] == 0
    assert result['inventory']['low_stock_count'] == 0
    assert result['inventory']['as_of'] == today().isoformat()
    client.cookies.set('session', tokens['pharmacist'])
    assert client.get('/api/reports').status_code == 403
    client.cookies.set('session', tokens['cashier'])
    assert client.get('/api/reports').status_code == 403


def test_reports_v12_top_sellers_paid_only_and_historical_prices(setup):
    client, Session, ids, _ = setup
    first = _sell(client, ids['batch'], 2)
    assert first.status_code == 200, first.text
    # Stored sale price changes should not retroactively change historical sales.
    assert client.patch(f"/api/batches/{ids['batch']}/price", json={'sale_price': 5000}).status_code == 200
    second = _sell(client, ids['batch'], 1, 5000)
    assert second.status_code == 200, second.text
    payload = client.get('/api/reports').json()
    assert Decimal(str(payload['revenue'])) == Decimal('9000')
    assert payload['invoice_count'] == 2
    assert len(payload['top_medicines']) == 1
    leader = payload['top_medicines'][0]
    assert leader['medicine_id'] == ids['medicine']
    assert leader['name'] == 'Thuốc kiểm thử'
    assert leader['unit'] == 'Hộp'
    assert leader['quantity'] == 3
    assert Decimal(str(leader['revenue'])) == Decimal('9000')
    assert payload['inventory']['available_units'] == 7

    assert client.post(f"/api/invoices/{first.json()['id']}/cancel", json={'reason': 'Khách trả hàng'}).status_code == 200
    after = client.get('/api/reports').json()
    assert after['invoice_count'] == 1
    assert after['top_medicines'][0]['quantity'] == 1
    assert Decimal(str(after['top_medicines'][0]['revenue'])) == Decimal('5000')
    assert after['inventory']['available_units'] == 9


def test_reports_v12_date_filter_excludes_sale_outside_period(setup):
    client, Session, ids, _ = setup
    result = _sell(client, ids['batch'], 1)
    assert result.status_code == 200, result.text
    past = today()-timedelta(days=7)
    payload = client.get('/api/reports', params={'start': str(past), 'end': str(past)}).json()
    assert payload['invoice_count'] == 0
    assert payload['top_medicines'] == []
    # Inventory remains current, independent of the historical period.
    assert payload['inventory']['available_units'] == 9
