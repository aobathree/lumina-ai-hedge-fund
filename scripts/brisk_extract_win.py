#!/usr/bin/env python3
"""BRiSK の全銘柄の録音（aobathree-brisk docs/17、brisk-nautilus-parquet-v3）から、指定の銘柄だけを JSON で出す（読むだけ）。

録音は Windows にしか無く、Mac へ写さない（aobathree-brisk の規律）。Windows の python に標準入力で渡して、
Windows の中で読み、結果だけを標準出力で受け取る:

  ssh kabs-win 'python -I - preopen 20261120 7011 5803 …' < scripts/brisk_extract_win.py > preopen.json
  ssh kabs-win 'python -I - trades  20261120 7011 5803 …' < scripts/brisk_extract_win.py > trades.json

preopen: 09:00:00 より前の最後の気配の更新（板 10 段、予定値段・株数、特別気配の印、成行の株数）と銘柄の定義
trades:  その日の全約定（片をまたいだ重複は trade_id で除く）。[時刻 ms（UNIX）, 値段, 株数, 仕掛けた側]
"""
import datetime as dt
import glob
import json
import os
import sys

JST = dt.timezone(dt.timedelta(hours=9))
ROOT = os.path.join(os.path.expanduser('~'), 'brisk-data', 'brisk-recordings', 'sbi')


def pieces(day):
    found = sorted(glob.glob(os.path.join(ROOT, f'sbi_{day}_*')))
    if not found:
        raise SystemExit(f'{day} の片が無い: {ROOT}')
    return found


def _num(v):
    return None if v is None else float(v)


def preopen(day, codes):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    ids = [f'{c}.XTKS' for c in codes]
    first = pieces(day)[0]
    open_ns = int(dt.datetime.strptime(day, '%Y%m%d').replace(hour=9, tzinfo=JST).timestamp() * 1e9)
    cols = ['identifier', 'ts_event', 'bids', 'asks', 'last_price', 'indicative_price', 'indicative_volume',
            'indicative_imbalance_side', 'reference_price', 'quote_flag', 'quote_side', 'market_buy_quantity',
            'market_sell_quantity', 'bid_total_quantity', 'ask_total_quantity', 'volume']
    t = pq.read_table(os.path.join(first, 'data', 'order_book_depths.parquet'), columns=cols,
                      filters=[('identifier', 'in', ids)])
    ts = pc.cast(t['ts_event'], 'int64')
    t = t.filter(pc.less(ts, open_ns))
    eq = pq.read_table(os.path.join(first, 'data', 'equity.parquet'),
                       columns=['id', 'name', 'base_price', 'max_price', 'min_price', 'lot_size'],
                       filters=[('id', 'in', ids)]).to_pylist()
    out = {'trading_date': day, 'piece': os.path.basename(first), 'codes': {}}
    for e in eq:
        out['codes'][e['id'].split('.')[0]] = {'name': e['name'], 'base_price': _num(e['base_price']),
                                               'max_price': e['max_price'], 'min_price': e['min_price'],
                                               'lot_size': e['lot_size'], 'book': None}
    ident = t['identifier'].to_pylist()
    last_idx = {}
    for i, s in enumerate(ident):
        last_idx[s] = i
    for s, i in last_idx.items():
        row = t.slice(i, 1).to_pylist()[0]
        book = {
            'ts_jst': row['ts_event'].astimezone(JST).isoformat(),
            'ts_unix': row['ts_event'].timestamp(),
            'bids': [{'price': _num(l['price']), 'size': _num(l['size']), 'orders': l['count']} for l in row['bids']],
            'asks': [{'price': _num(l['price']), 'size': _num(l['size']), 'orders': l['count']} for l in row['asks']],
        }
        for k in cols[4:]:
            book[k] = _num(row[k]) if row[k] is not None else None
        out['codes'].setdefault(s.split('.')[0], {})['book'] = book
    return out


def trades(day, codes):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    ids = [f'{c}.XTKS' for c in codes]
    seen = {}
    out = {'trading_date': day, 'pieces': [], 'codes': {c: [] for c in codes}}
    for p in pieces(day):
        out['pieces'].append(os.path.basename(p))
        t = pq.read_table(os.path.join(p, 'data', 'trades.parquet'),
                          columns=['identifier', 'ts_event', 'price', 'size', 'aggressor_side', 'trade_id'],
                          filters=[('identifier', 'in', ids)])
        ms = (pc.cast(t['ts_event'], 'int64').to_numpy() // 1_000_000).tolist()
        price = [float(x) for x in t['price'].to_pylist()]
        size = [float(x) for x in t['size'].to_pylist()]
        side = [str(x) for x in t['aggressor_side'].to_pylist()]
        tid = t['trade_id'].to_pylist()
        for s, a, b, c, d, k in zip(t['identifier'].to_pylist(), ms, price, size, side, tid):
            code = s.split('.')[0]
            key = (code, k)
            if key in seen:
                continue
            seen[key] = True
            out['codes'][code].append([a, b, c, d])
    for code in out['codes']:
        out['codes'][code].sort(key=lambda r: r[0])
    return out


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    mode, day, codes = sys.argv[1], sys.argv[2], sys.argv[3:]
    res = preopen(day, codes) if mode == 'preopen' else trades(day, codes) if mode == 'trades' else None
    if res is None:
        raise SystemExit(f'mode は preopen か trades: {mode}')
    json.dump(res, sys.stdout, ensure_ascii=True)  # Windows の標準出力は cp932 なので ASCII で出す


if __name__ == '__main__':
    main()
