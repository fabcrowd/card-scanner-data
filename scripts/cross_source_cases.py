"""Accepted-wrong cases (native/250/200 cells) per model; F-only and B-only vs the others. Writes cases.txt."""
import json, sys, io
sys.path.insert(0, 'scripts')
from score_real_labels import base_name
B = json.load(open('data/reports/cross-source/boxes.json'))
out = io.open('data/reports/cross-source/cases.txt', 'w', encoding='utf-8')
def P(*a): print(*a, file=out)
def wrong(b, m):
    r = b['reads'][m]; return r['accepted'] and base_name(r['chosen_name'] or '') != base_name(b['gt_name'] or '')
def ln(b, cell):
    p = b['photo'].split('/'); rr = b['reads']
    s = lambda m: ('WRONG ' if wrong(b, m) else ('ok ' if rr[m]['accepted'] else 'abstain ')) + str(rr[m]['chosen_name'])
    return f"{b['src']} {p[-2]}/{p[-1]} box{[round(x) for x in b['box'][:2]]} cell={cell} GT={b['gt_name']} ({b['gt_id']}) | ship: {s('shipped')} | B: {s('B')} | F: {s('F')}"
cells = [c for c in B if c['cell'] in ('native', 250, 200)]
for m in ('shipped', 'B', 'F'):
    P(f"== all accepted-wrong for {m} at native/250/200:", sum(wrong(b, m) for b in cells))
for tag, sel in (('F-only', 'F'), ('B-only', 'B'), ('ship-only', 'shipped')):
    others = [m for m in ('shipped', 'B', 'F') if m != sel]
    P(f"== {tag} wrong (others not wrong in same cell/box)")
    for b in cells:
        if wrong(b, sel) and not any(wrong(b, o) for o in others):
            P(ln(b, b['cell']))
P("== F wrong, any cell native/250/200")
for b in cells:
    if wrong(b, 'F'): P(ln(b, b['cell']))
out.close()
