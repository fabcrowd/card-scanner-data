import json, os
R = json.load(open("data/reports/cross-source/results.json"))
order = ["native>=200", "native<200", "native_all", "250", "200", "150", "120"]
for src in ["hibid", "goodwill-pack2", "ALL"]:
    print(f"\n### {src}\n")
    print("| cell | model | n | top1 name | top5 name | accepted | acc-right | ACC-WRONG | sure | SURE-WRONG | hub1 | hub5 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in order:
        for m in ["shipped", "B", "F"]:
            d = R[src][m].get(c)
            if not d: continue
            print(f"| {c} | {m} | {d['n']} | {d['top1_name']} | {d['top5_name']} | {d['accepted']} | {d['accepted_right']} | {d['accepted_wrong']} | {d['sure']} | {d['sure_wrong']} | {d['hub_top1_share']} | {d['hub_top5_share']} |")
