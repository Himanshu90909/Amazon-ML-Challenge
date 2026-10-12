"""Fast diagnostic: of the true matches in train ground truth, how many are already
found by an EXACT normalized name/address match (cheap), vs needing fuzzy token
blocking (expensive, and what blew up the candidate counts to ~570/row)?
"""
import sys, csv, time, collections
sys.path.insert(0, '.')
from entity_resolution import rows, norm

DATA_DIR = "../data/student_resource/dataset/train"

t0 = time.time()
records = {}  # rid -> (norm_name, norm_addr, country)
exact_name = collections.defaultdict(set)   # (country, norm_name) -> {rid}
exact_addr = collections.defaultdict(set)   # (country, norm_addr) -> {rid}

for path in (f"{DATA_DIR}/train_source2.tsv", f"{DATA_DIR}/train_source3.tsv"):
    for r in rows(path):
        rid, country = r["entity_id"], r["country"]
        name, addr = norm(r["business_name"]), norm(r["business_address"])
        records[rid] = (name, addr, country)
        if name:
            exact_name[(country, name)].add(rid)
        if addr:
            exact_addr[(country, addr)].add(rid)

print(f"loaded {len(records):,} s2/s3 records in {time.time()-t0:.1f}s", flush=True)

gt = {}
with open(f"{DATA_DIR}/train_ground_truth.tsv", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        ids = row["matched_entity_ids"]
        if ids.strip():
            gt[row["source1_entity_id"]] = set(ids.split(","))

print(f"{len(gt):,} S1 rows have >=1 true match", flush=True)

s1_info = {}
for r in rows(f"{DATA_DIR}/train_source1.tsv"):
    if r["entity_id"] in gt:
        s1_info[r["entity_id"]] = (norm(r["business_name"]), norm(r["business_address"]), r["country"])

total_true = 0
hit_name_exact = 0
hit_addr_exact = 0
hit_either_exact = 0
hit_neither = 0
neither_examples = []

for s1id, truth in gt.items():
    qn, qa, country = s1_info[s1id]
    name_set = exact_name.get((country, qn), set()) if qn else set()
    addr_set = exact_addr.get((country, qa), set()) if qa else set()
    for mid in truth:
        total_true += 1
        in_name = mid in name_set
        in_addr = mid in addr_set
        if in_name:
            hit_name_exact += 1
        if in_addr:
            hit_addr_exact += 1
        if in_name or in_addr:
            hit_either_exact += 1
        else:
            hit_neither += 1
            if len(neither_examples) < 8:
                neither_examples.append((s1id, mid))

print(f"\ntotal true match pairs: {total_true:,}")
print(f"caught by exact NAME match: {hit_name_exact:,} ({hit_name_exact/total_true:.2%})")
print(f"caught by exact ADDR match: {hit_addr_exact:,} ({hit_addr_exact/total_true:.2%})")
print(f"caught by EITHER exact match: {hit_either_exact:,} ({hit_either_exact/total_true:.2%})")
print(f"need fuzzy/token blocking (neither exact): {hit_neither:,} ({hit_neither/total_true:.2%})")
print("\nsample pairs needing fuzzy blocking:")
for s1id, mid in neither_examples:
    qn, qa, country = s1_info[s1id]
    cn, ca, ccountry = records[mid]
    print(f"  S1={s1id} name={qn!r} addr={qa!r}")
    print(f"  {mid}    name={cn!r} addr={ca!r}")
