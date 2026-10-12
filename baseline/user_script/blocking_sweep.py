"""Build full token postings ONCE (no max_df filtering), then sweep several max_df
cutoffs in-memory to find the sweet spot: how many candidates/row it costs vs how
much EXTRA true-match recall it buys beyond what exact name/address matching
already gives for free.
"""
import sys, csv, time, collections
sys.path.insert(0, '.')
from entity_resolution import rows, norm, toks

DATA_DIR = "../data/student_resource/dataset/train"
SAMPLE_S1 = 150000  # how many S1 rows (with >=1 true match) to evaluate against

t0 = time.time()
records = {}
exact_name = collections.defaultdict(set)
exact_addr = collections.defaultdict(set)
# (country, field, token) -> set of rid   (built once, filtered later by frequency)
postings_all = collections.defaultdict(set)

for path in (f"{DATA_DIR}/train_source2.tsv", f"{DATA_DIR}/train_source3.tsv"):
    for r in rows(path):
        rid, country = r["entity_id"], r["country"]
        name, addr = norm(r["business_name"]), norm(r["business_address"])
        records[rid] = (name, addr, country)
        if name:
            exact_name[(country, name)].add(rid)
        if addr:
            exact_addr[(country, addr)].add(rid)
        for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
            for tk in set(toks(text)):
                postings_all[(country, field, tk)].add(rid)

print(f"loaded+tokenized {len(records):,} s2/s3 records in {time.time()-t0:.1f}s "
      f"({len(postings_all):,} distinct (country,field,token) keys)", flush=True)

gt = {}
with open(f"{DATA_DIR}/train_ground_truth.tsv", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        ids = row["matched_entity_ids"]
        if ids.strip():
            gt[row["source1_entity_id"]] = set(ids.split(","))

t0 = time.time()
s1_info = {}
s1_toks = {}
for r in rows(f"{DATA_DIR}/train_source1.tsv"):
    rid = r["entity_id"]
    if rid in gt:
        if len(s1_info) >= SAMPLE_S1:
            continue
        qn, qa, country = norm(r["business_name"]), norm(r["business_address"]), r["country"]
        s1_info[rid] = (qn, qa, country)
        name_tok = set(toks(r["business_name"]))
        addr_tok = set(toks(r["business_address"]))
        s1_toks[rid] = (name_tok, addr_tok)

print(f"prepared {len(s1_info):,} sample S1 rows in {time.time()-t0:.1f}s", flush=True)

# document frequency per (country, field, token)
freq = {k: len(v) for k, v in postings_all.items()}

for max_df in (25, 50, 100, 200, 500, 1000):
    t0 = time.time()
    total_true = 0
    hit_exact = 0
    hit_exact_or_token = 0
    total_candidates = 0
    n_rows = 0
    for s1id, (qn, qa, country) in s1_info.items():
        truth = gt[s1id]
        n_rows += 1
        cand = set()
        name_set = exact_name.get((country, qn), set()) if qn else set()
        addr_set = exact_addr.get((country, qa), set()) if qa else set()
        cand |= name_set
        cand |= addr_set
        name_tok, addr_tok = s1_toks[s1id]
        for field, tokset in (("n", name_tok), ("a", addr_tok)):
            for tk in tokset:
                key = (country, field, tk)
                if freq.get(key, 10**9) <= max_df:
                    cand |= postings_all.get(key, set())
        total_candidates += len(cand)
        for mid in truth:
            total_true += 1
            in_exact = mid in name_set or mid in addr_set
            if in_exact:
                hit_exact += 1
            if in_exact or mid in cand:
                hit_exact_or_token += 1
    print(f"max_df={max_df}: avg candidates/row={total_candidates/n_rows:.1f} | "
          f"recall(exact only)={hit_exact/total_true:.4f} | "
          f"recall(exact+token)={hit_exact_or_token/total_true:.4f} | "
          f"({time.time()-t0:.1f}s)", flush=True)
