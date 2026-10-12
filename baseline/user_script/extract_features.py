"""Build a per-candidate-pair labeled feature table on a SAMPLE of source-1 rows,
so we can tune thresholds / fit a classifier fast instead of re-running the full
slow row-by-row accept() loop for every threshold guess.
"""
import sys, time, csv
sys.path.insert(0, '.')
from entity_resolution import rows, build_indexes, candidates, norm, norm_ws
from rapidfuzz.fuzz import ratio, token_set_ratio

DATA_DIR = "../data/student_resource/dataset/train"
OUT_DIR = "../pipeline/work/train_userscript"
import os
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000

s1_path = f"{DATA_DIR}/train_source1.tsv"
sources = [("2", f"{DATA_DIR}/train_source2.tsv"), ("3", f"{DATA_DIR}/train_source3.tsv")]

t0 = time.time()
print("Building indexes...", flush=True)
records, exact, postings = build_indexes(sources, max_df=1000)
print(f"Indexed {len(records):,} candidates; {len(postings):,} rare-token blocks in {time.time()-t0:.1f}s", flush=True)

# precompute space-preserving name for every record once (used for name-token jaccard)
name_ws_cache = {}

gt = {}
with open(f"{DATA_DIR}/train_ground_truth.tsv", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        ids = row["matched_entity_ids"]
        gt[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


t0 = time.time()
out_path = f"{OUT_DIR}/features_sample.tsv"
n_rows = 0
n_pairs = 0
cand_recall_hit = 0
cand_recall_total = 0
with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["s1_id", "cand_id", "ns", "ass", "name_jacc", "addr_jacc", "exact_name", "exact_addr", "label"])
    for r in rows(s1_path):
        n_rows += 1
        if n_rows > SAMPLE_N:
            break
        rid = r["entity_id"]
        c = candidates(r, exact, postings)
        truth = gt.get(rid, set())
        if truth:
            cand_recall_total += len(truth)
            cand_recall_hit += len(truth & c)

        q_name_n = norm(r["business_name"])
        q_addr_ws = norm_ws(r["business_address"])
        q_name_ws = norm_ws(r["business_name"])
        q_name_tokset = set(q_name_ws.split())
        q_addr_tokset = set(q_addr_ws.split())

        for cid in c:
            name, addr, country, source, addr_ws = records[cid]
            ns = ratio(q_name_n, name) / 100.0 if name else 0.0
            ass = token_set_ratio(q_addr_ws, addr_ws) / 100.0 if addr_ws and q_addr_ws else 0.0

            name_ws_c = name_ws_cache.get(cid)
            if name_ws_c is None:
                # reconstruct: we don't have raw name text here, approximate via squished
                # -> not reliable for jaccard, so instead cache the address token set only
                name_ws_c = ""
                name_ws_cache[cid] = name_ws_c
            c_addr_tokset = set(addr_ws.split()) if addr_ws else set()

            addr_j = jaccard(q_addr_tokset, c_addr_tokset)

            w.writerow([rid, cid, f"{ns:.4f}", f"{ass:.4f}", "", f"{addr_j:.4f}",
                        int(q_name_n == name), int(q_addr_ws == addr_ws),
                        int(cid in truth)])
            n_pairs += 1
        if n_rows % 50000 == 0:
            print(f"processed {n_rows:,} rows, {n_pairs:,} pairs in {time.time()-t0:.1f}s", flush=True)

print(f"DONE {n_rows:,} rows -> {n_pairs:,} pairs in {time.time()-t0:.1f}s", flush=True)
if cand_recall_total:
    print(f"blocking recall on sample: {cand_recall_hit}/{cand_recall_total} = {cand_recall_hit/cand_recall_total:.4f}", flush=True)
