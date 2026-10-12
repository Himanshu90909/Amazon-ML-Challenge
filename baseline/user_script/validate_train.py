import sys, time, csv
sys.path.insert(0, '.')
from entity_resolution import rows, build_indexes, candidates, accept

DATA_DIR = "../data/student_resource/dataset/train"
OUT_DIR = "../pipeline/work/train_userscript"
import os
os.makedirs(OUT_DIR, exist_ok=True)

s1_path = f"{DATA_DIR}/train_source1.tsv"
sources = [("2", f"{DATA_DIR}/train_source2.tsv"), ("3", f"{DATA_DIR}/train_source3.tsv")]

t0 = time.time()
print("Building indexes...", flush=True)
records, exact, postings = build_indexes(sources, max_df=1000)
print(f"Indexed {len(records):,} candidates; {len(postings):,} rare-token blocks in {time.time()-t0:.1f}s", flush=True)

thresholds = (0.92, 0.94, 0.78)

# load ground truth: source1_entity_id -> set(matched_ids)
gt = {}
with open(f"{DATA_DIR}/train_ground_truth.tsv", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        ids = row["matched_entity_ids"]
        gt[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()

t0 = time.time()
tp = fp = fn = 0
cand_recall_hit = 0
cand_recall_total = 0
n_rows = 0
match_path = f"{OUT_DIR}/matching_results.tsv"
with open(match_path, "w", encoding="utf-8", newline="") as mf:
    mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
    mw.writerow(["source1_entity_id", "matched_entity_ids"])
    for r in rows(s1_path):
        n_rows += 1
        rid = r["entity_id"]
        c = candidates(r, exact, postings)
        pred = set(x for x in c if accept(r, records[x], thresholds))
        truth = gt.get(rid, set())

        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
        if truth:
            cand_recall_total += len(truth)
            cand_recall_hit += len(truth & c)

        mw.writerow([rid, ",".join(sorted(pred))])
        if n_rows % 200000 == 0:
            print(f"processed {n_rows:,} in {time.time()-t0:.1f}s | tp={tp} fp={fp} fn={fn}", flush=True)

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
beta = 0.5
f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) else 0.0
print(f"\nDONE in {time.time()-t0:.1f}s over {n_rows:,} rows", flush=True)
print(f"tp={tp} fp={fp} fn={fn}", flush=True)
print(f"precision={precision:.4f} recall={recall:.4f} F0.5={f_beta:.4f}", flush=True)
print(f"blocking recall (true matches present in candidate set): {cand_recall_hit}/{cand_recall_total} = {cand_recall_hit/cand_recall_total:.4f}", flush=True)
