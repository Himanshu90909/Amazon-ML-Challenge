import csv, re, unicodedata, collections, time, pickle, sys
from rapidfuzz.fuzz import ratio, token_set_ratio

STOP = set("the and of for inc llc ltd limited corp corporation company co pvt private llp no near road rd street st lane ln avenue ave block india united states usa".split())

def norm(s):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)
def norm_ws(s):
    t = re.sub(r"[^\w\s]+", " ", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()
def toks(s):
    return [x for x in re.findall(r"\w+", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)
            if len(x) >= 3 and x not in STOP]
def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")

TESTDIR = "../data/student_resource/dataset/test"
EXPECTED = {2: 4887273, 3: 5082316}

t0 = time.time()
freq = collections.Counter()
records = {}
for n in (2, 3):
    cnt = 0
    for r in rows(f"{TESTDIR}/test_source{n}.tsv"):
        cnt += 1
        rid, country = r["entity_id"], r["country"]
        records[rid] = (norm(r["business_name"]), norm(r["business_address"]), country,
                         norm_ws(r["business_address"]))
        for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
            for t in set(toks(text)):
                freq[(country, field, t)] += 1
    print(f"file {n}: rows={cnt} (expected {EXPECTED[n]}) records_dict={len(records)} elapsed={time.time()-t0:.1f}s", flush=True)
    if cnt != EXPECTED[n]:
        print(f"ABORT: row count mismatch for file {n}, likely filesystem truncation. Stopping.", flush=True)
        sys.exit(1)

print(f"pass1 OK: {len(records)} records in {time.time()-t0:.1f}s", flush=True)

MAX_DF = 300
postings = collections.defaultdict(list)
for n in (2, 3):
    cnt = 0
    for r in rows(f"{TESTDIR}/test_source{n}.tsv"):
        cnt += 1
        rid, country = r["entity_id"], r["country"]
        for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
            for t in set(toks(text)):
                if freq[(country, field, t)] <= MAX_DF:
                    postings[(country, field, t)].append(rid)
    if cnt != EXPECTED[n]:
        print(f"ABORT: row count mismatch pass2 file {n}. Stopping.", flush=True)
        sys.exit(1)
print(f"pass2 OK: {len(postings)} postings in {time.time()-t0:.1f}s", flush=True)

INFILE = "../../incoming_review/downloaded_file"
current = {}
with open(INFILE, encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        current[row[0]] = row[1] if len(row) > 1 else ""
print(f"current submission loaded: {len(current)} rows", flush=True)

s1_path = f"{TESTDIR}/test_source1.tsv"
empty_rows = [r for r in rows(s1_path) if not current.get(r["entity_id"])]
print(f"empty rows: {len(empty_rows)}", flush=True)

additions = {}
missing_rid_errors = 0
t1 = time.time()
for i, r in enumerate(empty_rows):
    country = r["country"]
    nm_q = norm(r["business_name"])
    adws_q = norm_ws(r["business_address"])
    evidence = collections.Counter()
    for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
        weight = 2 if field == "n" else 1
        for t in set(toks(text)):
            for rid in postings.get((country, field, t), ()):
                evidence[rid] += weight
    if not evidence:
        continue
    top = [rid for rid, _ in evidence.most_common(30)]
    best = []
    for rid in top:
        rec = records.get(rid)
        if rec is None:
            missing_rid_errors += 1
            continue
        nm, ad, c, adws = rec
        if c != country:
            continue
        ns = ratio(nm_q, nm) / 100.0 if nm else 0.0
        ass = token_set_ratio(adws_q, adws) / 100.0 if (adws and adws_q) else 0.0
        if ns >= 0.93 and ass >= 0.90:
            best.append((rid, ns, ass))
    if len(best) == 1:
        additions[r["entity_id"]] = best[0][0]
    if (i + 1) % 20000 == 0:
        print(f"  scanned {i+1}/{len(empty_rows)} in {time.time()-t1:.1f}s, additions={len(additions)}, missing_rid_errors={missing_rid_errors}", flush=True)

print(f"DONE. total strict-fuzzy additions: {len(additions)} / {len(empty_rows)} empty rows. missing_rid_errors={missing_rid_errors}", flush=True)
with open("/tmp/fuzzy_additions2.pkl", "wb") as f:
    pickle.dump(additions, f)
print("saved /tmp/fuzzy_additions2.pkl", flush=True)
