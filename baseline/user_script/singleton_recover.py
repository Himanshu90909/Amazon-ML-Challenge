import csv, re, unicodedata, collections, time, pickle
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
t0 = time.time()
freq = collections.Counter()
records = {}
for n in (2, 3):
    for r in rows(f"{TESTDIR}/test_source{n}.tsv"):
        rid, country = r["entity_id"], r["country"]
        records[rid] = (norm(r["business_name"]), norm(r["business_address"]), country,
                         norm_ws(r["business_address"]), norm_ws(r["business_name"]))
        for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
            for t in set(toks(text)):
                freq[(country, field, t)] += 1
print(f"pass1 done {time.time()-t0:.1f}s, {len(records)} records", flush=True)

MAX_DF = 300
postings = collections.defaultdict(list)
for n in (2, 3):
    for r in rows(f"{TESTDIR}/test_source{n}.tsv"):
        rid, country = r["entity_id"], r["country"]
        for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
            for t in set(toks(text)):
                if freq[(country, field, t)] <= MAX_DF:
                    postings[(country, field, t)].append(rid)
print(f"pass2 done {time.time()-t0:.1f}s, {len(postings)} postings", flush=True)

# load current submission, find empty rows
INFILE = "../../incoming_review/downloaded_file"
current = {}
with open(INFILE, encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        current[row[0]] = row[1] if len(row) > 1 else ""

s1_path = f"{TESTDIR}/test_source1.tsv"
empty_rows = [r for r in rows(s1_path) if not current.get(r["entity_id"])]
print(f"empty rows: {len(empty_rows)}", flush=True)

# strict fuzzy pass: only add if name>=93 AND addr_token_set>=90, single top candidate, unambiguous
additions = {}
t1 = time.time()
for i, r in enumerate(empty_rows):
    country = r["country"]
    nm_q, ad_q = norm(r["business_name"]), norm(r["business_address"])
    adws_q, nmws_q = norm_ws(r["business_address"]), norm_ws(r["business_name"])
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
        nm, ad, c, adws, nmws = records[rid]
        if c != country:
            continue
        ns = ratio(nm_q, nm) / 100.0 if nm else 0.0
        ass = token_set_ratio(adws_q, adws) / 100.0 if (adws and adws_q) else 0.0
        if ns >= 0.93 and ass >= 0.90:
            best.append((rid, ns, ass))
    if len(best) == 1:
        additions[r["entity_id"]] = best[0][0]
    if (i + 1) % 20000 == 0:
        print(f"  scanned {i+1}/{len(empty_rows)} empty rows in {time.time()-t1:.1f}s, additions so far={len(additions)}", flush=True)

print(f"DONE. total strict-fuzzy additions: {len(additions)} out of {len(empty_rows)} empty rows", flush=True)
with open("/tmp/fuzzy_additions.pkl", "wb") as f:
    pickle.dump(additions, f)
print("saved /tmp/fuzzy_additions.pkl")
