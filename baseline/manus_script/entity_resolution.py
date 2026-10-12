#!/usr/bin/env python3
"""Offline business entity resolution pipeline for the Amazon ML challenge.

Base: Manus AI's version (country-aware exact index + rare-token blocking with a
ranked top-K candidate cap + resumable chunked writes). Fixed: address scoring was
comparing the raw spaced query address against a SQUISHED (no-space) candidate
address via token_set_ratio, which needs whitespace to find token boundaries -- it
silently degraded to near char-ratio. Now both sides use a space-preserving norm.

Example:
  python entity_resolution.py --data-dir dataset/test --output-dir output
"""
import argparse, csv, re, unicodedata, collections, os, time, heapq
from rapidfuzz.fuzz import ratio, token_set_ratio

STOP = set("the and of for inc llc ltd limited corp corporation company co pvt private llp no near road rd street st lane ln avenue ave block india united states usa".split())

def norm(s):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)

def norm_ws(s):
    """Like norm(), but keeps single spaces between tokens -- needed for
    token_set_ratio, which relies on whitespace to find token boundaries."""
    t = re.sub(r"[^\w\s]+", " ", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()

def toks(s):
    return [x for x in re.findall(r"\w+", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)
            if len(x) >= 3 and x not in STOP]

def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")

def build_indexes(src_paths, max_df=200):
    freq = collections.Counter()
    records = {}
    exact = collections.defaultdict(list)
    for source, path in src_paths:
        for r in rows(path):
            rid, country = r["entity_id"], r["country"]
            name, addr = norm(r["business_name"]), norm(r["business_address"])
            addr_ws = norm_ws(r["business_address"])
            records[rid] = (name, addr, country, source, addr_ws)
            for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
                for t in set(toks(text)):
                    freq[(country, field, t)] += 1
            if name: exact[(country, "n", name)].append(rid)
            if addr: exact[(country, "a", addr)].append(rid)
    postings = collections.defaultdict(list)
    for source, path in src_paths:
        for r in rows(path):
            rid, country = r["entity_id"], r["country"]
            for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
                for t in set(toks(text)):
                    if freq[(country, field, t)] <= max_df:
                        postings[(country, field, t)].append(rid)
    return records, exact, postings

def candidates(r, exact, postings, limit=100):
    country, exact_ids, evidence = r["country"], set(), collections.Counter()
    for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
        nt = norm(text)
        if nt: exact_ids.update(exact.get((country, field, nt), ()))
        for t in set(toks(text)):
            weight = 2 if field == "n" else 1
            for rid in postings.get((country, field, t), ()):
                evidence[rid] += weight
    ranked = heapq.nlargest(limit, evidence, key=lambda rid: evidence[rid])
    return exact_ids | set(ranked)

def accept(query, rec, thresholds):
    name, addr, _, _, addr_ws = rec
    ns = ratio(norm(query["business_name"]), name) / 100.0 if name else 0.0
    q_addr_ws = norm_ws(query["business_address"])
    ass = token_set_ratio(q_addr_ws, addr_ws) / 100.0 if (addr_ws and q_addr_ws) else 0.0
    n_hi, a_hi, both = thresholds
    return (ns >= n_hi or ass >= a_hi or
            (ns >= both and ass >= both) or
            (ns >= 0.88 and ass >= 0.70) or
            (ass >= 0.88 and ns >= 0.70))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="directory containing test_source1/2/3.tsv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-df", type=int, default=200)
    ap.add_argument("--name-threshold", type=float, default=0.92)
    ap.add_argument("--address-threshold", type=float, default=0.94)
    ap.add_argument("--both-threshold", type=float, default=0.78)
    ap.add_argument("--candidate-limit", type=int, default=100)
    ap.add_argument("--start-row", type=int, default=0, help="0-based Source-1 row to start at")
    ap.add_argument("--end-row", type=int, default=None, help="exclusive 0-based Source-1 row end")
    ap.add_argument("--append", action="store_true", help="append rows without rewriting output headers")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    def src(n):
        p2 = os.path.join(args.data_dir, f"test_source{n}.tsv")
        return p2 if os.path.exists(p2) else os.path.join(args.data_dir, f"train_source{n}.tsv")
    s1 = os.path.join(args.data_dir, "test_source1.tsv") if os.path.exists(os.path.join(args.data_dir, "test_source1.tsv")) else os.path.join(args.data_dir, "train_source1.tsv")
    sources = [("2", src(2)), ("3", src(3))]
    t0 = time.time()
    print("Building country-aware indexes...", flush=True)
    records, exact, postings = build_indexes(sources, args.max_df)
    print(f"Indexed {len(records):,} candidates; {len(postings):,} rare-token blocks in {time.time()-t0:.1f}s", flush=True)
    thresholds = (args.name_threshold, args.address_threshold, args.both_threshold)
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    mode = "a" if args.append else "w"
    t0 = time.time()
    with open(match_path, mode, encoding="utf-8", newline="") as mf, open(cand_path, mode, encoding="utf-8", newline="") as cf:
        mw, cw = csv.writer(mf, delimiter="\t", lineterminator="\n"), csv.writer(cf, delimiter="\t", lineterminator="\n")
        if not args.append:
            mw.writerow(["source1_entity_id", "matched_entity_ids"])
            cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        for i, r in enumerate(rows(s1)):
            if i < args.start_row: continue
            if args.end_row is not None and i >= args.end_row: break
            c = candidates(r, exact, postings, args.candidate_limit)
            scored = sorted(c)
            pred = sorted(rid for rid in scored if rid in records and accept(r, records[rid], thresholds))
            mw.writerow([r["entity_id"], ",".join(pred)])
            cw.writerow([r["entity_id"], ",".join(scored)])
            if (i + 1) % 50000 == 0: print(f"Processed row {i + 1:,} in {time.time()-t0:.1f}s", flush=True)
    print("Wrote", match_path)
    print("Wrote", cand_path)

if __name__ == "__main__":
    main()
