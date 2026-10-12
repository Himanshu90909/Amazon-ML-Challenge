#!/usr/bin/env python3
"""Offline business entity resolution pipeline for the Amazon ML challenge.

Uses only the supplied TSV files: Unicode normalization, country-aware exact
indexes, rare-token blocking, and precision-oriented RapidFuzz scoring.

Example:
  python entity_resolution.py --data-dir dataset/test --output-dir output
"""
import argparse, csv, re, unicodedata, collections, os, time
from rapidfuzz.fuzz import ratio, token_set_ratio

STOP = set("the and of for inc llc ltd limited corp corporation company co pvt private llp no near road rd street st lane ln avenue ave block india united states usa".split())

def norm(s):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)

def norm_ws(s):
    """Like norm(), but keeps single spaces between tokens -- needed for token_set_ratio,
    which relies on whitespace to find token boundaries. norm() squishes everything into
    one blob, which silently breaks token_set_ratio (it degrades to near-char-ratio).
    """
    t = re.sub(r"[^\w\s]+", " ", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()

def toks(s):
    return [x for x in re.findall(r"\w+", unicodedata.normalize("NFKC", s or "").lower(), flags=re.UNICODE)
            if len(x) >= 3 and x not in STOP]

def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")

def build_indexes(src_paths, max_df=1000):
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

def candidates(r, exact, postings):
    country, out = r["country"], set()
    for field, text in (("n", r["business_name"]), ("a", r["business_address"])):
        nt = norm(text)
        if nt: out.update(exact.get((country, field, nt), ()))
        for t in set(toks(text)):
            out.update(postings.get((country, field, t), ()))
    return out

def accept(query, rec, thresholds):
    name, addr, _, _, addr_ws = rec
    ns = ratio(norm(query["business_name"]), name) / 100.0 if name else 0.0
    q_addr_ws = norm_ws(query["business_address"])
    ass = token_set_ratio(q_addr_ws, addr_ws) / 100.0 if addr_ws else 0.0
    n_hi, a_hi, both = thresholds
    return (ns >= n_hi or ass >= a_hi or
            (ns >= both and ass >= both) or
            (ns >= 0.88 and ass >= 0.70) or
            (ass >= 0.88 and ns >= 0.70))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="directory containing test_source1/2/3.tsv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-df", type=int, default=1000)
    ap.add_argument("--name-threshold", type=float, default=0.92)
    ap.add_argument("--address-threshold", type=float, default=0.94)
    ap.add_argument("--both-threshold", type=float, default=0.78)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    s1 = os.path.join(args.data_dir, "test_source1.tsv")
    sources = [("2", os.path.join(args.data_dir, "test_source2.tsv")),
               ("3", os.path.join(args.data_dir, "test_source3.tsv"))]
    print("Building country-aware indexes...")
    records, exact, postings = build_indexes(sources, args.max_df)
    print(f"Indexed {len(records):,} candidates; {len(postings):,} rare-token blocks")
    thresholds = (args.name_threshold, args.address_threshold, args.both_threshold)
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    with open(match_path, "w", encoding="utf-8", newline="") as mf, open(cand_path, "w", encoding="utf-8", newline="") as cf:
        mw, cw = csv.writer(mf, delimiter="\t", lineterminator="\n"), csv.writer(cf, delimiter="\t", lineterminator="\n")
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        for i, r in enumerate(rows(s1), 1):
            c = candidates(r, exact, postings)
            # candidate_pairs is exactly the set scored below.
            scored = sorted(c)
            pred = sorted(rid for rid in scored if accept(r, records[rid], thresholds))
            mw.writerow([r["entity_id"], ",".join(pred)])
            cw.writerow([r["entity_id"], ",".join(scored)])
            if i % 100000 == 0: print(f"Processed {i:,} Source-1 rows", flush=True)
    print("Wrote", match_path)
    print("Wrote", cand_path)

if __name__ == "__main__":
    main()
