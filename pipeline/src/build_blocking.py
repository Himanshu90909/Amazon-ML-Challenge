"""Build token tables and run the blocking join to produce candidate pairs.

Usage: python3 build_blocking.py <split>   where split is 'train' or 'test'
"""
import sys
import time
import os
import pandas as pd
import duckdb
from blocking import build_token_frame, compute_doc_freq, compute_stoplist, filter_stoplist, estimate_fanout

DATA_ROOT = "../../data/student_resource/dataset"
OUT_ROOT = "../work"

WORD_CAP = 50
NUMERIC_CAP = 500
MAX_ALLOWED_FANOUT = 150_000_000  # safety valve: raise caps down further if estimate exceeds this


def load_source(split, n):
    path = f"{DATA_ROOT}/{split}/{split}_source{n}.tsv"
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def main(split):
    os.makedirs(f"{OUT_ROOT}/{split}", exist_ok=True)
    t0 = time.time()
    print(f"[{split}] loading sources...", flush=True)
    s1 = load_source(split, 1)
    s2 = load_source(split, 2)
    s3 = load_source(split, 3)
    print(f"[{split}] loaded s1={len(s1)} s2={len(s2)} s3={len(s3)} in {time.time()-t0:.1f}s", flush=True)

    s1.to_parquet(f"{OUT_ROOT}/{split}/s1.parquet", index=False)
    s2.to_parquet(f"{OUT_ROOT}/{split}/s2.parquet", index=False)
    s3.to_parquet(f"{OUT_ROOT}/{split}/s3.parquet", index=False)

    t0 = time.time()
    tok1 = build_token_frame(s1)
    print(f"[{split}] tokenized s1 in {time.time()-t0:.1f}s ({len(tok1)} tokens)", flush=True)
    t0 = time.time()
    tok2 = build_token_frame(s2)
    print(f"[{split}] tokenized s2 in {time.time()-t0:.1f}s ({len(tok2)} tokens)", flush=True)
    t0 = time.time()
    tok3 = build_token_frame(s3)
    print(f"[{split}] tokenized s3 in {time.time()-t0:.1f}s ({len(tok3)} tokens)", flush=True)

    freq1 = compute_doc_freq(tok1)
    freq2 = compute_doc_freq(tok2)
    freq3 = compute_doc_freq(tok3)

    word_cap, numeric_cap = WORD_CAP, NUMERIC_CAP
    while True:
        stop2 = compute_stoplist(freq2, word_cap, numeric_cap)
        stop3 = compute_stoplist(freq3, word_cap, numeric_cap)
        f1_23 = freq1[~freq1.index.isin(stop2 | stop3)]
        f2f = freq2[~freq2.index.isin(stop2)]
        f3f = freq3[~freq3.index.isin(stop3)]
        est2 = estimate_fanout(f1_23, f2f)
        est3 = estimate_fanout(f1_23, f3f)
        print(f"[{split}] caps word={word_cap} num={numeric_cap} -> estimated fanout s2={est2:.0f} s3={est3:.0f}", flush=True)
        if est2 + est3 <= MAX_ALLOWED_FANOUT or word_cap <= 5:
            break
        word_cap = max(5, word_cap // 2)
        numeric_cap = max(50, numeric_cap // 2)

    stoplist = stop2 | stop3
    tok1f = filter_stoplist(tok1, stoplist)
    tok2f = filter_stoplist(tok2, stoplist)
    tok3f = filter_stoplist(tok3, stoplist)
    print(f"[{split}] after stoplist filter: tok1={len(tok1f)} tok2={len(tok2f)} tok3={len(tok3f)} stoplist_size={len(stoplist)}", flush=True)

    tok1f.to_parquet(f"{OUT_ROOT}/{split}/tok1.parquet", index=False)
    tok2f.to_parquet(f"{OUT_ROOT}/{split}/tok2.parquet", index=False)
    tok3f.to_parquet(f"{OUT_ROOT}/{split}/tok3.parquet", index=False)

    con = duckdb.connect()
    con.execute("SET memory_limit='250GB'")

    for other, name in [(2, "s2"), (3, "s3")]:
        t0 = time.time()
        con.execute(f"""
            COPY (
                SELECT DISTINCT t1.entity_id AS source1_entity_id, t2.entity_id AS matched_id
                FROM read_parquet('{OUT_ROOT}/{split}/tok1.parquet') t1
                JOIN read_parquet('{OUT_ROOT}/{split}/tok{other}.parquet') t2
                  ON t1.token = t2.token AND t1.country = t2.country
            ) TO '{OUT_ROOT}/{split}/candidates_{name}.parquet' (FORMAT PARQUET)
        """)
        n_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_ROOT}/{split}/candidates_{name}.parquet')").fetchone()[0]
        print(f"[{split}] candidates vs {name}: {n_pairs} pairs in {time.time()-t0:.1f}s", flush=True)

    con.close()
    print(f"[{split}] BLOCKING DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
