"""Build V2 composite-key blocking and run the join to produce candidate pairs.

Usage: python3 build_blocking2.py <split>   where split is 'train' or 'test'
"""
import sys
import time
import os
import pandas as pd
import duckdb
from blocking2 import build_key_frame

DATA_ROOT = "../../data/student_resource/dataset"
OUT_ROOT = "../work"

# safety cap on any single composite key's frequency (should almost never trigger --
# composite keys are already rare -- but guards against pathological repeats)
KEY_CAP = 3000


def load_source(split, n):
    path = f"{DATA_ROOT}/{split}/{split}_source{n}.tsv"
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def main(split):
    os.makedirs(f"{OUT_ROOT}/{split}", exist_ok=True)
    t0 = time.time()

    for n in (1, 2, 3):
        pq_path = f"{OUT_ROOT}/{split}/s{n}.parquet"
        if os.path.exists(pq_path):
            print(f"[{split}] s{n}.parquet already exists, skipping reload", flush=True)
            continue
        df = load_source(split, n)
        df.to_parquet(pq_path, index=False)
        print(f"[{split}] loaded+saved s{n}: {len(df)} rows in {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()

    s1 = pd.read_parquet(f"{OUT_ROOT}/{split}/s1.parquet")
    s2 = pd.read_parquet(f"{OUT_ROOT}/{split}/s2.parquet")
    s3 = pd.read_parquet(f"{OUT_ROOT}/{split}/s3.parquet")

    t0 = time.time()
    key1 = build_key_frame(s1)
    print(f"[{split}] built keys for s1 in {time.time()-t0:.1f}s ({len(key1)} keys)", flush=True)
    t0 = time.time()
    key2 = build_key_frame(s2)
    print(f"[{split}] built keys for s2 in {time.time()-t0:.1f}s ({len(key2)} keys)", flush=True)
    t0 = time.time()
    key3 = build_key_frame(s3)
    print(f"[{split}] built keys for s3 in {time.time()-t0:.1f}s ({len(key3)} keys)", flush=True)

    for n, kf in [(1, key1), (2, key2), (3, key3)]:
        freq = kf.groupby("token")["entity_id"].nunique()
        stop = set(freq[freq > KEY_CAP].index)
        if stop:
            kf2 = kf[~kf["token"].isin(stop)]
            print(f"[{split}] key{n}: dropped {len(stop)} over-cap keys ({len(kf)}->{len(kf2)} rows)", flush=True)
        else:
            kf2 = kf
        kf2.to_parquet(f"{OUT_ROOT}/{split}/key{n}.parquet", index=False)

    con = duckdb.connect()
    con.execute("SET memory_limit='250GB'")
    for other, name in [(2, "s2"), (3, "s3")]:
        t0 = time.time()
        con.execute(f"""
            COPY (
                SELECT DISTINCT t1.entity_id AS source1_entity_id, t2.entity_id AS matched_id
                FROM read_parquet('{OUT_ROOT}/{split}/key1.parquet') t1
                JOIN read_parquet('{OUT_ROOT}/{split}/key{other}.parquet') t2
                  ON t1.token = t2.token AND t1.country = t2.country
            ) TO '{OUT_ROOT}/{split}/candidates2_{name}.parquet' (FORMAT PARQUET)
        """)
        n_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_ROOT}/{split}/candidates2_{name}.parquet')").fetchone()[0]
        print(f"[{split}] candidates2 vs {name}: {n_pairs} pairs in {time.time()-t0:.1f}s", flush=True)
    con.close()
    print(f"[{split}] BLOCKING2 DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
