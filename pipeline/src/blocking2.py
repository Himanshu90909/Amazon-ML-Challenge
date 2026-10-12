"""V2 blocking: composite keys (word+number, and word-pairs among the longest tokens)
instead of single-token blocking. Single common tokens ("care", "leesburg", "601") were
causing either explosive joins or (once capped) heavy recall loss, because ANY individual
token above a modest frequency had to be dropped. Composite 2-part keys are combinatorially
much rarer, so we get good recall WITHOUT needing an aggressive per-token frequency cap.
"""
import itertools
import pandas as pd
from normalize import tokenize, numeric_tokens

MAX_WORDS = 8
MAX_NUMS = 4
TOP_N_FOR_PAIRS = 4


def build_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    ids = df["entity_id"].values
    names = df["business_name"].values
    addrs = df["business_address"].values
    countries = df["country"].values

    for eid, name, addr, country in zip(ids, names, addrs, countries):
        words = set()
        for t in tokenize(name, drop_legal_stopwords=True, min_len=3):
            if not t.isdigit():
                words.add(t)
        for t in tokenize(addr, drop_legal_stopwords=True, min_len=3):
            if not t.isdigit():
                words.add(t)
        nums = set()
        for n in numeric_tokens(name, min_digits=2):
            nums.add(n)
        for n in numeric_tokens(addr, min_digits=2):
            nums.add(n)

        # keep the longest words (more specific / rarer) and cap counts
        words_l = sorted(words, key=len, reverse=True)[:MAX_WORDS]
        nums_l = sorted(nums)[:MAX_NUMS]

        keys = set()
        for w in words_l:
            for n in nums_l:
                keys.add("wn|" + w + "|" + n)

        top_for_pairs = words_l[:TOP_N_FOR_PAIRS]
        for a, b in itertools.combinations(sorted(top_for_pairs), 2):
            keys.add("ww|" + a + "|" + b)
        if len(words_l) == 1:
            keys.add("w1|" + words_l[0])

        for k in keys:
            records.append((eid, k, country))

    return pd.DataFrame(records, columns=["entity_id", "token", "country"])
