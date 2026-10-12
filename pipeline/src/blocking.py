"""Blocking / candidate generation.

Token-based blocking over BOTH business_name and business_address, partitioned by exact
country match. We block on both fields because some records have the name transliterated
into a different script or replaced by a DBA/trade name while the address stays in Latin
script and matches closely — name-only blocking would drop those pairs before the
matching model ever sees them.

Word tokens and numeric tokens are capped SEPARATELY: word tokens are capped hard (a
common English/Hindi address word carries little discriminating power and causes a
combinatorial join explosion), while numeric tokens (street/unit numbers) are capped much
looser since they are short, precise, and inherently rare.
"""
import pandas as pd
from normalize import tokenize, numeric_tokens


def build_token_frame(df: pd.DataFrame, min_token_len: int = 3) -> pd.DataFrame:
    records = []
    ids = df["entity_id"].values
    names = df["business_name"].values
    addrs = df["business_address"].values
    countries = df["country"].values

    for eid, name, addr, country in zip(ids, names, addrs, countries):
        toks = set()
        for t in tokenize(name, drop_legal_stopwords=True, min_len=min_token_len):
            toks.add(t)
        for t in tokenize(addr, drop_legal_stopwords=True, min_len=min_token_len):
            toks.add(t)
        for n in numeric_tokens(name, min_digits=2):
            toks.add("num_" + n)
        for n in numeric_tokens(addr, min_digits=2):
            toks.add("num_" + n)
        for t in toks:
            records.append((eid, t, country))

    return pd.DataFrame(records, columns=["entity_id", "token", "country"])


def compute_doc_freq(token_df: pd.DataFrame) -> pd.Series:
    return token_df.groupby("token")["entity_id"].nunique()


def compute_stoplist(doc_freq: pd.Series, word_cap: int = 50, numeric_cap: int = 500):
    is_num = doc_freq.index.str.startswith("num_")
    drop_word = doc_freq[(~is_num) & (doc_freq > word_cap)].index
    drop_num = doc_freq[is_num & (doc_freq > numeric_cap)].index
    return set(drop_word) | set(drop_num)


def filter_stoplist(token_df: pd.DataFrame, stoplist: set) -> pd.DataFrame:
    return token_df[~token_df["token"].isin(stoplist)]


def estimate_fanout(freq_s1: pd.Series, freq_other: pd.Series) -> float:
    """Upper-bound estimate of join pair count (with duplicates across shared tokens)."""
    joined = pd.concat([freq_s1.rename("f1"), freq_other.rename("f2")], axis=1).dropna()
    return float((joined["f1"] * joined["f2"]).sum())
