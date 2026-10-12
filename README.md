# Amazon ML Challenge: Business Entity Resolution

Solution code for the 72-hour Amazon **Business Entity Resolution** hackathon. The task: given millions of business records (name, address, country), cluster the ones that refer to the same real-world entity and output `matching_results.tsv`.

## Approach
1. **Normalization** (`pipeline/src/normalize.py`): unicode/case folding, legal-suffix stopword removal, token and numeric-token extraction. Spaces in addresses are preserved (over-normalizing hurt `token_set_ratio`).
2. **Blocking** (`pipeline/src/blocking*.py`, `build_blocking*.py`): DuckDB-based candidate generation partitioned by country.
   - v1: token blocking over name + address with separate caps for word and numeric tokens.
   - v2: composite keys (word+number, word pairs) to keep recall high without join explosion.
3. **Matching** (`baseline/user_script/entity_resolution.py`): fuzzy scoring (RapidFuzz) on blocked pairs plus connected-components clustering.
4. **Recovery and validation** (`baseline/user_script/`): singleton recovery passes, exact-match coverage checks, blocking sweeps and train-set validation against ground truth.

## Layout
```
baseline/user_script/   main matching script + analysis/validation helpers
baseline/manus_script/  alternate baseline implementation
pipeline/src/           normalization + blocking pipeline
logs/                   run logs (blocking sweeps, coverage, validation)
```

## Notes
- Datasets and output TSVs are not committed (hundreds of MB to GBs); see `.gitignore`.
- Target: improve accuracy to ~0.98 and reach top 50 on the leaderboard.
