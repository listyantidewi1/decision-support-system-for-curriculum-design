## IMPLEMENTED — all tests pass (44/44 new, 49/49 + 11/11 existing), nothing committed

### Exact commands run and their real output

```
d:\Projects\skill-extraction\.venv-ml\Scripts\python.exe tests\test_split.py
  -> 44/44 passed            (exit 0; full PASS list printed, 0 FAILED)
d:\Projects\skill-extraction\.venv-ml\Scripts\python.exe tests\test_iaa.py
  -> 49/49 passed            (regression check, unchanged)
d:\Projects\skill-extraction\.venv-ml\Scripts\python.exe tests\test_gold_standard_pipeline.py
  -> 11/11 passed            (regression check: sample.py / build_pool.py changes)
d:\Projects\skill-extraction\.venv-ml\Scripts\python.exe -m gold_standard.posting_keys --flags record_flags_20261115T164444Z.parquet
  -> 85,740 keys, 18,667 in pool; missing employer ID 290 / ASEAN 23 / OECD 963;
     1 unparseable date  (reproduces the audit exactly)
d:\Projects\skill-extraction\.venv-ml\Scripts\python.exe -m gold_standard.groups
  -> 8,487 groups, 6,185 singleton, largest 274 postings (1.47% of pool)
```
A full three-regime dry run on the synthetic fixture also ran (script in the scratchpad: `D:\Temp\claude\ab89b9006fdb02ce\a440564a-58c1-410b-a15d-517b6bae322d\scratchpad\run\dryrun.py`); the same path is covered permanently by `tests/test_split.py`, which builds the fixture, keys, groups, a capped sample and all three regimes from scratch each run.

### Files created
- `d:\Projects\skill-extraction\gold_standard\textsig.py` (88 l) — pure shingle/Jaccard/MinHash kernel, shared by build_pool and groups
- `d:\Projects\skill-extraction\gold_standard\posting_keys.py` (259 l) — sidecar `posting_keys.parquet` + manifest; loose employer hash, occupation merge, per-element dates
- `d:\Projects\skill-extraction\gold_standard\groups.py` (423 l) — union-find closure of employer-exact ∪ employer-loose ∪ template(J≥0.30); `shingles/template_pairs/leakage_groups` + manifest; refuses to write on the feasibility guards
- `d:\Projects\skill-extraction\gold_standard\split.py` (1242 l) — registry gate, apportionment, parity-greedy allocator, three regimes, §8 manifest, fail-closed write, version log + index
- `d:\Projects\skill-extraction\gold_standard\split_assert.py` (394 l) — I1–I17, pure, raises `InvariantFailure`
- `d:\Projects\skill-extraction\gold_standard\split_registry.json` (90 l) — CONFIG_A (runnable), CONFIG_A2 / CONFIG_B / ENLARGE_R1 / ENLARGE_R2 (registered, `runnable:false`), each with its own content hash
- `d:\Projects\skill-extraction\gold_standard\seal.py` (253 l) — `materialise_labels`, `score_on_test` with receipts, single-open refusal, VOID-on-tamper
- `d:\Projects\skill-extraction\gold_standard\synth_fixture.py` (254 l) — deliverable 5; `python -m gold_standard.synth_fixture --root <dir>` writes raw batches + flags + pool with planted leakage, plus `write_registry()` for fixture-sized targets
- `d:\Projects\skill-extraction\tests\test_split.py` (776 l) — standalone runner, `test_iaa.py` style

### Files modified
- `d:\Projects\skill-extraction\gold_standard\__init__.py` (+102) — the §2 constants
- `d:\Projects\skill-extraction\gold_standard\build_pool.py` (+63/-4) — raw projection widened with `employer_hash` + `query_family`, both carried into every pool row; template signature computed where the description is still in scope and written to the `posting_shingles.parquet` sidecar (not into the pool: 662k rows × a shingle list is unusable); docstring states the rebuild requirement
- `d:\Projects\skill-extraction\gold_standard\sample.py` (+106/-10) — `_cap_per_group`, `_load_group_map`, `group_cap`/`exclude_groups` on `draw_sample`, group-quarantined main draw, group stats in the manifest. Pilot/practice paths untouched (`group_cap=None` ⇒ byte-identical; pinned by a test)

Untracked artifacts written under the gitignored `DATA/gold_standard/`: `posting_keys.parquet`(+manifest), `shingles.parquet`, `template_pairs.parquet`, `leakage_groups.parquet`(+manifest).

### POOL REBUILD IS REQUIRED before the splitter runs on real data
`sentence_pool.parquet` on disk has none of the new columns. I did **not** regenerate it (per instruction, and because `pilot_100_manifest.json` pins `pool_sha1_16=9061b3b07b7d971c`). The splitter is runnable today anyway: `posting_keys.py` reconstructs the keys from the raw batches and `groups.py` recomputes shingles from them, so the real run order is
`posting_keys → groups → sample --main → split --regime {grouped_random,temporal,cross_country}`.
`main_1200` was deliberately **not drawn** — that is the real gold sample and a research decision, not a code step. Every manifest records `pool_append_stable: false` with the reason (`build_pool`'s `grp.sample(n=cap, random_state=SEED)` reshuffles pool membership when a batch is added, so a corpus refresh is a new gold-standard version, never an update).

### Real-corpus findings worth carrying into the write-up
- Largest leakage group 274/18,667 = **1.47%** of the pool — matches the audit's prediction exactly.
- **ASEAN's largest group is 239/5,000 = 4.78%, just under the 5% feasibility guard.** One more batch could trip it, and `groups.py` will then refuse to write. Flagging it now.
- Template coverage at τ=0.30: 1,062 multi-posting components over **3,000 postings (16.1%)**, against the audit probe's 14.5% — higher because this layer shingles *cleaned* text, as designed.
- **170 cleaned-text pairs at Jaccard ≥ 0.85** that the ladder's raw-text dedup could not see (`n_pairs_jaccard_ge_085_cleaned`). Residual near-duplicates inside the "already de-duplicated" eligible set.
- The loose hash merged **102** times, creating 95 groups that exist only because of it; max 3 exact employers per loose class, 996 pool postings in such classes — well inside the 2% runaway guard.
- Occupation `UNRESOLVED`: Indonesia 0%, ASEAN 47%, OECD 74% — the audit's numbers, with no keyword guessing.

### Deviations from the contract (each deliberate, none silent)
1. **Deliverable 1 vs the contract's sidecar-only rule.** The task said carry the keys through `build_pool` into the pool; the contract said sidecar and leave the pool alone. I did both: `build_pool` now writes the columns (for the next rebuild) *and* the shingle sidecar, while `posting_keys.py` is authoritative until then. No pool rebuild was run.
2. **blake2b masked to 63 bits** (`HASH_FN="blake2b-64-masked-63"`). A full 2⁶⁴ shingle hash overflows parquet's pandas round-trip (`OverflowError: int too big to convert`, hit on the first dry run). The mask is part of the hash definition, so signatures stay byte-identical across writers.
3. **Tie-break is `sha1(group_id|regime_id|part)`**, not the contract's `sha1(group_id+regime_id)`, which cannot break a tie *between parts* because it does not depend on the part.
4. **The group cap is applied globally as well as per stratum.** `draw_sample`'s rounding top-up path draws straight from `pool`; capping only inside `_take_stratum` left it as an uncapped back door, and leakage groups cross strata.
5. **A forcing conflict the contract left undefined:** an all-late (temporal) or hold-out-country group whose employer is unverifiable cannot go to test (I9) nor to train (post-cut data / country purity). It is **excluded** with an explicit reason rather than placed. Unit-tested.
6. **`holdout_rule` is a parameter** (default = the registered thresholds) so a fixture-sized corpus can exercise the regime; any deviation is recorded as `holdout_rule_overridden: true`, and a derived country is never recorded as an override.
7. **`allocation_cost`** is reported as both the greedy delta sum and an interpretable `final_weighted_abs_deviation` (the raw greedy sum is normally negative, which reads as a bug in a manifest).
8. `main()` for `--regime cross_country` builds **one sibling split per non-primary stratum** from `derive_holdouts`, so both external rows of Tabel 3.5 are served by one command.

### Not built, and why
- **`extend_split` / CONFIG_B nesting assertion.** It needs adjudicated labels and a re-draw that cannot exist yet; CONFIG_B is registered `runnable:false` and the registry gate refuses it. The nesting requirement is recorded in the registry body, not enforced in code.
- **R=20 re-split variance study, grouped-vs-random contrast arm, grouped k-fold** — out of scope per PART 2 item 9.
- **No dashboard, Streamlit page, Markdown report or figure** — PART 2 item 14. Deliverable surface is the manifests plus `tests/test_split.py`.
- **I6 is asserted at τ=0.30 only**; t050/t070 are reported (monotonicity makes asserting them uninformative), and a test proves the monotonicity holds in the built map.
- `tag_support` and `enlargement.triggered` are `null` with `labels_available:false` — they cannot be computed before adjudication, and `seal.materialise_labels` is wired as their evaluator.

### Test coverage map (every item you asked for)
group atomicity across *every* boundary for all three regimes (group/posting/employer-exact/employer-loose/template) · determinism (same seed ⇒ identical `assignment_sha256` and identical frame) · group larger than a whole split (P1 refusal) and oversize group in dev/test (I11) · exact-size accounting (P2/P3/P4/P5, every weight mix, parity correction under odd forced weight) · temporal ordering (per-stratum cut, nothing post-cut in train/dev, nothing pre-cut in test, straddler-drop unit test) · cross-country disjointness at group level plus both derived siblings · the refusal path end to end (tampered group map ⇒ `InvariantFailure` on I4 and **nothing left on disk**) · empty/degenerate input (empty allocation, all-excluded, empty temporal frame, missing inputs) · positive controls that each of I2/I3/I4/I5/I6/I7/I8/I9/I10/I11/I17 actually *fires* · registry gate (unknown, edited, un-runnable) · seal (label materialisation, AMBIGUOUS drop, single-open refusal, VOID-on-tamper, T5 static check).