# INTEGRATION + VERIFICATION REPORT — split layer, post-repair-sprint

All commands below were run with `d:\Projects\skill-extraction\.venv-ml\Scripts\python.exe` (absolute path, never `run.ps1`), cwd `d:\Projects\skill-extraction`. **No git commit was made.** No secrets, keys or personal details were written anywhere. No network, no new dependencies.

---

## HEADLINE

* **Both FATAL findings are dead, re-verified by re-running the adversaries' own reproductions.** F01: 142 → **0** straddling template pairs out of an independently recounted 13,430. F02: the 4-sentence case goes from **F1 1.0 over n=1** to **F1 0.40 over n=4**.
* **All four real splits are writable** (grouped_random PRIMARY hits exactly 720/240/240 with 0 exclusions) and **17/17 invariants pass on every one of them**.
* **Every test file under `tests/` now passes**, except two whose failures are a missing optional dependency and a concurrent edit by another session (details in §4). The split layer is **197/197** across its seven suites.
* `split.py` survived the two concurrent editors **undamaged** (§2).
* **5 of 6 agents' reports checked out.** One claim did not: the textsig/split_assert/split-logic agents all reported 18 failures in `tests/test_split.py` caused by `synth_fixture` not drawing pilot/practice. By the time I took the repo that fixture had already been fixed and only **2** tests failed, for two different reasons (§4).

---

## 1. CROSS-FILE REQUESTS — reconciled

### Applied (14)

| # | From | Change | File |
|---|---|---|---|
| 1 | textsig | `shingle_max_chars: 2500` was a **lie** after the window removal → replaced the four restated fields with `**textsig.signature_params()` | `gold_standard/build_pool.py` |
| 2 | textsig | sidecar now carries `signature_cache: textsig.cache_fingerprint({pid: shingle_sha1_16})`, so `groups._try_sidecar` can accept it | `gold_standard/build_pool.py` |
| 3 | textsig | `SHINGLE_MAX_CHARS` re-labelled as a **ladder-parity** constant with a pointer to `textsig.LEAKAGE_WINDOW_CHARS` | `gold_standard/__init__.py` |
| 4 | textsig | `shingles_manifest.json` added to the fail-closed input gate | `gold_standard/split.py` |
| 5 | textsig | `load_shingles(..., require_ids=sorted(posting ids))`, and `groups.ShingleCacheError` re-raised as `SplitRefusal` → one refusal taxonomy | `gold_standard/split.py` |
| 6 | seal | public aliases `prf` / `greedy_match` / `percentile_ci` + a "do not change the semantics, these are load-bearing for every published test number" note | `gold_standard/iaa.py` |
| 7 | seal | `train_sha256` / `dev_sha256` / `test_blind_sha256` written into `SEAL.json` **and** `manifest["parts"]` (contract §8, never implemented) | `gold_standard/split.py` |
| 8 | seal | `seal._sealed_sentence_ids` now verifies `test_blind_sha256` **first** (the TEXT, not just the id list), skipped only for splits sealed by an older writer | `gold_standard/seal.py` |
| 9 | seal | the dead `_carry_superseded` branch pointed at `out_dir.parent`, where `seal.py` actually writes `test_open_log.jsonl` | `gold_standard/split.py` |
| 10 | keys | `_equality_edges` now filters on `posting_keys.is_missing_hash`, not `!= ""` (an object-dtype `"nan"` would have collapsed every such posting into ONE employer group) | `gold_standard/groups.py` |
| 11 | keys | `employer_loose_status` carried into `leakage_groups.parquet` + `employer_loose_status_counts` and `n_postings_without_a_loose_hash` in the manifest | `gold_standard/groups.py` |
| 12 | split_assert | `raw_batches` entries gained `rows` (contract §8 / I14) — I14 now reports `raw_batch_rows_recorded: true` | `gold_standard/split.py` |
| 13 | split_assert | contract I13's promised manifest hash implemented: `manifest_sha256_excluding_timestamps`, canonical JSON with `identity.created_utc` + `group_layer.created_utc` removed | `gold_standard/split.py` |
| 14 | split_assert / split-logic | `quarantine_sets` is now **fail-closed** (`required=True`) instead of returning three empty id sets for an absent `pilot_100.parquet`; `pilot_100.parquet` and `practice_10.parquet` added to the input gate | `gold_standard/split.py` |

### Rejected / redirected (4)

* **textsig #5 — "have `synth_fixture` draw pilot/practice".** Rejected in that form: `build_fixture` writes the raw corpus and pool, and runs *before* `posting_keys`/`groups`/`sample` exist, so it cannot draw a group-aware quarantine set. The correct place is the test fixtures, where it already is (`tests/test_split.py::_fixture`, `tests/test_split_manifest.py::_fixture`, `tests/test_split_logic.py`). **Residual:** the 25-pilot/10-practice recipe is now duplicated in three test files; a shared `synth_fixture.draw_quarantine(gs_dir)` helper would be worth adding later. Not done, to avoid churn in files three agents had just rewritten.
* **split-logic #1 — "scope `i10_quarantine` to the assigned parts".** Already implemented by the split_assert agent before I arrived (`split_assert.py:767`), and the two agents did **not** conflict. The remaining failure was in the *test*, not the invariant (§4).
* **split-logic #3 / split-manifest #2,#3 — "surface `purity`, `exact_capacity`, the pool-scope counters; keep the I16 marginal basis".** No action needed: verified already flowing (`cross_country.purity.share_test_from_holdout_country = 0.8615` on the real SG sibling, `holdout_group_scope: "pool"`, `regime_capacity.exact_capacity`, I16 `problems: []` on all four real splits).
* **split-manifest #3 — pool-scope `n_employers_spanning_boundary`.** Not added. It needs a join inside the cross-country branch and the published sample-scope figure already states its scope (`n_employers_spanning_boundary_scope`); adding an unvalidated second number during integration was the wrong trade. Reported as open.

### The two flagged incompatibility risks — both checked, both clean

* **Shingle-cache fingerprint (textsig writing it while split_assert asserted against it).** Verified byte-identical on a seeded 50-posting cache: `textsig.cache_content_fingerprint == split_assert.shingle_cache_fingerprint == d5a87dbfbabf6e57…`. On all four real splits I7 reports `cache_fingerprint_status: "verified"`, `measures_full_text: True`, `signature_window_chars: None`, `n_postings_without_signature: 0`. No drift.
* **Empty-vs-degenerate employer hash (posting_keys changing semantics while groups.py was owned by someone else).** The two layers agreed by accident (`!= ""` happened to match every hash `posting_keys` writes); they now share one predicate (`is_missing_hash`). Policy is **DISTINCT, never collapsed**, asserted in `posting_keys_manifest.empty_key_policy.choice = "distinct"` and enforced in `groups._equality_edges`. Real-corpus effect: 419 pool postings carry no loose hash and 378 groups are employer-unverifiable, each its own group.

---

## 2. `gold_standard/split.py` AFTER TWO CONCURRENT EDITORS — undamaged

* `ast.parse` clean; 44 top-level defs, **0 duplicates**, 0 duplicated class methods.
* Imports clean (`from gold_standard import split`), and the file runs end to end four times on the real corpus (§5).
* The seam between the two editors is coherent: the allocator branch calls `_exact_capacity(...)` and stores `regime_meta["exact_capacity"]`; the manifest block ~110 lines later reads `targets` from `entry["body"]["targets"]`, **recounts** `achieved` from `frames[...]` (refusing on a recount mismatch), and reports the allocator's object separately as `regime_capacity.capacity`. No half-applied edit, no contradictory logic, no orphaned variable.
* Only real find: the merged file had **no** `pyarrow.parquet` import, which my `raw_batches.rows` change needs — added.

---

## 3. REAL-CORPUS REBUILD (`DATA/gold_standard`), correct order

```
.venv-ml\Scripts\python.exe -m gold_standard.posting_keys     95 s
.venv-ml\Scripts\python.exe -m gold_standard.groups          113 s
```

Pre-rebuild artifacts backed up to `D:\Temp\claude\ab89b9006fdb02ce\a440564a-58c1-410b-a15d-517b6bae322d\scratchpad\integ\pre_rebuild\`.

**posting_keys:** 85,740 eligible, 18,667 in pool. In-pool loose-hash status `ok 18,102 / short_residual 146 / residual_empty 2 / company_missing 417` — exactly the keys agent's prediction. `n_loose_disabled_by_short_residual: 0` (the named regression field).

**Group structure (asked-for numbers):**

| | shipped (LSH @2,500 chars, old keys) | **rebuilt (exact @full text, fixed keys)** |
|---|---|---|
| groups | 8,487 | **8,455** (−32) |
| singleton groups | 6,185 | **6,149** (−36) |
| largest group | 274 = 1.47% of 18,667 | **276 = 1.48% of 18,667** |
| largest group, Indonesia | 153/5,667 = **2.70%** | **153/5,667 = 2.70%** (unchanged) |
| loose-merge count | 102 | **106** |
| employer-loose edges | 9,796 | **9,936** (+140) |
| employer-exact edges | 9,779 | 9,779 |
| template edges at τ=0.30 | 4,799 | **13,430** |
| template edges straddling a group | (not measured) | **0** |
| adversary's cross-group full-text pairs now grouped | — | **142 / 142** |

Per-stratum largest group: asean 241/5,000 = **4.82%**, indonesia 153/5,667 = 2.70%, oecd 133/8,000 = 1.66%.

**⚠ FLAG, loudly:** ASEAN sits **9 postings** below `groups.MAX_COMPONENT_SHARE_OF_STRATUM = 0.05`, at which point `gold_standard.groups` refuses to write at all. The next corpus batch can trip it. The guard has still never fired on real data (it fires in `tests/test_split_groups.py`).

Final digests: `posting_keys.parquet b0fc3dba1ee49e68`, `shingles.parquet f543887f35e48b6d`, `template_pairs.parquet 24b4affc390ef554`, `leakage_groups.parquet 8ceed08ba78a61b7`.

**On the "101":** the review's 101 was enumerated with MinHashLSH (recall 0.8393, zero false positives), so it is a strict subset of the exact answer. My independent exact recount of the same property over the **shipped** map gives **142**; all 142 — therefore all 101 — are now co-grouped, and **0 of 13,430** full-text pairs straddle.

**Also verified: the `build_pool → groups` fingerprint contract I just wired actually works.** `build_pool` has not been re-run (it would invalidate the frozen `pool_sha1_16 = 9061b3b07b7d971c`), so I reconstructed its sidecar exactly as the patched code writes it and ran `build_groups` against it in a scratch dir: **sidecar ACCEPTED**, 256-posting raw audit with 0 mismatches, build 39 s instead of 113 s, and `n_groups / n_singletons / template edges / loose merges / shingles_fingerprint` all **identical** to the `DATA` build. Before this patch the sidecar was unconditionally refused.

---

## 4. EVERY TEST FILE — real counts

```
.venv-ml\Scripts\python.exe tests\<file>.py
```

| file | result |
|---|---|
| `test_split.py` | **44/44 passed** (was 42/44 when I took it) |
| `test_split_assert.py` | **42/42 passed** |
| `test_split_groups.py` | **28/28 passed** |
| `test_split_keys.py` | **13/13 passed** |
| `test_split_logic.py` | **23/23 passed** |
| `test_split_manifest.py` | **31/31 passed** |
| `test_split_seal.py` | **16/16 passed** |
| `test_iaa.py` | **49/49 passed** |
| `test_gold_standard_pipeline.py` | **11/11 passed** |
| `test_annotation_server.py` | **7/23 passed, 16 FAILED** — every failure is `ModuleNotFoundError: No module named 'fastapi'` |
| `test_repair_sprint.py` | **exit 1 at import** as run directly; **8/8 passed** with the repo root on `sys.path` |
| `test_corpus_lineage_and_contract.py` | **10/10** · `test_evidence_first_repairs.py` **7/7** · `test_scientific_validation_pack.py` **5/5** · `test_stability_ablation_pack.py` **5/5** · `test_upstream_evidence_repair.py` **4/4** — *but see the honesty note below* |

**HONESTY NOTE, and it matters.** Those last six files are **pytest-style with no `__main__` runner** and take pytest's `tmp_path` fixture. Running `python tests/<file>.py` imports the module and **exits 0 having run zero tests** — an exit code of 0 there is not a pass, and anyone reading a green run of the whole directory is being misled about 39 test functions. I got the counts above by supplying the one fixture they use, via `D:\Temp\claude\ab89b9006fdb02ce\a440564a-58c1-410b-a15d-517b6bae322d\scratchpad\integ\work\run_pytest_style.py`. **Open item: six test files in this repo cannot be run by the project's own documented command.** I did not add runners to them — that is six files of unrelated churn mid-integration — but it should be done.

### The two tests I changed, and why (neither was weakened)

1. **`test_quarantine_and_accounting_invariants_fire` (line 640)** — the test was internally inconsistent, not the invariant. It asserted at line 629 that an empty quarantine reference must FAIL, then at line 640 handed the *same* empty set as the `practice` reference and asserted a PASS. The hardened `require_nonempty` check correctly failed it. **Fix: the test now passes a non-empty practice reference** (`{"not-in-this-frame"}`) so the scope assertion isolates what it is about, and I **added** `assert sa.i10_quarantine(df, practice, practice)["passed"]` so the clean case is still a control. The emptiness rule is still asserted, three lines up.
2. **`test_split_refuses_to_write_when_an_invariant_fails`** — a fixture omission exposed by my own change: the test copies inputs into a temp dir, and `shingles_manifest.json` / `pilot_100.parquet` / `practice_10.parquet` are **inputs now**, so the run was refused at the input gate and never reached the I4 refusal it is about. **Fix: the test copies them.** The refusal path it asserts is unchanged and still fires on `I4_employer_exact_disjoint`.

**Pre-existing, not mine, not fixed:** `test_annotation_server.py` needs `fastapi` (rule 6 forbids installing it) and `test_repair_sprint.py` needs the repo root on `sys.path`. **Also: `tests/test_annotation_server.py`, `annotation_server/*`, `gold_standard/app.py`, `fly.toml`, `requirements.txt` and `.dockerignore` were modified by ANOTHER SESSION while I worked** (that file grew from 14 to 23 tests mid-run). My annotation-server count is a snapshot of a moving target; nothing in the split layer touches it.

---

## 5. ALL THREE REGIMES END TO END, REAL CORPUS

**Exact input.** `DATA/gold_standard` holds no `main_1200.parquet`, so I built one from the **real** corpus without writing into `DATA/`: the ten real inputs (`sentence_pool.parquet` 662,248 rows, `posting_keys.*`, the rebuilt `shingles.*` / `leakage_groups.*` / `template_pairs.parquet`, `pilot_100.parquet`, `practice_10.parquet`) hard-linked into `D:\Temp\claude\ab89b9006fdb02ce\a440564a-58c1-410b-a15d-517b6bae322d\scratchpad\integ\gs`, then `main_1200` drawn **there** against the REBUILT group map with `group_cap = MAX_SENT_PER_GROUP` and the 110 quarantine sentences / 103 quarantine groups excluded → 1,200 sentences, strata **540 indonesia / 240 asean / 420 oecd**, exactly `STRATUM_QUOTA`. Registry: the real `gold_standard/split_registry.json`, `CONFIG_A` (720/240/240). Driver: `…\scratchpad\integ\work\e2e_real.py`, log `…\scratchpad\integ\e2e_real.log`.

| split | train | dev | test | excluded | 17/17 invariants | `exact_sizes_met` | max cross-part Jaccard |
|---|---|---|---|---|---|---|---|
| **`grouped_random_v1`** (PRIMARY, exact) | **720** | **240** | **240** | 0 | **pass** | `true` | 0.2000 |
| `temporal_v1` | 720 | 240 | 207 | 33 | **pass** | `null` | 0.1054 |
| `cross_country_v1/sg` | 801 | 267 | 130 | 2 | **pass** | `null` | 0.2000 |
| `cross_country_v1/us` | 771 | 257 | 162 | 10 | **pass** | `null` | 0.0945 |

All four **writable**. The shortfalls are reported, not hidden: e.g. `temporal` prints `WARNING unmet target -> test: registered target 240 achieved 207 (delta -33; regime capacity was 207, regime-determined — this regime makes no exactness claim)`. I7 on each: `cache_fingerprint_status "verified"`, `measures_full_text True`, `signature_window_chars None`, 329,636–376,800 pairs compared, 0 postings without a signature. I10: `reference_sizes pilot_100 {100,100,94} / practice_10 {10,10,9}` — non-vacuous. I14: `raw_batch_rows_recorded: true`, 25 batches with row counts. I16: `problems: []` on all four. I15 on both cross-country siblings: `group_rule_scope: "pool-wide group->country map supplied by the producer"`, `n_employers_spanning_boundary: 0`.

**Two positive controls on the real written split** (`…\scratchpad\integ\work\post_checks.py`):
* **I13 digest is reproducible**: two independent runs of `grouped_random` over identical inputs both give `manifest_sha256_excluding_timestamps = d9db18a2ad192b4507d0cc16e7b00565e85a95f584400c65671568c6b4a07dbd`, and identical part digests.
* **The newly-written `test_blind_sha256` catches a text edit under a fixed id list**: 30 gold spans materialised over 240 sealed sentences → clean score; then one sealed row's `sentence_text` edited (ids untouched) → `SealViolation: test_blind.parquet does not match the sealed file digest`. Before this change that edit was invisible.

---

## 6. THE TWO FATAL FINDINGS — re-reproduced, before/after

### F01 — the 2,500-char truncation straddle

Script `…\scratchpad\integ\work\f01_verify.py`, 16 min, **independent of `template_pairs.parquet`**: it reads the freshly rebuilt signature cache, builds its own inverted index over 12,984,349 candidate pairs and verifies each by exact Jaccard.

```
cache: 18,667 postings, 10,223,058 shingles (mean 547.7)
INDEPENDENT full-text recount: 13,430 pairs at J >= 0.30

BEFORE  SHIPPED map (LSH over 2,500-char prefixes): 142 / 13430 pairs in DIFFERENT leakage groups
AFTER   REBUILT map (exact over full cleaned text):   0 / 13430 pairs in DIFFERENT leakage groups

adversary-class cross-group list: 142 pairs, now co-grouped 142, still straddling 0
  of those, invisible at 2,500 chars (truncated J < 0.30): 126; truncated to exactly 0.0: 42
  u-4e564dfcf7df2563 / u-addb057dcdb16fc6: full_J 0.7374, truncated 0.2814 -> SAME group
  u-06bd527a264a3749 / u-0cbab9a792c7f2c6: full_J 0.4873, truncated 0.0496 -> SAME group
  u-219cc1d69f377e54 / u-a31167f29b81ecae: full_J 0.4862, truncated 0.4862 -> SAME group  (the pure LSH-recall miss)
```

### F02 — the gold-empty-sentence denominator

Script `…\scratchpad\integ\work\f02_verify.py`, scoring the adversary's exact case with the **pre-fix core kept verbatim** in `tests/test_split_seal.py::_old_score_gold_only` and with the shipped `seal.py`, on identical inputs (4 sealed sentences, 1 ACCEPT gold span on s1, predictions = the correct span plus 3 spurious spans on s2/s3/s4):

```
BEFORE  micro = {tp 1, n_gold 1, n_pred 1, precision 1.0, recall 1.0, f1 1.0}
        n_test_sentences: 1        (sealed n = 4)

AFTER   micro = {tp 1, n_gold 1, n_pred 4, precision 0.25, recall 1.0, f1 0.4}
        micro_f1_ci95 = [0.0, 0.8571] over 2000 defined resamples
        n_test_sentences: 4   n_gold_bearing: 1   n_gold_empty: 3
        n_pred_spans_on_gold_empty_sentences: 3   n_false_positive_spans: 3
```

The same denominator rule is visible on real data: the real sealed test reports `30 spans over 30 of 240 sealed sentences … 210 sealed sentences carry no gold span and are scored as empty-gold units`, `n_test_sentences 240`.

---

## 7. WHERE THE CONTRACT ITSELF IS WRONG

18 items, each naming the clause, the measured evidence and what the code does instead. Rather than only reporting them, I appended them to `d:\Projects\skill-extraction\docs\experiments\split_spec_v1.md` as **APPENDIX C — DEFECTS IN THIS CONTRACT**, explicitly marked **"NOT RATIFIED. The v1.0.0 body above is unchanged and remains the contract of record."** The v1.0.0 text is untouched; ratifying any item means issuing v1.1.0. The headline ones: §D2's 2,500-char window and its `MinHashLSH(0.25,128)` edge set (the spec's own "LSH recall is below 1.0" residual no longer describes the code); §D1's "I7 is stronger than closure"; §3.4's `< 3` branch and its ASCII-only normalisation; §D4's precondition **P2, which is unsatisfiable alongside the contract's own exclusion reasons**; §D6/I16 being a key-presence rule too weak to catch the defect it exists for; §10's silence on the scoring denominator — which is precisely how the second fatal finding shipped with every invariant green; and §6.8/§8 prescribing hardcoded corpus-specific prose inside a manifest the contract itself declares `pool_append_stable: false`.

---

## 8. STILL BROKEN / OPEN — stated plainly

1. **Six test files cannot be run by the project's documented command** (`test_corpus_lineage_and_contract`, `test_evidence_first_repairs`, `test_scientific_validation_pack`, `test_stability_ablation_pack`, `test_upstream_evidence_repair`, `test_repair_sprint`): pytest-style, no `__main__`, so `python tests/<f>.py` runs **zero** tests and exits 0. They do all pass (39/39) under my fixture shim. **This is the most dangerous thing left: a green directory run proves less than it looks like it proves.**
2. **`test_annotation_server.py` needs `fastapi`** (16/23 fail on the import). Rule 6 forbids installing it.
3. **ASEAN is 9 postings from tripping `MAX_COMPONENT_SHARE_OF_STRATUM`**, at which point `gold_standard.groups` refuses to write.
4. **`build_pool` has not been re-run**, so `DATA/gold_standard/posting_shingles.parquet` still does not exist and `groups` rebuilds from the 25 raw batches (+74 s). Correct and safe, just slower; re-running `build_pool` would invalidate the frozen `pool_sha1_16` that the pilot binding records, so it should wait for a deliberate gold-standard version bump. The fast path is verified working (§3).
5. **A single tampered sidecar row under a forged-but-self-consistent manifest** is caught only with p = 256/18,667 = 1.4%. Wholesale forgery is caught with certainty. Stated in the code and the manifest.
6. **Residuals that are policy, not bugs**: τ = 0.30 is a chosen floor; employer hashes are string hashes; a two-hop template bridge through a NON-POOL posting is grouped by neither layer; 419 pool postings carry no loose hash; a short-residual merge is conservative for leakage but wrong for any per-employer count; the external test is **not** country-pure (SG 86.2% / US ~90%) and `country_purity: "NOT claimed"` says so.
7. **Not done, reported**: pool-scope `n_employers_spanning_boundary`; a shared `synth_fixture.draw_quarantine()` to de-duplicate the pilot/practice recipe across three test fixtures; `occupation_stratified`'s unregistered 0.5 threshold; `DESIGN_EFFECT = 1.18` as a bare constant.
8. **Another session is editing this repo concurrently** (`annotation_server/*`, `gold_standard/app.py`, `tests/test_annotation_server.py`, `fly.toml`, `requirements.txt`, `.dockerignore`). Nothing it touched is in the split layer, but my annotation-server count is a snapshot.
9. **Note for the orchestrator:** this step was the integration/verification of the splitter only. The user's other two asks — **the exact human-annotation volume** and **finishing the last learning-curve compute point at 4,800** — are **not** covered by anything in this report and are still outstanding.

### Files I changed
`gold_standard/split.py`, `gold_standard/seal.py`, `gold_standard/groups.py`, `gold_standard/build_pool.py`, `gold_standard/iaa.py`, `gold_standard/__init__.py`, `tests/test_split.py`, `docs/experiments/split_spec_v1.md`. Rebuilt in place: `DATA/gold_standard/{posting_keys.parquet, posting_keys_manifest.json, shingles.parquet, shingles_manifest.json, template_pairs.parquet, leakage_groups.parquet, leakage_groups_manifest.json}`. Backups, scripts and logs under `D:\Temp\claude\ab89b9006fdb02ce\a440564a-58c1-410b-a15d-517b6bae322d\scratchpad\integ\`.
