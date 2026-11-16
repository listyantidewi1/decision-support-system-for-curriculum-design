# GROUPED SPLIT SPECIFICATION v1.0.0 — single implementation contract

**Spine taken: Design A** (correctness-first), because the one mandatory clause in the proposal (PROP:1192 / PROP:1436 `"tidak boleh"`) is a *prohibition*, and A's architecture reduces it to an O(n) set-disjointness assertion that runs before any byte is written. **Grafted from Design B:** the pre-registration registry, the loose employer hash, the direct (non-union-find) pairwise re-derivation, the cell-coarsening ladder, the achieved-vs-target reporting invariant, the blind/labels seal with receipts, the derived temporal quantile, the straddler-to-early rule, the Indeed-ID-matched temporal control, the rule-based hold-out country choice, and the whole power/support arithmetic including `MIN_TEST_SUPPORT_SPANS_PER_TAG = 120`.

Everything below was checked against the code and data. Verified facts used:
`gold_standard/build_pool.py:~200` parses dates **per element** already (`pd.to_datetime(rec.date_posted, errors="coerce", utc=True)` inside the row loop) — no fix needed there. `build_pool` projects raw to `[source_record_id, batch_id, title, description, date_posted]`, and `governance/ladder.LOAD_COLUMNS` **already loads** `employer_hash` and `query_family` but **not** `company`. The pool retains **no** description text (confirmed from the 20 written columns). `acquisition/schema.py:21` `EMPLOYER_HASH_SALT = "ojp-rpl-2026"` is a plain in-code constant and `_sha1_16` is sha1→16 hex, so a second hash is reproducible without secrets. `governance/configs/tabel_3_9_v1.json` `duplication.near_dup = {num_perm:128, threshold:0.85, shingle_words:5, max_chars:2500}`. `gold_standard/iaa.py:474 CI_WIDTH_CLAIMABLE = 0.20`. `sample.py:40 MAX_PER_POSTING = 2`, `STRATUM_QUOTA = {indonesia .45, asean .20, oecd .35}`, `SEED = 42`, `MAIN_N = 1200`. `DATA/gold_standard/` holds `pilot_100.*` and `practice_10.*` but **no `main_1200.*`** — the main draw is still changeable. `sentence_pool.parquet` = 662,248 rows. `pilot_100_manifest.json` pins `pool_sha1_16 = 9061b3b07b7d971c`. `datasketch 2.0.0` present. Raw batches (25) carry `company`, `employer_hash`, `query_family`, `description`.

---

## PART 0 — THE EIGHT DECISIONS

### D1. Is `group_id` the transitive closure of (same employer) OR (same template)?

**Yes — transitive closure of the union of three relations, over the 18,667 pool postings only, not the 85,740 eligible.**

Relations: (a) equal non-empty `employer_hash`; (b) equal non-empty `employer_hash_loose`; (c) exact-Jaccard-verified shingle similarity ≥ 0.30.

Numbers that justify the closure: at τ=0.30 the largest employer∪template component is **274 of 18,667 pool postings (1.5%)**, and **148 of 5,667 (2.6%)** inside Indonesia — both far below the 20% test share, so closure cannot make the split infeasible. 61% of corpus employers are singletons (15,535/25,442), so the group population stays granular. Transitivity is required and not optional: 781 employers span more than one stratum and 2,264 span more than one country, so a per-stratum or per-country grouping would leak by construction.

**Both designs got the node-set question wrong, in opposite directions.** Design A asks for global closure over all 85,740 eligible postings and justifies it with those 781/2,264 cross-boundary employer counts — but those numbers argue for grouping *across strata*, which is a property of the relation, not of the node set. `employer_hash` equality is an equivalence relation, so restricting its nodes to the pool yields **exactly the same partition of pool postings**; global scope buys literally nothing for relations (a) and (b). The only gain is two-hop template bridges through postings that can never be sampled, and it costs ~16 min of MinHash plus loading 85,740 full descriptions. Design B scopes to the pool but then has no answer for those two-hop bridges. **Resolution: pool scope, and the residual two-hop risk is closed by invariant I7** — a direct all-pairs max-Jaccard check between parts, re-derived from the shingle cache rather than from the union-find. If two cross-part postings are related only through a non-pool third posting, I7 measures their actual similarity and asserts it is below τ, which is the quantity that matters. That is a stronger guarantee than A's closure at 1/50th the cost.

### D2. Template similarity: how, and at what threshold — given the pool has no description text

Description text lives **only** in `DATA/corpus_raw/batch-*/records.parquet`. `gold_standard/groups.py` therefore reads those 25 files directly with its own projection (it must **not** widen `governance/ladder.LOAD_COLUMNS`) and persists a **shingle cache**, so no later step re-reads the raw layer.

Pipeline, per pool posting: `build_pool._html_to_text(description)` → `.lower().split()` → first 2,500 **characters** of the raw string are taken *before* splitting (matching `governance/ladder.minhash_near_dups`) → 5-word shingles → each shingle hashed with `hashlib.blake2b(s.encode("utf-8"), digest_size=8)` to a uint64 → stored as a sorted unique uint64 list.

Cleaned text, not raw, is a deliberate deviation from the ladder: it makes the same template posted through two sites with different HTML wrappers match. Consequence to measure and report: the ladder's 0.85 dedup ran on **raw** text, so cleaned-text pairs at ≥0.85 can exist in the eligible set even though 0 eligible rows carry `near_dup_minhash`. Emit `n_pairs_jaccard_ge_085_cleaned` — residual near-duplicates the ladder's raw-text pass could not see.

Candidate generation: `MinHashLSH(threshold=0.25, num_perm=128)`. **Insert all 18,667 first, then query all** — the ladder's incremental insert-or-flag loop produces an order-dependent flag, which is unusable as an edge set. Every candidate pair is then verified by **exact Jaccard on the uint64 shingle sets**, so the edge set has zero false positives and is independent of insertion order. Building the LSH below the clustering threshold raises recall near the cut.

**`TAU_BUILD = 0.30`; assert at 0.30, 0.50 and 0.70.** Design A wins this on a monotonicity argument I checked and accept: edges at higher τ are a subset of edges at lower τ, so components at 0.30 are supersets of those at 0.50 and 0.70, and a split that is leak-free at 0.30 is leak-free at every higher threshold *for free*. Building at 0.50 (Design B) leaves a reviewer free to ask about 0.30 and get no answer. **Design B's stated rationale for 0.50 is contradicted by the data B itself cites**: B claims 0.50 keeps the largest employer+template group at 1.5% of the pool, but the measured table shows 1.5% at 0.70, 1.5% at 0.50 and 1.5% at 0.30 (2.6% within Indonesia) — 0.30 costs nothing on the metric B used to choose 0.50. Cost of 0.30: multi-posting components cover 14.5% of pool postings rather than 1.8%, i.e. more constraint on the allocator, which the measured component sizes show is affordable.

Honest claim wording, fixed in the manifest: *"no employer (exact or loose hash) and no template cluster at cleaned-text Jaccard ≥ 0.30 straddles any two parts, and the maximum directly measured cross-part Jaccard is X"* — never "no template straddles".

### D3. Exact sizes and ratio

**720 / 240 / 240 stays.** PROP:1431 states it verbatim and the proposal is the pre-registration. The learning-curve finding is recorded as a **separate, separately-registered enlargement rule**, not as a reason to move the primary ratio.

Design B's arithmetic for 600/240/360 is correct — I re-derived it: local slope (0.468−0.424)/720 = 6.1e-5 F1/sentence, so moving 120 sentences out of train costs 0.0073 F1, which is 4× inside the 0.030 seed spread, while the test CI at F1=0.5 narrows from 0.109 to 0.089. It is the better design *by measurement* and it is still rejected as primary, because editing a verbatim pre-registered number for an unmeasurable gain is a worse viva liability than a slightly wide CI. It is registered as `CONFIG_A2` with its arithmetic and timestamp, un-run, which converts the vulnerability into evidence that the optimisation was computed before the data was opened and deliberately not taken.

**Design B's disproportionate test allocation (Indonesia 0.55/0.15/0.30) is rejected.** It contradicts PROP:1433's natural-distribution clause in the raw allocation, and B's own numbers show it buys 108→132 Indonesian test sentences, CI width 0.162→0.147. A 0.015 gain is not worth a deviation. Test keeps the drawn sample's stratum shares (`STRATUM_QUOTA`, giving 108/48/84), and the pooled figure is reported both unweighted and stratum-weighted.

**Registered enlargement rule** (`ENLARGE_R1`, proposal-origin trigger; `ENLARGE_R2`, code-origin allocation):
- R1 trigger: any of SH/SS/KN/TT/WB with **projected test span support < `MIN_TEST_SUPPORT_SPANS_PER_TAG = 120`**, projected from the adjudicated **train+dev** gold only, test never opened. 120 is derived, not invented: `iaa.CI_WIDTH_CLAIMABLE = 0.20` ⇒ N ≥ 4·1.96²·0.25/0.04 = 96.04 spans at worst-case p=0.5, × DEFF 1.18 = 113.3 → 120. Label it `origin: "code-origin; no numeric support threshold exists in the proposal or in Paket B v3, which assigns it to the researcher (B3:32)"`.
- R2 allocation: additional sentences split 60/20/20, **total gold capped at 2,400 (1,440/480/480)**, because the measured curve gives +0.045 F1 from 720→1,440 and +0.0035 from 1,440→2,880, the latter inside the 0.030 seed spread. Enlargement is **nested**: Config B's test is a strict superset of Config A's test (`extend_split` asserts by row-level sha that no base sentence changed part), so an enlargement can only add evidence and can never re-draw a luckier test set.

### D4. How exact counts coexist with group atomicity

**By a parity proof, not by an optimiser.** Design A's subset-sum DP and Design B's singleton-reserve heuristic are both replaced by a guarantee.

Precondition: `MAX_SENT_PER_GROUP = 2` is enforced **at draw time** in `sample.py` for the main draw only (A's idea; it extends the existing `MAX_PER_POSTING = 2` rationale one level up, and it *reduces* within-split clustering, which lowers DEFF and narrows the CI — B's own statistical framework argues for it). Every group then weighs 1 or 2 sentences.

Theorem. Let targets be `(720, 240, 240)`, all even; let every group weight be ≤ 2; let forced assignments be applied first and parity corrected. Place all weight-2 groups before any weight-1 group. Then every part's remaining capacity is even at all times during the weight-2 phase (it starts even and decreases by 2), so a remaining capacity is 0 or ≥ 2. Total remaining capacity always equals the total weight of unplaced groups, so if any group is unplaced some part has capacity > 0, hence ≥ 2, hence the group is placeable. The weight-2 phase therefore always completes. The weight-1 phase then consumes unit capacity against a total capacity equal to the number of remaining weight-1 groups, so it also always completes. **Exact sizes are guaranteed, with no backtracking and no DP.**

Assert the five preconditions explicitly, and **raise** on any failure rather than shipping an off-size split:
- P1 `max(weight) ≤ 2`;
- P2 `Σ weight == 1200 == 720+240+240`;
- P3 forced weight into train ≤ 720, and no forced weight into dev or test;
- P4 after forced placement, any part whose remaining capacity is **odd** receives exactly one weight-1 group chosen by the deficit rule, restoring even parity;
- P5 `n_weight1_groups ≥ n_odd_capacity_parts`.

What happens when no assignment satisfies both: it cannot occur under P1–P5, and if a precondition fails the run **raises** with the offending group ids and weights. Atomicity is never traded for size — that ordering is fixed. The oversize guard (`MAX_GROUP_SHARE = 0.25` of the smallest part quota = 60 sentences) is vacuous at weight ≤ 2 and is kept as a cheap assertion so a future enlarged sample with a looser cap fails loudly.

The temporal and cross-country regimes make **no exactness claim**: a date cut and a country boundary cannot be bent to a target. There, achieved sizes are reported against a `MIN_TEST_SENTENCES = 100` guard and the regime is flagged `underpowered: true` rather than reshaped.

### D5. Leakage invariants — see §7 (seventeen write-time invariants I1–I17, plus four test-suite-only controls T1–T4).

### D6. Manifest schema — see §8. Convention: **every** quota and distribution is an object `{"target": …, "achieved": …, "delta": …}`, and invariant **I16** fails the write if any key under `targets.*` has no counterpart under `achieved.*`. This is Design B's single best idea and it structurally prevents the class of bug the project already hit once (a target reported as achieved).

### D7. Three regime artifacts — see §9.

### D8. What each design got wrong

**Design A**
1. Global closure over 85,740 eligible postings is justified with the 781/2,264 cross-boundary employer counts, which are about the *relation*, not the node set. For employer edges the pool-restricted partition is identical. ~16 min and multi-GB of description text bought nothing; the real residual (two-hop template bridges) is better closed by a direct pairwise check A does not have.
2. A's invariant list has **no direct cross-part pairwise similarity check**. A claims exact verification gives "zero false-positive edges", which is true of pairs but not of components: after closure a component can contain postings whose mutual Jaccard is far below τ, so "no cluster straddles" ≠ "no similar pair straddles". Without I7, A's guarantee is only as good as its own union-find.
3. A5/A13 assert template disjointness at 0.50 and 0.70 and call this "turning the implication into a tested fact that defuses the free-parameter objection". Those two assertions are **logically implied** by passing at 0.30 and can never fail independently. Harmless, but it is not evidence.
4. A declines the loose employer hash, calling name normalisation a new free parameter. A **closed, committed token list is deterministic and has no tuning knob**; A left blocker 3 unmitigated for no reason.
5. A's suggested minimum per-tag test support of "30 spans" is unjustified and **contradicts the repo's own `CI_WIDTH_CLAIMABLE = 0.20`**, which implies ~120. A's number would license per-tag claims the project's own IAA report calls unclaimable.
6. A's temporal variant quarantines straddling groups from **both** sides. Assigning them wholly to the early side and dropping only their post-cut sentences (B) keeps the same two purity claims and recovers the pre-cut sentences as training data. A's choice is strictly dominated — it discards usable data from a 1,200-sentence budget.
7. A predicts "roughly 145 SG and 138 US sentences" from stratum shares. The sample passes through `ANNOTATOR_LANGUAGES`, a 20–300 char filter, exact-text dedup, `BORDERLINE_SHARE`, and `MIN_LOCAL_LANG_SHARE['asean'] = 0.30`; the SG share of the drawn ASEAN quota is not the SG share of the ASEAN pool. Measure, never predict.
8. A's content-addressed `group_id = "G"+sha1(members)[:12]` is unreadable and changes wholesale on any corpus refresh, making version-log diffs useless. Use `"g:"+min(member posting_ids)` (equally order-independent, inspectable) and keep the member-set hash as a separate field.

**Design B**
1. B's `TEMPLATE_TAU = 0.50` rationale is contradicted by the measured table B cites (largest group 1.5% at 0.70, 0.50 **and** 0.30) — see D2.
2. B asserts "two different seeds produce DIFFERENT assignments" as a write-time invariant. It cannot be checked in a single run without running the allocator twice, and a sufficiently constrained split could legitimately be seed-invariant, making it flaky. Demoted to test T3, reformulated as "the RNG is consumed and the shuffle order differs".
3. B's `og2` keyword map from `job_title` → occupation group would label **74% of OECD and 46% of ASEAN** postings with an unvalidated, unmeasured classifier embedded in split logic. Occupation is explicitly best-effort (`sejauh memungkinkan`, PROP:1192); a wrong label is worse than an honest `UNRESOLVED`. Rejected.
4. B requires the hold-out country to be chosen and **removed from `main_1200` before it is drawn**, citing SG/US sentences in `pilot_100`. This is wrong in the respect that matters: pilot is pre-registered calibration-only and never trains anything (PROP:1430), and *both* designs already quarantine pilot/practice groups from every regime, so no ordering constraint binds. Worse, excluding SG from the main draw would leave **zero annotated SG sentences**, making the free external test impossible and forcing a 240×3-expert annotation block. Rejected: the cross-country regime is a re-partition of the same annotated 1,200.
5. B's disproportionate test allocation — rejected on B's own numbers (D3).
6. B's "repo-wide static check that no module references `test_labels.parquet`" would flag the test suite itself. Scoped to a test with an allowlist (`seal.py`, `tests/`).
7. B's invariant "direct all-pairs MinHash Jaccard between every train and test posting" (the best idea in either document) never says where the shingles come from, and the pool has no description text. Without a persisted shingle cache this invariant silently re-reads 25 raw parquets on every run.
8. B wants `employer_hash_loose` added to `acquisition/schema.add_provenance()` **and** backfilled in `build_pool` from `company`. `add_provenance` would need re-running over 25 batches; and `ladder.LOAD_COLUMNS` does not carry `company`, so `build_pool` cannot see it either. One source only: `groups.py` reads the raw parquets directly.
9. B's `MIN_FAMILY_POSTINGS = 30` folding of small families "into a coarse parent" presumes a parent taxonomy that does not exist. Small families are handled by the cell-coarsening ladder instead.

**Both designs missed two things.**
- **Design A and B both put the provenance fix inside `build_pool` and so inside `sentence_pool.parquet`.** `pilot_100_manifest.json` pins `pool_sha1_16 = 9061b3b07b7d971c`; rewriting the pool file breaks that binding and forces a re-segmentation run whose `_latest_flags()` may pick up a newer `record_flags_*.parquet` than the one the pilot was drawn against (there are five in `results/corpus_audits/`). **The fix is a sidecar**, `DATA/gold_standard/posting_keys.parquet`, which leaves `sentence_pool.parquet` byte-identical, keeps the pilot's provenance binding valid, costs no segmentation, and keeps employer hashes out of the annotator-facing worksheets (neither design noticed that adding the columns to `SAMPLE_COLUMNS` would ship employer hashes to the three annotators).
- **The pool is not append-stable.** `build_pool` caps strata with `grp.sample(n=cap, random_state=SEED)` over the whole joined frame, so adding `batch-003` reshuffles the entire ASEAN/OECD pool membership. A new batch therefore invalidates the pool, the sample **and** every split; the gold standard must be frozen against a *named* flags file plus pool sha, and a corpus refresh produces a new gold-standard version rather than an update. Record this in the manifest as `pool_append_stable: false`.

---

## PART 1 — IMPLEMENTATION CONTRACT

### §1. Files

Create:
- `gold_standard/posting_keys.py` → `DATA/gold_standard/posting_keys.parquet` + `posting_keys_manifest.json`
- `gold_standard/groups.py` → `DATA/gold_standard/shingles.parquet`, `template_pairs.parquet`, `leakage_groups.parquet`, `leakage_groups_manifest.json`
- `gold_standard/split.py` (CLI + allocator + three regimes)
- `gold_standard/split_assert.py` (pure, IO-free invariant functions)
- `gold_standard/seal.py` (label materialisation + receipts)
- `gold_standard/split_registry.json` (committed, pre-registered configs)
- `tests/test_split.py` (standalone runner, style of `tests/test_iaa.py`)

Modify:
- `gold_standard/sample.py` — add `group_cap` and `exclude_groups` parameters (§5).
- `gold_standard/__init__.py` — add the constants in §2.
- `gold_standard/build_pool.py` — widen the raw projection by exactly two names, `employer_hash` and `query_family`, for forward compatibility. **Do not re-run `build_pool` and do not rebuild `sentence_pool.parquet` before `main_1200` is drawn.** The sidecar is authoritative for splitting.

Do not touch: `governance/ladder.py`, `governance/rules.py`, `governance/configs/*`, `acquisition/schema.py`, `gold_standard/iaa.py`, `gold_standard/app.py`, `gold_standard/export_xlsx.py`, `SAMPLE_COLUMNS`.

### §2. Constants (`gold_standard/__init__.py`)

```python
SPLIT_SPEC_VERSION = "1.0.0"
# group layer
TAU_BUILD = 0.30
TAU_ASSERT = (0.30, 0.50, 0.70)
LSH_THRESHOLD = 0.25
MINHASH_NUM_PERM = 128
SHINGLE_WORDS = 5
SHINGLE_MAX_CHARS = 2500
# sampling
MAX_SENT_PER_GROUP = 2            # main draw only; pilot/practice unchanged
# allocation
SPLIT_TARGETS = {"train": 720, "dev": 240, "test": 240}
AXIS_WEIGHTS = {"stratum": 8, "country_language": 4, "site": 2,
                "quarter": 2, "occupation_group": 2, "length_band": 1}
MAX_GROUP_SHARE = 0.25
MIN_CELL_SENTENCES = 10
# regimes
TEMPORAL_QUANTILE = 0.80
MIN_TEST_SENTENCES = 100
# leakage
SENT_NEAR_DUP_JACCARD = 0.90
# power / enlargement (code-origin)
MIN_TEST_SUPPORT_SPANS_PER_TAG = 120
ENLARGE_MAX_GOLD_N = 2400
LEGAL_FORM_TOKENS = frozenset("""
pt cv tbk persero pte ltd limited llc inc incorporated corp corporation co company
gmbh bv nv sa sarl plc ab oy as sdn bhd pty srl spa kk kft
group groups holding holdings ventures
international global worldwide
indonesia singapore malaysia philippines vietnam thailand asia apac emea
""".split())
```

### §3. `posting_keys.py`

Inputs: the **pinned** flags file (CLI `--flags`, default = newest `results/corpus_audits/record_flags_*.parquet`; record its name and sha256), and the 25 raw parquets read directly with projection `[source_record_id, acquisition_timestamp, company, employer_hash, query_family, date_posted]`.

1. Eligible ids: `flags[flags.eligible.astype(str) == "True"].source_record_id`.
2. First sighting: `sort_values(["acquisition_timestamp","source_record_id"], kind="stable").drop_duplicates("source_record_id")`. Mandatory — 14,747 ids carry more than one `query_family` and 49 carry more than one non-empty `employer_hash` across sightings.
3. `employer_hash` taken as-is. Treat `""`, null, `"nan"`, `"None"`, `"NaN"` as **missing**, never as a value; `employer_known = employer_hash is present`. Expect 290 missing of 5,718 eligible Indonesian (5.07%; 14.6% of eligible Jobstreet ID), 23/12,568 ASEAN, 963/67,454 OECD.
4. `employer_hash_loose = make_employer_hash_loose(company)`:
   NFKC → casefold → strip all non-alphanumeric to spaces → collapse whitespace → tokenise → drop every token in `LEGAL_FORM_TOKENS` → rejoin in original order (**do not sort**). If the residual is empty or shorter than 3 characters, return `""` — a company whose entire name is legal-form tokens must **not** share a hash with every other such company. Otherwise `_sha1_16(f"{EMPLOYER_HASH_SALT}|{residual}")`, importing `EMPLOYER_HASH_SALT` and `_sha1_16` from `acquisition.schema`. The raw `company` string is **never** written to any output.
5. `occupation_group`: `query_family` with the committed merge map `{devops_cloud→devops, system_analyst→system_analysis}`; `legacy_english_roles` and `legacy_akkasi_roles` → `"UNRESOLVED"`. No keyword inference.
6. `date_posted_utc`: `.map(lambda v: pd.to_datetime(v, errors="coerce", utc=True))` — **per element**, matching `build_pool`. A vectorised `to_datetime(..., errors="coerce")` NaTs 2,759 Indonesian ISO-timestamp rows (48%). Expect exactly 1 unparseable row in 85,740; it is quarantined, never guessed.
7. Write `posting_keys.parquet`: `posting_id, employer_hash, employer_hash_loose, employer_known, query_family, occupation_group, date_posted_utc, in_pool`. `in_pool` is true for the 18,667 `posting_id`s present in `sentence_pool.parquet`. Manifest records the flags file + sha256, every raw batch with row count + sha256, the merge map, the token list, and the missing-hash counts by stratum and site.

### §4. `groups.py`

1. Load `posting_keys.parquet`, restrict to `in_pool` (18,667). Assert all 18,667 pool `posting_id`s join; a miss means a stale sidecar → raise.
2. Shingles: raw `description` from the batches (first sighting), `build_pool._html_to_text`, `[:SHINGLE_MAX_CHARS]`, `.lower().split()`, 5-word shingles, `blake2b(..., digest_size=8)` → uint64, sorted unique. **Never `hash()`** — `PYTHONHASHSEED` randomisation would destroy determinism. Cache to `shingles.parquet` (`posting_id`, `n_shingles`, `shingles: list<uint64>`; ~60 MB).
3. Candidates: `MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)`; **insert all, then query all**, both in sorted `posting_id` order; collect unordered pairs `(min,max)`. ~208 s measured for 18,667 postings.
4. Verify every candidate by exact Jaccard on the uint64 sets. Write `template_pairs.parquet` (`a, b, jaccard`) for all pairs with `jaccard ≥ min(TAU_ASSERT)`. Also emit the audit count `n_pairs_jaccard_ge_085_cleaned`.
5. `template_cluster_t{030,050,070}` = connected components of the pairs at each τ, computed independently.
6. Union-find over: employer-exact edges ∪ employer-loose edges ∪ pairs with `jaccard ≥ TAU_BUILD`. Nodes visited in sorted `posting_id` order, edges applied in sorted `(min,max)` order. `group_id = "g:" + min(member posting_ids)`; also store `group_members_sha1 = sha1("\n".join(sorted(members)))[:16]`.
7. Per group: `n_postings`, `group_employer_verifiable = all(employer_known)`, `edge_provenance` (which relations created each merge), `created_by_loose_only` (groups that exist only because of relation (b)).
8. **Feasibility guards** — refuse to write if any trips: largest component > 5% of any stratum's pool postings; any single loose edge merges components jointly exceeding 2% of pool postings. Measured headroom: largest component 274/18,667 = 1.5%, Indonesia 148/5,667 = 2.6%.
9. Write `leakage_groups.parquet` (`posting_id, group_id, group_members_sha1, employer_hash, employer_hash_loose, employer_known, template_cluster_t030, template_cluster_t050, template_cluster_t070, n_postings_in_group, group_employer_verifiable, edge_provenance`) + manifest with all parameters, `loose_merge_count`, per-τ component counts, largest component per τ and per stratum, and the sha256 of `sentence_pool.parquet`, `posting_keys.parquet` and every raw batch.

### §5. `sample.py` changes

- `_cap_per_group(df, cap, seed, group_of)`: shuffle `frac=1.0, random_state=seed`, `groupby(group_id, sort=False).head(cap)` — identical shape to the existing `_cap_per_posting`.
- `_take_stratum(df, k, local_frac, seed, group_cap=None, group_of=None)`: insert `df = _cap_per_group(df, group_cap, seed, group_of)` **immediately after** the existing `_cap_per_posting` line and **before** the content/borderline split, so every downstream pick and the top-up path all draw from an already-capped frame and the invariant holds by construction.
- `draw_sample(n, exclude_ids=None, seed=SEED, group_cap=None, exclude_groups=None)`. `exclude_groups` filters postings by `group_id` before the stratum loop.
- Pilot and practice call with `group_cap=None` and `exclude_groups=None` → **byte-identical** outputs. The main draw calls `group_cap=MAX_SENT_PER_GROUP` and `exclude_groups = {groups containing any pilot_100 or practice_10 posting}`, in addition to the existing `exclude_ids`.
- After the main draw, assert `max sentences per group == 2` and `len(sample) == 1200`; record in the manifest `max_sent_per_group`, `n_groups`, `n_weight1_groups`, `n_weight2_groups`, and the pilot/practice group-exclusion cost in postings per stratum.
- Regression test T4: re-running the pilot draw reproduces the existing `pilot_100.parquet` `sentence_id` list exactly.

### §6. `split.py`

**6.1 Registry gate.** `split_registry.json` holds every allowed config with `registered_utc`, `registered_git_commit`, `content_sha256`, `frozen: true`. `split.py` refuses a config absent from the registry, and refuses a registry entry whose recomputed `content_sha256` differs. Registered at creation: `CONFIG_A` (720/240/240, **primary**), `CONFIG_A2` (600/240/360, registered, not run, with the slope arithmetic as its rationale), `CONFIG_B` (1440/480/480, nested enlargement), `ENLARGE_R1`, `ENLARGE_R2`.

**6.2 Load and key.** Read `main_1200.parquet`; left-join `leakage_groups` and `posting_keys` on `posting_id`. Any unmatched sentence → raise. Derive `country_language = country_iso + "_" + posting_language`; `site`; `quarter` (from the pool column); `occupation_group`; `length_band`.

**6.3 Targets.** `SPLIT_TARGETS` are the part totals. Per-axis per-part target counts come from **largest-remainder apportionment of the drawn sample's realised marginals** (never of the corpus), so test mirrors the sample composition (PROP:1433) and rare-category surplus lands in train, the only place oversampling is permitted (PROP:1431).

**6.4 Cost.** `J = Σ_axes AXIS_WEIGHTS[a] · Σ_parts Σ_values |achieved_count − target_count|`, over **marginals**, never the cross-product (1,200 sentences over stratum × language × platform × quarter × occupation is far too sparse for cell-wise apportionment).

**6.5 Forced assignments, in this order.**
1. Groups containing any `pilot_100`/`practice_10` posting → `excluded`, reason `pilot_group` / `practice_group`. (Already excluded at draw time; asserted again.)
2. Groups with `group_employer_verifiable == False` → **train**, reason `unverifiable_employer`. The enforcement claim must be *provable*: a posting whose employer is unknown cannot be shown absent from the other side. Cost ≈ 27–35 of 1,200 sentences; they still train.
3. Groups with an unparseable `date_posted_utc` → `excluded`, reason `date_unparseable`.
4. Regime-specific forcing (§6.9, §6.10).

**6.6 Parity correction.** For each part whose remaining capacity is odd, place exactly one weight-1 group by the deficit rule. Assert P1–P5 of D4 before proceeding.

**6.7 Allocation.** Groups ordered by `(-weight, group_id)`; a seeded `np.random.default_rng(SEED)` permutation breaks ties **within equal (weight, …) classes only**, applied before the sort so the sort stays stable and deterministic. For each group, choose the part with remaining capacity ≥ weight that minimises ΔJ; tie-break by larger remaining capacity, then by part order `(train, dev, test)`, then by `sha1(group_id + regime_id)`. The weight-2 phase precedes the weight-1 phase (guaranteed by the `-weight` sort). No swap-improvement pass, no temperature, no wall-clock exit.

**6.8 Cell-coarsening report (not a constraint).** Ladder `L0 = stratum × occupation_group × quarter × site`; `L1` drops `site`; `L2` drops `quarter`; `L3` merges `occupation_group` to `{UNRESOLVED, other}`; `L4 = stratum`. Any cell under `MIN_CELL_SENTENCES = 10` in the drawn sample merges into its `L+1` parent until all cells are ≥ 10 or `L4` is reached. Report `achieved_stratum_resolution` (share of sentences realised at each level) and `unfilled_cells`. Publish year/quarter stratification for Indonesia as **NOT ACHIEVED** with the counts that prove it (Indonesia 79% 2026, 47% one quarter, 89 eligible postings at or before 2024; the corpus is a 12-month scrape, 2025-11-29 → 2026-11-15).

**6.9 Temporal regime.** Per-stratum cut, derived from a registered **quantile**, not a date: `cut[s]` = the smallest date `d` such that ≥ (1 − `TEMPORAL_QUANTILE`) of stratum `s`'s non-excluded **sample sentences** have `date_posted_utc ≥ d`. Deriving from the sample (not the pool) makes the realised test share ≈ 20% by construction; registering a quantile removes the temptation to nudge a date. A single global cut is measurably unusable (2026-09-01 ⇒ 57.2% of Indonesia vs 14.3% of OECD). Reference points from the eligible set: Indonesia ≈ 2026-10-23 (21.8%), ASEAN ≈ 2026-09-22, OECD ≈ 2026-06-08.

Each group gets `[min_date, max_date]`. Groups entirely before the cut → train/dev. Groups entirely on or after it → test. **Straddling groups → wholly to the early side; their post-cut sentences are dropped from the regime entirely** (reason `straddles_temporal_cut_dropped`), not moved into train. Train/dev then contain no post-cut data and test contains no pre-cut data — both purity claims stay literally true, and the pre-cut sentences are not wasted. Measured scale of the collision: at an Indonesian cut of 2026-10-01, 244 of 1,442 late employers also appear earlier, covering 29.4% of late and 36.0% of early postings.

Mandatory `temporal_confound_report` with `confounded: true` and `drift_attributable: false` whenever any axis TV distance early-vs-late exceeds 0.25. The evidence, printed in the manifest: of 2,707 Indonesian 2026Q4 postings, 1,598 Jobstreet / 476 Glints / 298 Indeed, while pre-2026Q3 Indonesia is almost purely Indeed (Jobstreet 56, Kemnaker 5, Kitalulus 0 — Kitalulus exists only in 2026Q4), and 950 of 1,420 Indonesian-language postings sit in 2026Q4. The artifact carries its own disclaimer: a drop here cannot be attributed to terminology drift.

**Sub-report inside the same regime** (not a fourth artifact): `indeed_id_matched`, a platform-matched temporal slice restricted to `site == "indeed"` and `stratum == "indonesia"` on both sides. Indeed ID is the one Indonesian platform present across the whole window (2,555 eligible postings). It is confound-free and small (expect ~40–60 test sentences), so it is emitted with its own n, its own bootstrap CI and `underpowered: true`. It is the only number that can support or refuse a drift claim.

**6.10 Cross-country regime.** Hold-out countries chosen **by a registered rule the examiner can re-run**, not picked: among non-Indonesian countries, keep those with ≥ 400 pool postings, ≥ 200 distinct employers, ≥ 10,000 candidate sentences of 20–300 chars, and English dominant (the three annotators read only English and Indonesian; the ASEAN pool holds 73 Thai, 154 Vietnamese, 3 Indonesian-language postings); then take the largest by candidate sentences within each non-Indonesian stratum. This resolves deterministically to **SG** (3,023 / 925 / 96,429) for the ASEAN row of Tabel 3.5 and **US** (2,621 / 1,564 / 98,215) for the OECD row. Indonesia is never held out — it is the primary claim basis.

Both hold-outs are **re-partitions of the same annotated 1,200**, costing zero extra annotator time. Country exclusion alone is insufficient (2,264 employers span more than one country), so the rule applies at **group** level: any group containing any posting from the held-out country moves **wholly** to the external test, so a multinational employer's other-country postings leave train rather than bridging the boundary. Train/dev are built from the remainder by the same allocator. Record the measured external-test size — **do not predict it** — and set `underpowered: true` below `MIN_TEST_SENTENCES`. Since train then exceeds 720, emit a deterministic size-matched `train_720` subsample (whole groups removed, weight-1 first) so an external-test gap is not confounded with train size (the curve gives .424 @720 vs .468 @1440, seed noise 0.030). Record `cross_lingual: false`: this measures cross-**country**, not cross-lingual, transfer. The only published anchor is NNOSE's English→English cross-dataset table (in-dataset span-F1 49.6–92.1 → cross-dataset 9.4–43.2), which mixes label-space mismatch with domain shift and is therefore an upper bound on the expected drop, not a prediction.

**6.11 Write order — fail closed.** Compute → assert (§7) → write to a temp directory → `os.replace` into place. A failed assertion leaves the previous split untouched and no partial split on disk. Then append one line to `DATA/gold_standard/splits/version_log.jsonl` and refresh `splits/index.json`.

**6.12 Label independence.** The assignment is computed from **metadata only** and is frozen **before** adjudication output exists. `split.py` writes no annotation column anywhere (invariant I17). Labels are materialised later by `seal.py`. This pre-registers the split independently of the labels — a property neither design stated, and it costs nothing because all three experts annotate all 1,200 sentences anyway (PROP:1430), so sealing is about locking the adjudicated **labels** away from the modelling loop, not about hiding text.

### §7. Leakage invariants

Write-time, in `split_assert.py`, pure functions over the in-memory assignment frame and the cached artifacts. All must pass before any file is created; one failure **raises**.

- **I1 Accounting.** Every input `sentence_id` lands in exactly one of `{train, dev, test, excluded}`; no duplicates; the union equals the input exactly; every excluded row carries a non-empty reason from the closed set `{pilot_group, practice_group, unverifiable_employer, date_unparseable, straddles_temporal_cut_dropped, heldout_country_group}`.
- **I2 Group disjointness** (the primary invariant). `group_id` sets of train, dev and test are pairwise disjoint. Dev is included although PROP:1192/1436 name only train vs test; the manifest records `dev_disjointness_is_stricter_than_proposal: true` as an inference, not a quotation.
- **I3 Posting disjointness.** `posting_id` sets pairwise disjoint. Implied by I2, asserted separately because it fails independently when the group join goes stale.
- **I4 Employer-exact disjointness.** Non-empty `employer_hash` sets pairwise disjoint.
- **I5 Employer-loose disjointness.** Non-empty `employer_hash_loose` sets pairwise disjoint.
- **I6 Template cluster disjointness.** `template_cluster_t030` sets pairwise disjoint. (`t050`/`t070` are logically implied by monotonicity and are **reported, not asserted** — asserting them adds no information.)
- **I7 Direct pairwise re-derivation** (independent of the union-find). `max` exact Jaccard over the cached shingle sets between any two postings in different parts `< TAU_BUILD`. This can only fail if the group map is stale or the shingle cache has drifted, which is exactly what it is for. ~720×240 + 720×240 + 240×240 ≈ 400k pairs on sorted uint64 arrays — seconds. Record the observed maximum.
- **I8 Sentence-text disjointness.** No exact `sentence_text` and no normalised `sentence_text` (casefold, punctuation stripped, whitespace collapsed) appears in more than one part; and no cross-part sentence pair has token-set Jaccard ≥ `SENT_NEAR_DUP_JACCARD`. This catches the one channel neither employer nor template grouping sees: identical boilerplate requirement lines shared by unrelated employers. `sample.py` already dedups exact text pool-wide; the normalised and near-duplicate forms are what can still bite.
- **I9 Test/dev verifiability.** Every dev and test posting has a non-empty `employer_hash`; `dev_test_employer_verifiable_share == 1.0`.
- **I10 Quarantine.** No `pilot_100` or `practice_10` `posting_id`, `sentence_id` or `group_id` in any part of any regime.
- **I11 Oversize guard.** No group in dev or test exceeds `MAX_GROUP_SHARE × part_quota` (= 60 sentences at 240). Vacuous at weight ≤ 2; kept so a future looser cap fails loudly.
- **I12 Exact sizes.** For `grouped_random`, achieved sizes equal `SPLIT_TARGETS` exactly, else raise. For `temporal` and `cross_country`, achieved test ≥ `MIN_TEST_SENTENCES` or the regime is written with `underpowered: true` and the number stated.
- **I13 Determinism.** `assignment_sha256` over canonically sorted `(sentence_id, part, group_id)` triples is recorded; the manifest hash excludes the timestamp field.
- **I14 Provenance binding.** The manifest records sha256 of `main_1200.parquet`, `sentence_pool.parquet`, `posting_keys.parquet`, `leakage_groups.parquet`, `shingles.parquet`, the flags file and every raw batch with its row count. A consumer loading a split re-checks the group-map sha256 and **refuses** on mismatch, so a split can never be interpreted against a regrouped corpus. Also record `pool_append_stable: false`.
- **I15 Regime-specific.** Temporal: `max(date)` over train∪dev `< cut[s] ≤ min(date)` over test, per stratum, and no group spans a cut. Cross-country: the held-out ISO appears in no part, and no group containing a held-out-country posting appears in any part.
- **I16 Reporting invariant.** Every key under `targets.*` has a counterpart under `achieved.*`. A target without a measured achieved value is a **write-time error**.
- **I17 No labels.** No file written by `split.py` contains any annotation/label column; `test_blind.parquet`'s column set is checked against an allowlist.

Test-suite-only (`tests/test_split.py`, not write-time):
- **T1 Positive controls.** Plant a known straddle — one shared employer hash, one shared loose hash, one template bridge at Jaccard ≈ 0.4, one duplicate normalised sentence — and assert that I2, I4, I5, I6, I7 and I8 each actually **fire**. An invariant never observed to fail is not evidence.
- **T2 Parity proof.** Synthetic weight distributions, including all-weight-2, all-weight-1, and odd forced weight, land exactly on `(720, 240, 240)`; a weight-3 group raises on P1.
- **T3 Seed consumption.** The RNG is consumed and the tie-break permutation differs between seeds (not "the assignment differs", which could legitimately be seed-invariant under tight constraints).
- **T4 Pilot immutability.** Re-running the pilot draw reproduces `pilot_100.parquet`'s `sentence_id` list exactly; the practice draw likewise.
- **T5 Seal access.** No module except `gold_standard/seal.py` and `tests/` references `test_labels.parquet` or reads a label column from a test artifact.

### §8. Manifest schema (`manifest.json`, one per split directory)

Every quota and every distribution is `{"target": …, "achieved": …, "delta": …}`; `unmet_targets` lists each as a human-readable string, mirroring `sample.py`'s existing WARNING convention so an unmet target is loud in stdout as well as recorded.

```
identity:        split_spec_version, regime_id, split_id, config_name, created_utc, git_commit, seed,
                 gold_standard_version, guideline ("AG-RPL-LIMA-v3.0"), governance_version
registry:        registry_entry_sha256, registered_utc, registered_git_commit, frozen
provenance:      sample_file+sha256, pool_file+sha256, posting_keys_file+sha256,
                 leakage_groups_file+sha256, shingles_file+sha256, flags_file+sha256,
                 raw_batches[{batch_id, rows, sha256}], pool_append_stable: false
group_layer:     tau_build, tau_asserted, lsh_threshold, num_perm, shingle_words, max_chars,
                 text_pipeline ("build_pool._html_to_text, lowercased, whitespace-split"),
                 hash_fn ("blake2b-64"), n_nodes, n_employer_exact_edges, n_employer_loose_edges,
                 loose_merge_count, n_verified_template_pairs_per_tau,
                 largest_component_postings_per_tau (global and per stratum),
                 n_pairs_jaccard_ge_085_cleaned, n_groups, n_weight1_groups, n_weight2_groups,
                 max_sent_per_group_cap, max_sent_per_group_observed
sizes:           {train,dev,test}: {target, achieved, delta}; exact_sizes_met;
                 mechanism ("parity-greedy"); parity_corrections[]
marginals:       per axis (stratum, country_iso, country_language, posting_language,
                 is_indonesian_sentence, site, quarter, occupation_group, length_band)
                 × per part: {target, achieved, delta, tv_distance}
stratification:  ladder levels, MIN_CELL_SENTENCES, achieved_stratum_resolution{L0..L4},
                 n_cells_final, unfilled_cells[{cell, sample_n, missing_from_part, collapsed_to}],
                 occupation_stratified {indonesia: true, asean: false, oecd: false},
                 unresolved_occupation_share per stratum,
                 year_stratification: {verdict: "NOT ACHIEVED (indonesia)", per_year_counts}
exclusions:      counts by reason, full excluded sentence_id list
verifiability:   n_unknown_employer_sentences, by stratum and site, policy "forced_to_train",
                 dev_test_employer_verifiable_share (must be 1.0)
invariants:      {I1..I17: {passed, observed}}, max_cross_part_jaccard,
                 n_cross_part_near_dup_sentence_pairs,
                 dev_disjointness_is_stricter_than_proposal: true
power:           n_test_sentences, n_test_spans (measured or assumed),
                 spans_per_sentence {assumed: 1.6, measured: …, source}, deff (1.18),
                 expected_micro_f1_ci_width_at_0_5, paired_mde_80pct_power,
                 indonesia_only_ci_width, pooled_estimator {unweighted, stratum_weighted}
tag_support:     per tag {n_spans, n_sentences} per part, MIN_TEST_SUPPORT_SPANS_PER_TAG: 120,
                 derivation_note ("CI_WIDTH_CLAIMABLE=0.20 -> N>=96.04 -> xDEFF 1.18 -> 120"),
                 origin: "code-origin; absent from the proposal and from Paket B v3",
                 support_limited, claimable
enlargement:     evaluated_on "train+dev adjudicated gold only; test never opened",
                 triggered, triggered_tags, recommended_config, rule_ids ["ENLARGE_R1","ENLARGE_R2"],
                 rule_registered_utc, nested: true
temporal:        temporal_quantile, cut_date per stratum (derived), n_early, n_late,
                 n_straddling_groups, n_dropped_sentences, straddle_policy "early_side_drop_late",
                 temporal_confound_report (site/language/quarter early-vs-late + tv_distance),
                 confounded: true, drift_attributable: false,
                 indeed_id_matched {n_train, n_test, underpowered}
cross_country:   holdout_rule_text, candidate_table (per country: pool_postings, employers,
                 candidate_sentences, dominant_language, passed), holdout_country, holdout_stratum,
                 n_groups_moved_wholly, n_employers_spanning_boundary,
                 exclusion_cost {sentences, postings, groups} by stratum, cross_lingual: false,
                 train_720_subsample {present, sha256}
seal:            train_sha256, dev_sha256, test_blind_sha256, test_sentence_id_list_sha256,
                 labels_materialised (bool), test_labels_sha256 or null,
                 receipts[], opened_utc or null, test_open_log_path
regime_overlap:  note "the three regimes re-partition the SAME 1,200 sentences; their test sets
                 overlap, so the three numbers are not independent and must not be averaged"
assignment:      assignment_sha256
unmet_targets:   [strings]
```

### §9. The three regime artifacts

```
DATA/gold_standard/splits/
  version_log.jsonl          append-only; one line per write (the version log PROP:1431 requires)
  index.json                 regime -> {path, created_utc, assignment_sha256, sealed}
  test_open_log.jsonl        append-only receipts (seal.py only)

  grouped_random_v1/         PRIMARY. Exactly 720/240/240. The reported benchmark.
  temporal_v1/               Per-stratum quantile cut; straddlers to the early side, late halves dropped.
                             Includes the indeed_id_matched sub-report. Sizes regime-determined.
  cross_country_v1/sg/       ASEAN external test (Tabel 3.5 row 2). Whole groups touching SG -> external test.
  cross_country_v1/us/       OECD external test (Tabel 3.5 row 3). Indicative; claimability stated in-manifest.
```

Each split directory contains exactly: `assignment.parquet` (`sentence_id, posting_id, group_id, part, exclusion_reason`), `assignment.csv` (same, for inspection), `train.parquet` and `dev.parquet` (full sentence text + context + provenance, **no labels**), `test_blind.parquet` (`sentence_id` + text + context + provenance, **no labels**, column set allowlisted by I17), `manifest.json`, `SEAL.json` (stamp, `test_sentence_id_list_sha256`, `assignment_sha256`, `opened_utc: null`), and — only after `seal.py` runs — `labels/test_labels.parquet`. Cross-country directories additionally contain `train_720.parquet`.

`cross_country_v1` is **one regime with two sibling split ids**, which keeps the count at three regimes while serving both external rows of Tabel 3.5.

### §10. `seal.py`

`score_on_test(regime_id, split_id, model_config_sha, git_commit, reason)`:
1. Re-check `test_labels.parquet`'s sha256 against the frozen manifest; on mismatch, **void** every test number with an explicit verdict string rather than scoring the changed file.
2. Append a receipt to `test_open_log.jsonl`: `receipt_id, utc, config_sha, model_config_sha, git_commit, reason`.
3. Refuse a second open for the same `(config_sha, model_config_sha)` unless `--force-reopen` is passed **with** a written reason, which is also logged (PROP:1264: the test and the temporal window are opened once, for the selected configuration).
4. Return **aggregates only** on a first open — micro/macro F1 with bootstrap CI and per-tag support, never per-sentence errors.
5. Every published test metric carries a `receipt_id` that resolves in the log.

`materialise_labels(regime_id, split_id, adjudicated_gold)` is the only writer of `labels/test_labels.parquet`. Items adjudicated `AMBIGUOUS`/`EXCLUDE` are dropped from the strict test and recorded in `realised_test_n` (B3:413); the realised test is therefore ≤ 240 and is **never backfilled**.

The guarantee is **auditability, not impossibility** — nothing stops a researcher opening the parquet by hand. Claim exactly that.

### §11. Run order

```
.venv-ml\Scripts\python.exe -m gold_standard.posting_keys --flags record_flags_20261115T164444Z.parquet
.venv-ml\Scripts\python.exe -m gold_standard.groups
.venv-ml\Scripts\python.exe -m gold_standard.sample --main         # draws main_1200 with group_cap=2
.venv-ml\Scripts\python.exe -m gold_standard.split --config CONFIG_A --regime grouped_random
.venv-ml\Scripts\python.exe -m gold_standard.split --config CONFIG_A --regime temporal
.venv-ml\Scripts\python.exe -m gold_standard.split --config CONFIG_A --regime cross_country
.venv-ml\Scripts\python.exe tests\test_split.py
```

Run `python.exe` directly, never through `run.ps1` (its `ErrorActionPreference=Stop` turns pandas warnings into fatal errors). No network, no new dependencies — `datasketch 2.0.0`, pandas, numpy only. Same input + same seed ⇒ byte-identical output, which T1–T4 check.

---

## PART 2 — WHAT MUST NOT BE BUILT (scope discipline)

1. **Do not rebuild `sentence_pool.parquet`** before `main_1200` is drawn. It would break `pilot_100_manifest.json`'s `pool_sha1_16 = 9061b3b07b7d971c`, force a re-segmentation whose `_latest_flags()` may pick a different flags file, and change the pool sha every manifest records. The `build_pool` two-name widening is forward compatibility for the *next* rebuild only.
2. **Do not modify `governance/ladder.py`, `governance/rules.py`, `governance/configs/*` or `acquisition/schema.py`.** The ladder's `flags_out` does not need to carry `employer_hash`; the sidecar reads the raw batches. `LOAD_COLUMNS` must not grow a `company` entry.
3. **Do not re-draw `pilot_100` or `practice_10`.** They may already be in annotators' hands. T4 pins them.
4. **Do not add `employer_hash`, `employer_hash_loose` or `query_family` to `SAMPLE_COLUMNS`.** It would ship employer hashes into the annotator worksheets for no benefit.
5. **Do not build a job-title → occupation keyword classifier.** `UNRESOLVED` is the honest value for the 74% of OECD and 46% of ASEAN postings whose `query_family` is a scrape bucket.
6. **Do not build the group map over all 85,740 eligible postings.** Employer relations give an identical pool partition; the two-hop template residual is covered by I7 at 1/50th the cost.
7. **Do not build a swap-improvement / simulated-annealing allocator.** The parity proof removes the need, and a bounded stochastic pass adds a reproducibility surface for no measurable balance gain.
8. **Do not build a subset-sum DP.** Replaced by the proof.
9. **Do not implement the R=20 re-split variance study, the grouped-vs-random contrast arm, or grouped k-fold** in this work item. All three are valuable — the grouped-vs-random gap would be the first such measurement in this literature — but they need a frozen configuration and a receipt for `grouped_random_v1`, and grouped k-fold is incompatible with the single-open test rule (PROP:1264). Register them; run them later.
10. **Do not annotate a new external-country block, and do not exclude SG or US from `main_1200`.** The cross-country regime re-partitions the existing 1,200 at zero annotation cost. A dedicated block is the documented escalation under `ENLARGE_R1` if the measured external test falls below `MIN_TEST_SENTENCES`, not a prerequisite.
11. **Do not make 600/240/360 or 1440/480/480 the primary.** Register them, do not run them.
12. **Do not assert template disjointness at τ = 0.50 or 0.70.** Monotonicity makes those assertions uninformative; report them instead.
13. **Do not assert "two seeds give different assignments" at write time.** Test T3 only.
14. **Do not write a dashboard, Streamlit page, Markdown report or figure** for any of this. Manifests plus `tests/test_split.py` are the deliverable surface.
15. **Do not average or pool the three regimes' test scores.** They re-partition the same 1,200 sentences; their test sets overlap and the numbers are not independent replications. The manifest says so in `regime_overlap`.

## Residual limitations that must be carried into the write-up verbatim

τ = 0.30 is still a chosen floor — nothing rules out a true template pair at 0.25. MinHashLSH recall is below 1.0, so template edges can be **missed**; exact verification removes false positives, not false negatives, and grouping is a **lower bound** on true template relatedness. `employer_hash` and `employer_hash_loose` are string hashes, so abbreviations, misspellings, subsidiaries and rebrands are still missed; the claim is "employer-disjoint by the recorded hashes", never "employer-disjoint". 290 eligible Indonesian postings (5.07%; 14.6% of eligible Jobstreet ID) have no employer at all and are confined to train, which biases the test platform mix away from Jobstreet. Year stratification is **not achieved** for Indonesia. Occupation stratification is real for Indonesia only, and even there `software_developer` is 3,155 of 5,667 pool postings while `system_analyst` has 5. The Indonesian temporal hold-out cannot separate time from platform or language, and only the small `indeed_id_matched` slice can speak to drift. At 240 test sentences the micro-F1 CI is ≈ 0.109 wide, the paired MDE at 80% power is ≈ 0.055, and Indonesia-only is ≈ 0.162 — so nothing below 0.05 F1 is claimable, and at ~8% prevalence WB needs ~1,500 natural test sentences to reach 120 spans, which PROP:1433's natural-distribution clause forbids reaching by oversampling. WB and probably SS will publish as `support_limited: true, claimable: false` even under `CONFIG_B`. `spans_per_sentence = 1.6` is an assumption until `pilot_100` is adjudicated; every power figure scales as 1/√s and must be recomputed against the measured value. Finally, employer/template-grouped splitting, MinHash near-duplicate control across splits, the temporal hold-out and the cross-country external test have **no precedent** in the reviewed literature and must be framed as this project's own methodological contribution, with NNOSE's cross-dataset table (49.6–92.1 in-dataset → 9.4–43.2 cross-dataset, label-space caveat attached) as the only available anchor.
---

# APPENDIX C — DEFECTS IN THIS CONTRACT (found by implementation + adversarial review, 2026-11-16)

**Status: NOT RATIFIED. The v1.0.0 body above is unchanged and remains the contract of record.** This appendix exists so that the places where the *contract itself* is wrong are not rediscovered as implementation bugs, and so no reader concludes that the code's departures were silent. Each item names the clause, what the code does instead, and why. Ratifying any of these means issuing v1.1.0; until then the code's own docstrings and the written manifests carry the same statements at the point of use.

1. **§D2 / §4.2 — carrying `SHINGLE_MAX_CHARS` (2,500) into the leakage signature is wrong.** The parity it cites with `governance.ladder.minhash_near_dups` is not required (the ladder runs on RAW text for 0.85 exact-dup detection, and §D2's own "consequence to measure" paragraph concedes the two layers are not comparable). Measured cost of the window: 33,338,457 of 74,433,388 cleaned characters (44.8%) were never compared, and 126 real template pairs at full-text Jaccard >= 0.30 fell below the threshold once truncated — 42 of them to exactly 0.000. **Code: `textsig.LEAKAGE_WINDOW_CHARS = None`, full cleaned text.** `SHINGLE_MAX_CHARS` survives only as `textsig.LADDER_PARITY_WINDOW_CHARS`.
2. **§2's constant block mislabels `SHINGLE_MAX_CHARS` as a group-layer constant.** It is a ladder-parity constant.
3. **§D2's `MinHashLSH(threshold=0.25, num_perm=128)` is wrong as a specification of the edge set,** and the "Residual limitations" paragraph's "MinHashLSH recall is below 1.0, so template edges can be missed" no longer describes the implementation. Measured LSH recall at full text: 0.8393 (11,272 of 13,430 pairs), 380.8 s. **Code: exact inverted index over shingles with document frequency > 1, recall 1.0 by construction, 8.2 s.** The spec's cost justification for LSH does not hold on this corpus.
4. **§D1's claim that I7 is "a stronger guarantee than A's closure" is wrong.** I7 re-derives from the same signature cache, so it is a consistency check over that cache, not an independent oracle. The genuine residual — a two-hop template bridge through a NON-POOL posting — is closed by neither layer (and Part 2 §6 forbids the 85,740-posting build that would close it). Stated as a residual in `leakage_groups_manifest.claim_wording`.
5. **§3.4 — "If the residual is empty or shorter than 3 characters, return empty" is wrong in its `< 3` half.** The justification given argues only for the empty case; the `< 3` branch switched the loose relation OFF for exactly the legal-form variant families it exists to catch (146 pool postings, `employer_known` still True so they stayed freely placeable). **Code: a 1-2 character residual IS hashed, with `employer_loose_status = "short_residual"`.** §3.4's normalisation ("strip all non-alphanumeric to spaces") is also wrong: implemented ASCII-only it deleted 82 of 84 `residual_empty` pool postings' names outright (Thai/CJK/Katakana/Hangul), and a `\W`-based class splits Thai at every combining vowel mark. **Code keeps Unicode categories L/M/N.** Add `employer_loose_status` to the §3.7 column list, and record in §3 that the empty-key fork is DISTINCT, never collapsed (enforced in `groups._equality_edges`).
6. **§D4 precondition P2 (sum of weights == 1200 == 720+240+240) is unsatisfiable alongside the contract's own exclusion reasons.** §6.5.1 mandates `pilot_group` / `practice_group` exclusions and `date_unparseable` is a legal reason in `grouped_random`; either firing makes P2 false and left the PRIMARY regime unbuildable. **Code: capacity = assignable weight, the shortfall is charged to train, dev and test keep their registered sizes, and the register-vs-realised gap is reported in `sizes` / `unmet_targets` / `regime_capacity.exact_capacity` rather than raised.** §7's I12 ("achieved sizes equal `SPLIT_TARGETS` exactly, else raise") needs the same amendment.
7. **§D6 / §8 / I16 as written is a key-PRESENCE rule** ("I16 fails the write if any key under `targets.*` has no counterpart under `achieved.*`") and is too weak to catch the defect it was introduced for: a manifest that derived `target` from `achieved` published delta 0, `exact_sizes_met: true` and `unmet_targets: []` for a test set 28 short of 240, with I16 green. **It must be a VALUE rule.** Code: `split_assert.i16_reporting` now requires registered-target provenance, `delta == achieved - target`, marginal column closure, and that `unmet_targets` names every missed part.
8. **§8 does not say what `sizes[*].target` is.** It should read: target = the registered config, achieved = recounted from the written artifacts, plus a separately named regime capacity. That silence is how the circular target passed review.
9. **§6.3** requires per-axis targets from "largest-remainder apportionment of the drawn sample's realised marginals" without requiring each `(axis, part)` target column to close to the part total, and without saying the base is the ASSIGNABLE rows. Both are now implemented (two-stage largest remainder); the clause should state them.
10. **§8 under-specifies the confound report.** `temporal_confound_report (site/language/quarter early-vs-late + tv_distance)` names exactly the four axes whose narrowness was finding F11. It should read "every axis in `AXES`".
11. **§6.8 / §8 prescribe hardcoded prose and corpus-specific constants in the manifest** ("NOT ACHIEVED (indonesia)", "79% 2026", "89 eligible postings"), in an artifact the contract elsewhere declares `pool_append_stable: false`. **Code derives the verdict and its evidence per run and prints the criterion.** Write-up prose may quote measured numbers; a manifest must not assert them.
12. **§10 is silent on the SCORING DENOMINATOR,** which is how the second fatal finding shipped with every invariant green: a gold-empty sealed sentence was not a scoring unit, so a model could hallucinate on every non-skill sentence and still publish F1 1.0. §10 should read: "the scoring population is every sentence in `test_blind.parquet`; a gold-empty sealed sentence is a scoring unit."
13. **§10's signature is wrong.** It documents `score_on_test(regime_id, split_id, model_config_sha, git_commit, reason)`; the implemented signature is `score_on_test(split_id, predictions, *, model_config_sha, git_commit, reason, force_reopen=False, splits_dir=...)`. There is no `regime_id` (the split id identifies the regime, and cross-country has two sibling split ids under one regime) and the contract omits `predictions`, without which the function cannot exist.
14. **§10.1 names the wrong artifact as the sha authority** ("re-check `test_labels.parquet`'s sha256 against the frozen *manifest*"). The frozen digest lives in `SEAL.json`, which is the better place: the manifest is written before labels exist.
15. **§8's `seal:` block was not implemented** — `train_sha256`, `dev_sha256`, `test_blind_sha256` were promised and nothing wrote them, so `seal.score_on_test` could verify the test sentence-ID LIST but never the test TEXT. **Now written by `split.py` and checked by `seal.py`.**
16. **Contract I13 promised "a manifest hash excluding the timestamp" and no such field existed.** Now written as `manifest_sha256_excluding_timestamps` (canonical JSON, `identity.created_utc` and `group_layer.created_utc` removed), verified reproducible across two runs over the same inputs.
17. **I10's wording ("in any part of any regime") contradicts §6.5.1's forcing rule** if `excluded` counts as a part: §6.5.1 PUTS a quarantined group in `excluded`, so scanning that bucket made the mandated exclusion reason unreachable. §6.5.1 wins; I10 scans `train`/`dev`/`test` only, and the spec wording should say so.
18. **§8 / I14 ask for a row count per raw batch** and the producer recorded `batch` + `sha256` only. Now written (`raw_batches[{batch, rows, sha256}]`, 25 batches).
