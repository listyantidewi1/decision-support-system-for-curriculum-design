# Lexicon v3

`v3` is `v2` plus entries derived from **ESCO**, the European classification of Skills, Competences, Qualifications and Occupations, published by the European Commission. The ICT branch (ISCED-F 061, 295 concepts) was read through the public ESCO API on 2026-11-19. The concepts were sorted into technology names and knowledge terms and rewritten by `fivetag/build_lexicon_v3.py`; this list has been modified from the original and is not an ESCO product. `ESCO_DERIVATION.tsv` traces every entry to its ESCO concept; `ESCO_ICT.tsv` keeps each concept's URI and parent group.

- technology entries added: 232
- knowledge terms added: 150

---

Inherited from `v2`:

# Lexicon v2

`v2` is `v1` plus technology names derived from the O*NET 30.2 Database, file *Technology Skills*, by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under the CC BY 4.0 licence. The names were filtered (SOC 15-12xx, Hot Technology or In Demand) and rewritten by `fivetag/build_lexicon_v2.py`; this list has been modified from the original and is not an O*NET product. `ONET_DERIVATION.tsv` traces every added entry to the O*NET name it came from.

- O*NET names used: 194
- entries added to `technologies.txt`: 165
- all other files: identical to `v1`
