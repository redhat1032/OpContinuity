# Sources

OpContinuity's corpus is **not committed to git**. PDFs live in `docs/` (git-ignored) and
are listed in the machine-readable manifest [`scripts/sources.json`](scripts/sources.json).
`scripts/fetch_docs.py` downloads each file and checks its SHA-256.

```bash
python scripts/fetch_docs.py                # download missing files into docs/ and verify
python scripts/fetch_docs.py --verify-only  # check files already in docs/
python scripts/fetch_docs.py --strict       # also fail if a manual-only file is missing
```

The script exits nonzero on any hash mismatch or download error. A mismatching download is
renamed to `*.pdf.mismatch` and is never indexed.

## Corpus

Hashes were computed from the exact files originally committed (retrieved 2025-06-23).
"Verified" means the listed URL was re-downloaded on that date and produced a byte-identical
file.

| File | Title | Official / reputable URL | Retrieved | Verified | SHA-256 | License / status | Why included |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ERIC_ED106512.pdf` | Signalman 1 & C: Rate Training Manual (NAVEDTRA 10136-C), 1974. U.S. Naval Education and Training Command | https://files.eric.ed.gov/fulltext/ED106512.pdf (ERIC, U.S. Dept. of Education) | 2025-06-23 | ✅ 2026-10-09 | `da8b19451c4a2901270513fb2661b5ebfb1f11562fa30da4217ea75af0f31894` | U.S. Government work, public domain (17 U.S.C. §105) | Naval visual and radio signalling, flags, communications procedures |
| `FM3-25x26.pdf` | FM 3-25.26 Map Reading and Land Navigation (20 Jul 2001). HQ, Dept. of the Army | https://archive.org/download/FM3-25x26/FM3-25x26.pdf (Internet Archive) | 2025-06-23 | ✅ 2026-10-09 | `bf92a942048931c59f2b48b366acd9b5f8f40120b0866d702dec799a4e8be24c` | U.S. Government work, public domain; "Approved for public release: distribution is unlimited." Superseded by TC 3-25.26 | Map reading, compass use, land navigation |
| `FM4-25x11.pdf` | FM 4-25.11 First Aid (Dec 2002, with Change 1). HQ, Depts. of the Army, Navy, Air Force; USMC | https://archive.org/download/FM4-25x11/FM4-25x11.pdf (Internet Archive) | 2025-06-23 | ✅ 2026-10-09 | `62010e9dcc9204badd5940154d8b580260776405dba7bd5d99544722fdeb4ce3` | U.S. Government work, public domain; "Approved for public release; distribution is unlimited." | Field first aid and casualty care |
| `US-Air-Force-AFMAN36-2227V1-…-May-2004.pdf` | AFMAN 36-2227 Vol. 1, Combat Arms Training Programs: Individual Use Weapons (21 May 2004). Secretary of the Air Force | ⚠️ **None verified.** No official e-publishing.af.mil copy of this superseded edition was located | 2025-06-23 | ⚠️ Unverified | `4587d329d4e950a90e9cd5aab231746fcf91ffebfdbaf35385c6b2d988e6a0e7` | U.S. Government work, public domain; no releasability statement found in front matter (Distribution: F) | Weapons training and qualification procedures for realistic range and training scenes |
| `survival_manual.pdf` | FM 21-76 US Army Survival Manual (abridged third-party reprint) | ⚠️ **None verified.** No official or Internet Archive copy matches this file's hash | 2025-06-23 | ⚠️ Unverified | `d35bfb9875c7aeaf2109da9948d24c3ca2cc54e5cbf7c341413ed60ecbc5792d` | Underlying FM 21-76 text is a U.S. Government work; this abridged reprint is from an unknown third party, provenance unknown | Core survival topics: shelter, water, fire, food, navigation |

### Notes on unverified entries

- **AFMAN 36-2227V1 (2004).** It was originally downloaded from a non-official third-party
  website. That site still serves a byte-identical file, but it isn't an official or
  reputable source, so it isn't listed. The manifest has `"url": null`: place the file in
  `docs/` by hand and the script checks its hash. To-do: find an official or archived
  e-publishing.af.mil copy, or switch to the current edition.
- **FM 21-76 abridged reprint.** The file is a 2006 third-party abridgement, modified in
  2024, with unknown provenance. To-do: replace it with an official edition (for example
  FM 21-76 from the Internet Archive, or the current ATP 3-50.21), then update the hash.

## Deliberately excluded

- **TM 31-210, *Improvised Munitions Handbook* (1969)** was part of the original prototype
  corpus. It was **deliberately removed and excluded**, from the working tree and from git
  history. It's a manual for manufacturing improvised explosives and weapons, which is
  outside this project's writer-realism scope, and serving it through a RAG assistant would
  be an unacceptable misuse risk. It must not be re-added. The manifest lists it under
  `excluded`, and `fetch_docs.py` rejects a manifest that tries to include it.

## Adding a source

1. Prefer official publishers (armypubs.army.mil, e-publishing.af.mil, *.mil, *.gov,
   eric.ed.gov). Use the Internet Archive when an official copy no longer exists.
2. Download it, run `sha256sum`, and confirm the distribution / releasability statement.
3. Add an entry to `scripts/sources.json` and a row here, then run
   `python scripts/fetch_docs.py --verify-only` and `pytest`.
4. Rebuild the index with `python -m opcontinuity --rebuild`.
