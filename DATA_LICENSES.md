# Dataset licenses and citations

No corpus text is committed to this repository, in any form: not raw files,
not the prepared JSONL, not inside the raw response cache (gitignored), and
not inside the committed results DB (which is schema-restricted to derived
scalars). Only sampling manifests (item IDs + seeds) are committed. Users
obtain each corpus themselves under its own license via
`scripts/prepare_data.py`, which prints per-dataset instructions.

All URLs below are # VERIFY items (this file was written without live access).

## W&I+LOCNESS (English — GEC + CEFR bands A/B/C)
- Source: BEA-2019 Shared Task on Grammatical Error Correction; Write &
  Improve (Cambridge) + LOCNESS (native essays, Louvain).
- License: research use under the terms stated on the BEA-2019 data page;
  redistribution not permitted. # VERIFY exact terms
- Cite: Bryant, Felice, Andersen, Briscoe (2019), "The BEA-2019 Shared Task
  on Grammatical Error Correction", BEA@ACL. Also Granger (1998) for LOCNESS.
- Note: CEFR granularity is A/B/C bands, not six levels. English CEFR metrics
  are computed on 3 bands and reported as such — never pooled with six-level
  QWK from MERLIN languages.

## MERLIN (German, Italian, Czech — GEC target hypotheses + six-level CEFR)
- Source: MERLIN Written Learner Corpus for Czech, German, Italian v1.2,
  Eurac Research CLARIN Centre repository,
  https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/59
  (merlin-text-v1.2.zip + merlin-metadata-v1.2.zip; browser download —
  repository sits behind anti-bot protection). Downloaded 2026-07-24,
  no registration required.
- License: CC BY-SA 4.0 (Attribution-ShareAlike 4.0 International —
  VERIFIED 2026-07-24 from the LICENSE file inside merlin-text-v1.2.zip).
- Cite (corpus): Wisniewski, Abel, Vodičková, et al. (2018), "MERLIN Written
  Learner Corpus for Czech, German, Italian 1.2", Eurac Research CLARIN
  Centre, hdl:20.500.12124/59.
- Cite (paper): Boyd, Hana, Nicolas, Meurers, Wisniewski, Abel, Schöne,
  Štindlová, Vettori (2014), "The MERLIN corpus: Learner language and the
  CEFR", LREC.
- Notes from the real distribution: 2,287 texts (1,033 de / 813 it / 441 cs);
  ratings span A1-C2 plus literal "EMPTY"/"unrated" values (mapped to
  no-CEFR-label); target hypothesis TH1 (minimal correction) is the GEC
  reference, TH2 (appropriateness) deliberately excluded.

## COWS-L2H (Spanish — GEC only)
- Source: UC Davis Computational Linguistics Lab,
  https://github.com/ucdaviscl/cowsl2h (cloned 2026-07-25).
- License: Apache License 2.0 (VERIFIED 2026-07-25 from the repository's
  LICENSE file).
- Cite: Davidson, Yamada, Fernandez-Mira, Carando, Sanchez-Gutierrez,
  Sagae (2020), "Developing NLP tools with a new corpus of learner Spanish",
  LREC.
- Notes from the real repository: layout is <topic>/<term>/essays/ with
  corrected/ siblings (holistic corrections by graduate-level Spanish
  instructors); a subset carries a second instructor's correction (' (1)'
  files) used as an additional GEC reference; annotated/ error-type files
  are not GEC references.
- Note: COWS-L2H is organized by course level, which is NOT a CEFR label.
  Spanish is therefore GEC-only in this benchmark. A course→CEFR mapping was
  considered and rejected as not defensible without an external calibration
  study. # DECISION

## CLC-FCE (English supplement — optional)
- Source: Cambridge Learner Corpus, FCE subset released for research.
- Status: NOT integrated in v1. Listed because the spec allows it "if easily
  obtainable"; it is registration-gated, so it is a post-handoff nice-to-have,
  not a dependency. # DECISION
