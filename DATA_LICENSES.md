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
- Source: MERLIN project, distributed via CLARIN.
- License: CC BY-SA 4.0. # VERIFY version
- Cite: Boyd, Hana, Nicolas, Meurers, Wisniewski, Abel, Schöne, Štindlová,
  Vettori (2014), "The MERLIN corpus: Learner language and the CEFR", LREC.

## COWS-L2H (Spanish — GEC only)
- Source: UC Davis Computational Linguistics Lab, GitHub.
- License: per repository license file. # VERIFY
- Cite: Davidson, Yamada, Fernandez-Mira, Carando, Sanchez-Gutierrez,
  Sagae (2020), "Developing NLP tools with a new corpus of learner Spanish",
  LREC.
- Note: COWS-L2H is organized by course level, which is NOT a CEFR label.
  Spanish is therefore GEC-only in this benchmark. A course→CEFR mapping was
  considered and rejected as not defensible without an external calibration
  study. # DECISION

## CLC-FCE (English supplement — optional)
- Source: Cambridge Learner Corpus, FCE subset released for research.
- Status: NOT integrated in v1. Listed because the spec allows it "if easily
  obtainable"; it is registration-gated, so it is a post-handoff nice-to-have,
  not a dependency. # DECISION
