# Dataset licenses and citations

No corpus text is committed to this repository, in any form: not raw files,
not the prepared JSONL, not inside the raw response cache (gitignored), and
not inside the committed results DB (which is schema-restricted to derived
scalars). Only sampling manifests (item IDs + seeds) are committed. Users
obtain each corpus themselves under its own license via
`scripts/prepare_data.py`, which prints per-dataset instructions.

All URLs below are # VERIFY items (this file was written without live access).

## W&I+LOCNESS (English — GEC + CEFR bands A/B/C)
- Source: BEA-2019 Shared Task on Grammatical Error Correction
  (https://www.cl.cam.ac.uk/research/nl/bea2019st/),
  wi+locness_v2.1.bea19.tar.gz; Write & Improve (Cambridge) + LOCNESS
  (native essays, Louvain). Downloaded 2026-07-25.
- License (VERIFIED 2026-07-25 from licence.wi.txt and license.locness.txt
  inside the archive, as displayed at download):
  - W&I: Cambridge English Write & Improve (CEWI) Dataset Licence — copyright
    University of Cambridge; non-exclusive, non-transferable right for
    NON-COMMERCIAL research and educational purposes only; published excerpts
    limited to under 100 words; citation of Yannakoudakis et al. (2018)
    required in all publications.
  - LOCNESS: non-commercial use only; credit to CECL (UCLouvain) with a copy
    of publications to CECL; **no part of the corpus may be distributed to a
    third party without specific authorization from CECL**.
  - Redistribution is therefore NOT permitted for either corpus. This
    repository complies by construction: it ships only scripts and sampling
    manifests (item IDs + seeds), NEVER the texts — not in fixtures, not in
    the raw cache (gitignored), not in the results DB (schema-restricted to
    derived scalars and closed-set labels).
- Cite: Bryant, Felice, Andersen, Briscoe (2019), "The BEA-2019 Shared Task
  on Grammatical Error Correction", BEA@ACL; Yannakoudakis, Andersen,
  Geranpayeh, Briscoe, Nicholls (2018), "Developing an automated writing
  placement system for ESL learners", Applied Measurement in Education
  (required by the W&I licence). Also Granger (1998) for LOCNESS.
- Note: CEFR granularity is A/B/C bands, not six levels. English CEFR metrics
  are computed on 3 bands and reported as such — never pooled with six-level
  QWK from MERLIN languages. v2.1 ships per-band dev files, so dev sentences
  carry bands too (see DECISION 34).

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
