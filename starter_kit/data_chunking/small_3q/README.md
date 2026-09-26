# Small Track 1 chunking test set

This directory contains a fixed test corpus for **Genealogy, HSM, and Travel**.
It has three selected RETECO dev queries per domain, all their gold documents,
25 nearby distractor documents, and 25 distant distractor documents per domain.
There are no chunking or retrieval results here.

| Domain | Queries | Gold documents | Nearby noise | Distant noise | Total documents |
| --- | ---: | ---: | ---: | ---: | ---: |
| Genealogy | 3 | 6 | 25 | 25 | 56 |
| HSM | 3 | 7 | 25 | 25 | 57 |
| Travel | 3 | 6 | 25 | 25 | 56 |

`query_ids.json` records the selected dev IDs. `queries_<domain>.jsonl` contains
the original dev query records, including gold document IDs and answers.
`corpus_<domain>.jsonl` contains the selected documents with normalized text in
`{"id", "content"}` records. Every query and chunking method should use the same
domain corpus; do not draw new distractors per query or per method.

`manifest_<domain>.json` records every document ID, its group, word count, and
distance to a gold document. The `random` group in the manifest is the **distant
noise** group. It was sampled once within each domain using seed 42. Both noise
groups exclude selected gold documents and all known positive document IDs from
the RETECO train and dev examples/qrels.

For nearby noise, two-part document IDs were sorted by their numeric suffix;
neighbors were selected around the gold IDs in rounds. Distant noise was sampled
from the remaining two-part IDs at least **50 positions from every gold ID** in
that sorted list. `sorted_index_<domain>.tsv` preserves the full order used for
selection. Suffix proximity is only a proxy for shared source, not verified
document provenance.

The source is the public RETECO Track 1 TEMPO train/dev release. Corpus text was
normalized consistently before saving; the original corpus files were not
modified. This small fixed set is intended for local comparison, not for
estimating performance on the full domain corpus.
