# Provenance and preservation

The author-supplied eight-page manuscript is included unchanged. Its title and author order are used in the repository citation. Older editorial drafts and delivery documents are not included.

Scientific values in `evidence/` are selected from the existing archive, retaining the numerical values, case identities, seeds, assessment failures, and denominators. These files are compact projections, not byte-identical copies of the full archive exports. No thresholds or results were corrected, rescaled beyond explicit percentage-point presentation, or replaced.

`evidence/provenance.json` records the original archive-relative source paths and full-file hashes. `source_records` and record paths inside the evidence identify original archive entries; those entries are not all bundled here. A hash identifies an original input but does not make an omitted file available.

`evidence/source_manifest.json` records the original and published hashes of retained code and prompts. Complete retained modules are copied byte-for-byte. Two modules have packaging-only adaptations:

- `profiling/schemas.py`: the unused orchestration configuration helper is omitted; the original schema and profiling classes are retained.
- `readiness_challenge.py`: unused live-assessor context wiring and its imports are omitted; the original case generation, rules, repair, labels, and metrics are retained.

All retained function and class definitions in those two modules were checked for syntax-tree equality against the originals. Added replay, plotting, verification, and demonstration scripts wrap the saved measurements and retained toolkit; they do not substitute a new research method.

Agent prompts are retained unchanged for inspection. The operational evidence includes all 12 assigned attempts, their completion status, predictions, request and tool counts, token usage, and elapsed time. A recorded pre-request termination is described at that level, consistent with the manuscript. No failed attempt is converted into a success.

The public selection excludes original working directories, virtual environments, caches, full provider transcripts, local account settings, private journals, machine-specific records, duplicate archives, fitted model binaries, and row-level datasets. New runs and acquired inputs belong in ignored local directories.
