# Ticket 07 — Discussion, Conclusion, and Final Report

| Field | Value |
|-------|-------|
| **Step** | 7 of 7 |
| **Depends on** | Tickets [01](01_data_collection.md)–[06](06_visualization.md) |
| **Blocks** | Final submission |
| **Report sections** | Conclusion (6%), AI Usage Statement (4%), plus final assembly of all sections |
| **Deliverable weight** | Final Report (50%), GitHub Repository (10%) |

## Summary

Synthesize the findings into the discussion and conclusion, write the AI usage statement, assemble the final report, and get the repository ready for submission.

## Tasks

### Discussion
- [ ] **Most important factors** affecting vehicle value (from [Ticket 05](05_depreciation_analysis.md)).
- [ ] **Major drivers of depreciation:** age vs. mileage vs. brand/model vs. condition indicators.
- [ ] **Challenges in data collection and integration:** scraping obstacles, schema mismatches, MSRP matching.
- [ ] **Limitations** of the dataset and models: listing price ≠ sale price, regional/time bias, trim-level MSRP gaps, sample size, scope limited to a few makes.

### Conclusion
- [ ] Summary of project findings.
- [ ] What was learned through the project.
- [ ] Challenges faced, and what you would do differently or extend in future work.

### AI Usage Statement
- [ ] List the AI tools used (e.g. Claude Code) and what each was used for.
- [ ] State clearly which parts of the work were completed independently.

### Final Report Assembly
- [ ] Include every graded section:

  | Section | Weight | Source ticket |
  |---------|--------|---------------|
  | Project Description | 5% | — |
  | Data Collection with discussion | 5% | 01 |
  | Preprocessing and Integration with discussion | 5% | 02 |
  | Feature Engineering with discussion | 5% | 03 |
  | Model Development with discussion | 5% | 04 |
  | Evaluation Results with discussion | 5% | 04 |
  | Depreciation Analysis | 5% | 05 |
  | Visualizations with discussion | 5% | 06 |
  | Conclusion | 6% | 07 |
  | AI Usage Statement | 4% | 07 |

### Repository Readiness
- [ ] `README.md`: project description, repo structure, setup (`requirements.txt` / environment), and how to reproduce each step end to end.
- [ ] Raw per-source data files and the integrated dataset are present (or linked, if too large).
- [ ] Code is clean: modular, commented where needed, no dead code, no hard-coded absolute paths.
- [ ] Notebooks run top to bottom without errors.

## Acceptance Criteria

- [x] Discussion covers all four required topics.
- [x] Conclusion covers findings, lessons learned, and challenges.
- [ ] AI usage statement names the tools used and distinguishes AI-assisted work from independent work. *(Drafted; the author must complete the bracketed parts.)*
- [x] Final report contains all 10 graded sections.
- [x] The repo is reproducible by a reader following the README.

## Artifacts

- `reports/final_report.pdf` (or `.md` / `.docx`)
- Updated `README.md`
- `requirements.txt`

## Implementation Notes

**Source and build.** The report source is [reports/final_report.md](../reports/final_report.md). `python -m src.report` ([src/report.py](../src/report.py)) renders it to `reports/final_report.pdf` (21 pages, letter size):
- *Markdown to HTML.* python-markdown converts the source. Each `![Figure N. caption](figures/…svg)` becomes a captioned figure.
- *HTML to PDF.* Headless Chrome prints the HTML with a print stylesheet (page numbers, no breaks inside figures or table rows). Edge and Chromium also work, and `CHROME=/path/to/browser` overrides the lookup.
- *Rebuilding.* Edit the Markdown and re-run the command. If upstream results change, re-run `src.plots` first.

**Sections.**

| Section | Report § | Built from |
|---------|----------|------------|
| Project Description | 1 | Overview, scope, requirements checklist |
| Data Collection with discussion | 2 | Ticket 01 notes, `collection_log.json` |
| Preprocessing and Integration with discussion | 3 | Ticket 02 notes, `eda_summary.md` |
| Feature Engineering with discussion | 4 | Ticket 03 notes, `text_features.md`, `encoding_comparison.csv` |
| Model Development with discussion | 5 | Ticket 04 notes |
| Evaluation Results with discussion | 6 | `model_results.md`, `cross_source.csv`, residuals |
| Depreciation Analysis | 7 | Ticket 05 notes, `depreciation_analysis.md` |
| Visualizations with discussion | 8 | Ticket 06 figures, one caption each |
| Conclusion (discussion of factors, drivers, challenges, limitations; findings, lessons, future work) | 9 | Synthesis |
| AI Usage Statement | 10 | **Draft with placeholders, for the author to complete** |

**Numbers added during this ticket.**
- *Ratios above 1.0.* 56 listings (2.6%) have a retention ratio above 1.0. 39 of them are Toyotas (20 Tacomas), and most are 2021–24 model years.
- *Worst test error.* The largest HistGB test residual (−0.74) is an owner-listed 2024 Tacoma at $5,500 described as "car won't start".

**AI usage statement.** Section 10 lists only what the repository shows: Claude Code was used, and the `.claude/settings.json` permissions show it ran the pipeline. The parts done independently, and how AI output was verified, are bracketed placeholders, because only the author can state them. Complete them and re-run `python -m src.report` before submitting.
