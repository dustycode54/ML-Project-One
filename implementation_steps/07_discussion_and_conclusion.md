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

- [ ] Discussion covers all four required topics.
- [ ] Conclusion covers findings, lessons learned, and challenges.
- [ ] AI usage statement names the tools used and distinguishes AI-assisted work from independent work.
- [ ] Final report contains all 10 graded sections.
- [ ] The repo is reproducible by a reader following the README.

## Artifacts

- `reports/final_report.pdf` (or `.md` / `.docx`)
- Updated `README.md`
- `requirements.txt`
