---
name: circuit-sim-lab-report
description: Write Circuit Simulation Laboratory lab reports (course 0714-EEE-2102, University of Dhaka) as LaTeX-compiled PDFs that follow the official Report Writing Guideline — cover page + task sheet front matter, sections 1.1 Objective through 1.10 Reference, A4 Times New Roman 12 pt, 1.5 line spacing, IEEE references, % error analysis. Use whenever the user mentions writing, drafting, starting, continuing, fixing, or compiling a lab report or experiment report for this Circuit Simulation Lab, or wants simulation results (LTspice/PSpice exports, waveforms, schematics) turned into the report PDF — even if they never say the word "report".
---

# Circuit Simulation Lab Reports

The deliverable is one PDF per experiment: the instructor's cover page and task sheet as front matter, followed by the ten guideline sections (1.1 Objective → 1.10 Reference), typeset per the official formatting rules. Source lives in `<expN>/Lab_report_latex/report.tex`; the compiled `report.pdf` sits next to it. The authoritative spec is `Report Writing Guideline.pdf` at the workspace root.

Read `references/guideline.md` (the guideline digest) before drafting any section, and `references/latex-setup.md` when preamble or compile questions come up.

## Step 1 — Gather inputs

From the experiment folder `<expN>/`:

- `Intructions.txt` (note the spelling) — the instructor's per-experiment instructions
- `Cover_Page.pdf` — filled cover page → report page 1
- `2nd_page.pdf` — task sheet (CO–PO table + Expected Report Contents) → report page 2
- `Graphs/` — simulator `.txt` exports, rendered `.png`/`.pdf` figures, `plot_expN_*.py` scripts
- a schematic export/screenshot for section 1.6 — must be the *actual* simulation schematic, not a redrawn drawing

If the schematic export is missing, ask the user for it. When working autonomously, put a clearly labeled placeholder with a textual description of each circuit in 1.6 and a note that the real export must be inserted — never substitute a hand-redrawn diagram — and flag the gap in your final summary.

Name the simulator the way the course materials do. This course's cover page, task sheets, and manual say **PSpice** — use "PSpice" in report prose even when the raw exports come from LTspice or another SPICE tool; refer to the data files themselves as SPICE exports.

At the workspace root: `Manual.pdf` (theory/procedures for the experiments) and `Cover Page of Report.docx` (blank cover template).

Extract the text of `2nd_page.pdf` with `pdftotext -layout` and work from its Expected Report Contents: it says which guideline sections apply this time (e.g. whether 1.3 Design is needed) and what the analysis must cover (time constants, % error, parameter variation, …).

If `Cover_Page.pdf` or `2nd_page.pdf` is missing, ask the user for it instead of inventing front matter — those pages carry per-experiment dates, group members, and assigned tasks that cannot be derived from anything else in the workspace.

## Step 2 — Section map (fixed order)

Header block: experiment number + the full experiment name (copy it exactly from the task sheet/cover).

1. **Objective** — 2–4 items, action verbs (design, simulate, analyze, verify, investigate), naming the circuit/software concepts; no theory.
2. **Theory** — principle of operation, governing equations (numbered), parameter definitions, expected waveform behavior, assumptions; must explain *why* the circuit behaves as observed, not reproduce the textbook.
3. **Design** — only if the experiment requires component/parameter selection; otherwise omit the section entirely.
4. **List of Components** — table with Sl. No. / Component–Instrument / Specification–Value / Quantity; include the simulation software.
5. **Working Procedure** — numbered steps, reproducible by another student; no results, no conclusions.
6. **Working Diagram** — the actual simulation schematic export, labeled.
7. **Calculation** — given values → equations → substitution → intermediate steps → final value with units; show the steps, not just the answer.
8. **Result Analysis** — figures, simulated vs theoretical comparison tables, % error, parameter-variation results.
9. **Discussion** — analysis, not a repeat of 1.8: why the waveforms look as they do, why simulated differs from theoretical, parameter effects, model limitations, practical significance.
10. **Reference** — IEEE style, numbered in order of first citation, access dates for online sources.

Which sections apply is decided by the task sheet's Expected Report Contents — but read it as an abridged *content checklist*, not the final outline (exp1's sheet lists 8 items with no Design section, and folds calculations under Result Analysis). The guideline's outline still governs the structure: keep Calculation as its own section even when the sheet folds it into Result Analysis, and delete the Design section entirely when the experiment involves no design work. Deleting a section block lets the template renumber the rest gaplessly (List of Components becomes 1.3) — that's the default; a numbering gap looks like an error. If the instructor grades strictly against the guideline's fixed labels instead, keep the guideline numbers by bumping the section counter over the gap (`\setcounter{section}{3}` before List of Components when Design is omitted).

The % error definition, per-section details, and the guideline's own examples are in `references/guideline.md`.

## Step 3 — Produce the LaTeX

1. Copy `assets/template.tex` from this skill to `<expN>/Lab_report_latex/report.tex` (create the folder if absent). If a `report.tex` already exists there, treat it as the working draft: diff it against the template and bring over only the template's preamble/header improvements, preserving filled content — don't blindly overwrite. Clean out stale build artifacts (`.aux`/`.log`/`.out`, old `pg-*.png` renders) so verification reflects the current build alone. The template already implements every formatting rule — fill and adapt it; don't restructure the section order or numbering.
2. Set `\newcommand{\expnumber}{N}` in the preamble to the experiment number (sections then number N.1 … N.10), and fill the header block (`Experiment No. \expnumber` + the full experiment name copied exactly from the task sheet/cover). Check the front-matter paths (`../Cover_Page.pdf`, `../2nd_page.pdf` relative to `Lab_report_latex/`).
3. Figures: prefer the rendered PNGs in `Graphs/` (`\graphicspath` in the template already points there). If an export has no figure yet, write a `plot_expN_*.py` script in `Graphs/` modeled on `plot_exp1_1.py` (the polished exemplar — files named `*_claude.py` are earlier drafts) — overlaid traces, distinct colors, hollow markers, LTspice-style legend below the axis (`ltspice_legend()`), "Time (ms)" (t × 1e3) and "Voltage (V)" axis labels. Run plot scripts with `MPLBACKEND=Agg` (no display server here). Verify a plot by introspecting the figure inside the script (e.g. assert on line data and axis labels) — image viewing is unreliable in this environment, so don't depend on it.
4. Raw `.txt` exports are the primary data source for sections 1.7 and 1.8 — figures alone aren't enough. Read them with numpy (`np.genfromtxt(..., delimiter="\t", skip_header=1)`; the first column is time in seconds — the existing plot scripts show the exact format) and derive simulated quantities from the waveform data: a time constant is the interval from the switching instant to where the trace reaches 1 − e⁻¹ ≈ 63.2% of its final value on charging (or falls to e⁻¹ ≈ 36.8% on discharging), read off by interpolation; sinusoid steady-state amplitude and phase come from comparing output and input at corresponding zero crossings or peaks. Compute derived values from the data array, never by eyeballing the plot. These measured values feed the comparison tables and % error columns.
5. Compile from `Lab_report_latex/`:

   `env pdflatex -interaction=nonstopmode report.tex`

   Run it twice (labels and references resolve on the second pass). The `env` prefix is mandatory in this sandbox: a direct `pdflatex` call inherits a mangled program name and dies with `I can't find the format file 'ZCode-….AppImage.fmt'`. Seeing that error means `env` was skipped.

## Step 4 — Verify before handing over

- Clean compile: no errors in `report.log`, no undefined references after the second pass, and no Overfull/Underfull box warnings (`grep -cE "Overfull|Underfull" report.log` returns 0). Judge success by `report.log` and the exit code of the pdflatex call itself — piping pdflatex output through `tail`/`head` masks a nonzero exit status as 0.
- `pdfinfo report.pdf`: A4 paper (595 × 842 pts); page count = 2 front-matter pages + body.
- `pdftotext -f 1 -l 2 report.pdf -`: page 1 is the cover, page 2 the task sheet.
- Body text extract shows sections in guideline order (N.1 → N.10, or gapless after omissions); every figure and table is numbered, captioned (figures below, tables above), and referred to in the prose; equations are numbered.
- Fonts: `pdffonts report.pdf` — body pages show TeXGyreTermesX/newtx (Times look-alike); front-matter pages keep the original embedded fonts from the instructor PDFs.
- Completeness: `pdftotext report.pdf - | grep -n "\.\.\."` (and a scan of report.tex for "TODO" / bracketed placeholder notes) must come up empty — a report full of template placeholders passes every check above while being unfinished. Cross-check the body against the task sheet's Expected Report Contents so no required result (e.g. initial-charge calculation, parameter variation) is missing.
- Content pass: objectives use action verbs; the procedure contains no results; the discussion doesn't restate the result tables; % error = (Theoretical − Simulated)/Theoretical × 100; references are IEEE-styled.
- Visual page rendering (`pdftoppm -png -r 60` + Read on the PNGs) may or may not display in a given session — in this model it doesn't, so treat the programmatic checks above as the gate; other agents can view rendered pages, in which case a quick look at 2–3 pages is a reasonable extra.

## Common pitfalls

- Theory section becomes a textbook dump instead of explaining the observed behavior.
- Discussion repeats Result Analysis numbers instead of interpreting them.
- Results or conclusions leak into the Working Procedure.
- % error computed as simulated-over-theoretical, or with the subtraction flipped.
- The simulation software missing from the components table.
- A hand-redrawn schematic instead of the actual simulation export.
