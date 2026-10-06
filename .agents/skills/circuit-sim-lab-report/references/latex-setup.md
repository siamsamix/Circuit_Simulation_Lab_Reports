# LaTeX setup for lab reports

The environment is TeX Live 2026 (Arch Linux) with `pdflatex` at `/usr/sbin/pdflatex`. All packages used by the template are verified present: geometry, newtxtext/newtxmath, setspace, amsmath, graphicx, booktabs, float, caption, siunitx, enumitem, pdfpages, etoolbox, titlesec, sectsty, xcolor, hyperref, tcolorbox.

## Compile command

From `<expN>/Lab_report_latex/`:

```
env pdflatex -interaction=nonstopmode report.tex
```

Run twice so cross-references and labels resolve. Then verify a clean log (no errors, no "undefined references" after the second pass).

Check the exit code of the pdflatex invocation itself — piping its output through `tail`/`head` masks a failed compile as exit status 0 (the pipeline returns the last command's status). The log file is the ground truth either way.

You may see a kpathsea `configuration file texmf.cnf not found` warning scroll by first — harmless and expected in this sandbox; read past it for the actual error.

### Why the `env` prefix is mandatory here

A bare `pdflatex` invocation in this sandbox fails. The signature (after a `texmf.cnf not found` warning — ignore that part): the banner shows `preloaded format=ZCode-3.11.2-linux-x64.AppImage`, kpathsea spawns `mktexfmt ZCode-3.11.2-linux-x64.AppImage.fmt` (which may run for a while first — the failure isn't always immediate), and it ends with:

```
I can't find the format file `ZCode-3.11.2-linux-x64.AppImage.fmt'!
```

The process inherits a mangled program name (argv[0] from the sandbox wrapper), and pdflatex derives its format-file name from that. Prefixing with `env` resets argv[0] to `pdflatex`, and compilation succeeds. Seeing the `ZCode-…AppImage.fmt` error always means `env` was skipped — just rerun with the prefix. This is a sandbox quirk, not a TeX installation problem: the full package tree is intact under `/usr/share/texmf-dist`.

(`kpsewhich` in this shell also misbehaves — it reports every package missing. Ignore it; actual compilation is the ground truth.)

## How the template maps formatting rules → LaTeX

| Guideline rule | Template implementation |
|---|---|
| A4, 1 in margins | `\documentclass[12pt,a4paper]{article}` + `geometry` `margin=1in` |
| Times New Roman 12 pt | `newtxtext, newtxmath` (no TTF embedding needed; PSNFSS-clone via New TX) |
| 1.5 line spacing | `setspace` `\onehalfspacing` |
| 6 pt after each paragraph | `\setlength{\parskip}{6pt plus 1pt}` (plus `parskip` package to suppress indents) |
| Justified text | default in LaTeX |
| Headings bold, left-aligned, 14–16 pt | `titlesec` formatting for `\section`/`\subsection` |
| Bottom-center page numbers | `plain` page style, footer center |
| Figure captions centered, below | `caption` package, `justification=centering` |
| Table captions centered, above | `caption` package (caption placement follows placement key `\begin{table}[H]` + caption above tabular) |
| Equations centered, numbered | `amsmath` (`\begin{equation}`) — centered and numbered by default |
| SI units | `siunitx` — `\SI{10}{\kilo\ohm}`, `\SI{100}{\micro\farad}`, `\SI{1}{\kilo\hertz}` |
| Front matter | `pdfpages`: `\includepdf[pages={1}]{../Cover_Page.pdf}` etc. |
| Numbered sections N.1–N.10 | `\setcounter{section}{N}` before the first `\section` |
| Component tables | `booktabs` (`\toprule`/`\midrule`/`\bottomrule`) |
| Schematic/figures | `graphicx` + `float` `[H]` placement |

## Fonts on other machines

`newtxtext/newtxmath` is the reliable way to get Times New Roman look-alike math+text on TeX Live. If a package is ever missing there, `mathptmx` (PSNFSS, ubiquitous) is the fallback: replace `\usepackage{newtxtext,newtxmath}` with `\usepackage{mathptmx}`. Do not use `times` (deprecated).

## Compile failure triage

- **Missing front-matter PDF** — `\includepdf` fails hard if `../Cover_Page.pdf` doesn't exist. Check `Intructions.txt`-listed paths first; ask the user if a front-matter page is genuinely absent.
- **`I can't find the format file 'ZCode-…fmt'`** (preceded by a `texmf.cnf not found` warning) — rerun with the `env` prefix (see above).
- **Undefined references / "??" in text** — compile a second time.
- **Figure file not found** — the template sets `\graphicspath{{../Graphs/}{./figures/}}`; PNGs from `Graphs/` are referenced by bare filename.
- **Exit status 0 but no PDF produced** — output was piped and the pipeline masked the failure; re-run without a pipe or check `report.log` directly.
- **Orphan files in `Lab_report_latex/`** — old rendered pages (`pg-*.png`), stale `.aux`/`.log` from prior attempts, or scratch images can accumulate from earlier sessions. Harmless to the build, but identify and clean them so the folder holds only `report.tex`, its figures, and fresh build outputs.
