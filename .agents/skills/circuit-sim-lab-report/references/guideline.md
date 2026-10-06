# Report Writing Guideline — digest

Digest of `Report Writing Guideline.pdf` (workspace root), the authoritative spec for every Circuit Simulation Lab report. Workspace convention from the instructor's `Intructions.txt`: page 1 of the PDF is `Cover_Page.pdf`, page 2 is `2nd_page.pdf`, then the body below.

## Header

- **Experiment No.** — the assigned experiment number.
- **Name of the Experiment** — the complete title of the experiment.

## 1.1 Objective

State 2–4 specific objectives.

- Begin with action words: *to design, to simulate, to analyze, to verify, to investigate*.
- Mention the main circuit/software concepts being studied.
- Avoid lengthy theoretical explanations here.

Guideline example (half-wave rectifier):

> i. To design and simulate a half-wave rectifier circuit.
> ii. To observe the input and output waveforms.
> iii. To determine the DC output voltage and ripple characteristics.
> iv. To compare the simulated result with the theoretical result.

## 1.2 Theory

The theoretical background needed to understand the experiment. Include, where applicable:

- Principle of operation
- Relevant circuit concepts
- Governing equations
- Definitions of important parameters
- Expected waveform behavior
- Important assumptions

All important equations properly written and numbered if necessary. Guideline example:

> V_DC = V_m / π  … (1)

where V_m is the peak value of the input voltage.

The theory should explain **why the circuit behaves as observed**, not simply reproduce textbook material.

## 1.3 Design (if applicable)

Include only when the experiment requires circuit/component design or parameter selection:

- Circuit specifications
- Design equations
- Component-value calculations
- Selected component values
- Design constraints or assumptions

Guideline example: f_c = 1/(2πRC) … (2), then show how R and C were selected to obtain the desired cutoff frequency.

## 1.4 List of Components

A table of all components, instruments, and software used:

| Sl. No. | Component/Instrument | Specification/Value | Quantity |
|---|---|---|---|
| 1 | Resistor | 1 kΩ | 1 |
| 2 | Capacitor | 10 µF | 1 |
| 3 | Diode | 1N4007 | 1 |
| 4 | Function Generator | Sinusoidal source | 1 |
| 5 | Oscilloscope | 2-channel | 1 |
| 6 | Simulation Software | LTspice | 1 |

## 1.5 Working Procedure

Write the procedure in numbered steps. It must be sufficiently detailed that another student can reproduce the simulation. **Avoid writing results or conclusions in this section.**

## 1.6 Working Diagram

Insert the actual simulated circuit schematic. The diagram should clearly show:

- Component symbols and component values
- Input/source and ground
- Measurement points, and voltage/current probes if used
- Relevant circuit labels

For simulation experiments, use a screenshot/export of the actual simulation schematic rather than drawing it separately.

## 1.7 Calculation

Show all necessary calculations based on the theoretical equations and/or simulation data:

- Given values
- Required equations
- Substitution
- Intermediate steps
- Final calculated values with units

Calculations should show the steps, not just provide the final answer.

## 1.8 Result Analysis

Present and analyze the simulation results. May include:

- Simulation waveforms/graphs
- Measured/simulated values
- Theoretical values
- Percentage error:

  % Error = (Theoretical − Simulated) / Theoretical × 100

- Comparison tables
- Variation of output with input/component parameters

## 1.9 Discussion

Discuss the experiment in a concise but analytical manner. Address:

- Whether the expected behavior was observed
- Agreement between theoretical and simulated results
- Reasons for differences/errors
- Effect of changing circuit parameters
- Important waveform characteristics
- Practical significance of the result
- Limitations of the simulation/model, if relevant

**Do not simply repeat the Result Analysis.** A good discussion answers:

- Why did the simulated waveform have this shape?
- Why is the simulated value different from the theoretical value?
- What happens if a particular component value is increased or decreased?
- What does the result tell us about the circuit?

## 1.10 Reference

List all sources in IEEE referencing style. Numbered consecutively in the order first cited in the text; each entry must contain sufficient bibliographic information to identify and access the source; include the access date for online sources.

Guideline examples:

> [1] A. S. Sedra and K. C. Smith, *Microelectronic Circuits*, 7th ed. New York, NY, USA: Oxford University Press, 2015.
> [2] A. Author, B. Author, and C. Author, "Title of the paper," *Journal Name*, vol. XX, no. X, pp. XX–XX, Year.

## Formatting rules (all reports)

- Paper size: A4
- Font: Times New Roman
- Font size: 12 pt main text
- Heading size: 14–16 pt depending on heading level
- Text alignment: justified
- Line spacing: 1.5
- Paragraph spacing: 6 pt after each paragraph
- Margins: 1 inch on all sides
- Page numbers: bottom-center or bottom-right
- Section headings and subheadings: bold, left-aligned
- Figure captions: centered, below the figure
- Table captions: centered, above the table
- Equations: centered, clearly separated from the main text
- Units: SI units consistently throughout
- Figures and tables: numbered consecutively (Figure 1, Table 1) and referred to in the text
- Keep the report concise; include only relevant information
