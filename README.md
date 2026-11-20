# Fixed-Point Iteration Solver

A polished, modular Python terminal application demonstrating the **Simple Fixed-Point Iteration** method (also known as *one-point iteration* or *successive substitution*) for a Numerical Methods final project presentation.

---

## Features

- **Two Interactive Modes**:
  1. **Lecture Example Preset** — Instant demo of $f(x) = e^{-x} - x$, $g(x) = e^{-x}$, $x_0 = 0$, $\epsilon_s = 0.5\%$, `max_iter = 10`, with true root $x_{\text{true}} = 0.56714329$.
  2. **Custom Problem Mode** — Enter any single-variable expression in natural math syntax (`2x^2 - 5x + 1`, `exp(-x) - x`, `cos(x) - x`, `sqrt(2x + 3)`, `ln(x + 2)`). Leave $g(x)$ blank to automatically apply $g(x) = f(x) + x$, and leave $x_{\text{true}}$ blank to automatically compute a high-precision reference root via `sympy.nsolve`.
- **Symbolic Convergence Diagnostics**:
  - Automatically differentiates $g(x)$ with SymPy to obtain $g'(x)$.
  - Evaluates $g'(x_0)$, $g'(x_{\text{final}})$, and $g'(x_{\text{true}})$ to classify behavior into the four standard cases:
    - $0 \le g'(x) < 1$: **Convergent (Monotonic)**
    - $-1 < g'(x) \le 0$: **Convergent (Oscillatory / Spiral)**
    - $g'(x) > 1$: **Divergent (Monotonic)**
    - $g'(x) < -1$: **Divergent (Oscillatory)**
  - Includes a **Divergence Guard** that halts cleanly on $|x_i| > 10^{12}$, `NaN`, `Inf`, division by zero, or complex numbers.
- **Presentation-Ready Output**:
  - **Presentation Walkthrough Mode** (pauses after each step for live narration, with `'s'` to skip remaining pauses) and **Full Instant Report Mode**.
  - Bordered step-by-step iteration panels showing current value, exact formula substitution, computed estimate, explicit $\epsilon_a$, explicit $\epsilon_t$, linear error ratio $|\epsilon_t|_i / |\epsilon_t|_{i-1}$, and stopping check.
  - Clean `rich` summary table and final boxed verdict comparing observed $|\epsilon_t|_n / |\epsilon_t|_{n-1}$ against theoretical $|g'(x_{\text{final}})|$.
- **In-Terminal Unicode/Braille Visualizations (`plotext`)**:
  - **Graph A** — Original Function Root Plot ($y = f(x)$ and $y = 0$ with highlighted root).
  - **Graph B** — Two-Curve Intersection & Cobweb Plot ($f_1(x) = x$, $f_2(x) = g(x)$, and step-by-step cobweb trajectory).
  - **Graph C** — Relative Error Decay Plot ($\epsilon_a$ and $\epsilon_t$ vs. iteration $i$, with tolerance reference line $\epsilon_s$).

---

## Project Structure

```text
fixed-point-iteration-solver/
├── main.py            # CLI entry point, --demo flag, & interactive menu loop
├── solver.py          # Core fixed-point engine, SymPy parser, convergence & error analysis
├── visualizer.py      # Terminal ASCII/Unicode Braille graphs using plotext
├── ui.py              # ASCII banner, Rich panels, tables, and validated input prompts
├── requirements.txt   # Python dependencies (sympy, rich, plotext)
├── .gitignore         # Excludes .venv/, __pycache__/, and IDE files
└── README.md          # Project documentation & User Guide
```

---

## Setup & Installation

Requires **Python 3.10+** and a terminal supporting UTF-8 / Unicode characters (100–120 columns recommended).

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Quick Non-Interactive Demo

Run the Lecture Example Preset report and all three terminal plots directly without prompts:

```bash
python main.py --demo
```

---

## User Guide

### 1. Navigating the Main Menu

When you run `python main.py`, you are greeted by the Main Menu with four options:

| Option | Action | Description |
|:---:|---|---|
| **`1`** | **Lecture Example Preset** | Runs the textbook problem $f(x) = e^{-x} - x$ with $g(x) = e^{-x}$, $x_0 = 0$, $\epsilon_s = 0.5\%$, `max_iter = 10`, and $x_{\text{true}} = 0.56714329$. |
| **`2`** | **Custom Problem Mode** | Lets you enter your own $f(x)$, $g(x)$, initial guess, stopping criteria, and optional true root. |
| **`3`** | **Method & Convergence Reference** | Displays a quick mathematical cheat-sheet with all error formulas and the four $g'(x)$ convergence cases. |
| **`0`** | **Exit** | Closes the application cleanly. |

---

### 2. Using Custom Problem Mode

When you select **`2`**, the app guides you through six validated prompts:

1. **Original Function `f(x)`** — Enter an expression in `x` (without `= 0`, though `f(x) = ...` prefixes are automatically stripped).
   - **Supported syntax:**
     - Arithmetic & powers: `+`, `-`, `*`, `/`, `^` or `**` (e.g., `x^2 - 2*x - 3`)
     - Implicit multiplication: `2x`, `3x^2`, `2(x + 1)`
     - Constants & functions: `e`, `pi`, `exp(...)`, `ln(...)`, `log(...)`, `log10(...)`, `sqrt(...)`, `cbrt(...)`, `abs(...)`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`
   - **Example inputs:** `exp(-x) - x`, `e^(-x) - x`, `cos(x) - x`, `2x^2 - 5x + 1`

2. **Rearranged Function `g(x)`** — Enter `g(x)` such that $x = g(x)$.
   - **Tip:** Press **Enter** (leave blank) to automatically use $g(x) = f(x) + x$. The app will display the algebraic transformation clearly.

3. **Initial Guess `x0`** — Starting value $x_0$ (default: `0.0`).

4. **Error Tolerance `es` (%)** — Approximate relative error threshold $\epsilon_s$ in percent (default: `0.5`). Iteration stops early once $\epsilon_a < \epsilon_s$.

5. **Maximum Iterations** — Upper bound between `1` and `500` (default: `20`).

6. **Known True Root `x_true`** *(Optional)* — Enter a known root if available, or press **Enter** to let `sympy.nsolve` automatically find a high-precision reference root near your iterates so $\epsilon_t$ and $|\epsilon_t|_i / |\epsilon_t|_{i-1}$ can still be shown.

---

### 3. Choosing a Display Mode

Before solving, choose how steps are presented:

- **`1` — Presentation Walkthrough Mode**: Pauses after every iteration panel so you can explain the substitution, $\epsilon_a$, $\epsilon_t$, and stopping check live.
  - Press **Enter** to advance to the next iteration.
  - Type **`s`** + **Enter** at any time to skip remaining pauses and jump to the rest of the steps.
- **`2` — Full Instant Report Mode** *(default)*: Prints all iteration panels, the summary table, and the final verdict at once.

---

### 4. Viewing the Terminal Graphs

After the Final Convergence Verdict, the **Terminal Visualizations** menu lets you view plots without scrolling off-screen:

- **`1` — View All Graphs Sequentially**: Steps through Graph A → Graph B → Graph C with a pause between each.
- **`2` — Graph A (Original Function Root Plot)**: Shows $y = f(x)$, the horizontal line $y = 0$, the starting point $x_0$, and the estimated root.
- **`3` — Graph B (Two-Curve Intersection & Cobweb Plot)**: Plots $f_1(x) = x$ and $f_2(x) = g(x)$ with the step-by-step cobweb trajectory $(x_0, 0) \to (x_0, g(x_0)) \to (x_1, x_1) \to \dots$
- **`4` — Graph C (Error Decay Plot)**: Plots $\epsilon_a$ (%) and $\epsilon_t$ (%) across iterations against the $\epsilon_s$ tolerance line.
- **`0` — Return to Main Menu**: Start another problem or exit.

---

## Mathematical Formulation

Given $f(x) = 0$, rearrange into the fixed-point form $x = g(x)$ and iterate:

$$x_{i+1} = g(x_i), \quad i = 0, 1, 2, \dots$$

- **Approximate Percent Relative Error:**
  $$\epsilon_a = \left|\frac{x_{i+1} - x_i}{x_{i+1}}\right| \times 100\%$$
- **True Percent Relative Error:**
  $$\epsilon_t = \left|\frac{x_{\text{true}} - x_i}{x_{\text{true}}}\right| \times 100\%$$
- **Linear Convergence Error Ratio** (demonstrating $E_{i+1} \approx g'(\xi) E_i$):
  $$\frac{|\epsilon_t|_i}{|\epsilon_t|_{i-1}} \approx |g'(x^*)|$$

---

## Lecture Example Verification Summary

For $f(x) = e^{-x} - x$, $g(x) = e^{-x}$, $x_0 = 0$, $x_{\text{true}} = 0.56714329$:

| $i$ | $x_i$ | $\epsilon_a\ (\%)$ | $\epsilon_t\ (\%)$ | $\|\epsilon_t\|_i / \|\epsilon_t\|_{i-1}$ |
|---:|---:|---:|---:|---:|
| 0 | `0.000000` | — | `100.0000` | — |
| 1 | `1.000000` | `100.0000` | `76.3223` | `0.7632` |
| 2 | `0.367879` | `171.8282` | `35.1347` | `0.4603` |
| 3 | `0.692201` | `46.8536` | `22.0504` | `0.6276` |
| 10 | `0.564879` | `1.1089` | `0.3992` | `0.5660` |
