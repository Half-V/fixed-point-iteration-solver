"""Rich-styled terminal UI components for the Fixed-Point Iteration Solver.

Provides the ASCII art header banner, interactive menus, graceful input
validation, step-by-step iteration panels, summary tables, final verdict
cards, and a quick mathematical reference sheet. Designed to render cleanly
within standard 100-120 column terminals.
"""

from __future__ import annotations

import math
import sys
from typing import Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from solver import (
    ConvergenceAnalysis,
    IterationRecord,
    LECTURE_PRESET,
    ParsedProblem,
    SolverResult,
    build_problem,
    format_expr_readable,
    format_number,
    parse_math_expression,
)

_SUBSCRIPT_MAP = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")

_BANNER_ART = r"""
╔═╗╦ ╗ ╔ ╔═╗╦═╗  ╔═╗╔═╗╦╔╗╔╔╦╗  ╦╔╦╗╔═╗╦═╗╔═╗╔╦╗╦╔═╗╔╗╔  ╔═╗╔═╗╦  ╦  ╦╔═╗╦═╗
╠╣ ║ ╚╦╝ ╠═ ║ ║──╠═╝║ ║║║║║ ║   ║ ║ ╠═ ╠╦╝╠═╣ ║ ║║ ║║║║  ╚═╗║ ║║  ╚╗╔╝╠═ ╠╦╝
╚  ╩ ╔╩╗ ╚═╝╩═╝  ╩  ╚═╝╩╝╚╝ ╩   ╩ ╩ ╚═╝╩╚═╩ ╩ ╩ ╩╚═╝╝╚╝  ╚═╝╚═╝╩═╝ ╚╝ ╚═╝╩╚═
""".strip("\n")


def configure_utf8_console() -> Console:
    """Safely configure UTF-8 output on Windows and return a width-safe Rich Console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    probe = Console()
    # Keep layout crisp inside 100-120 col terminals without stretching too wide
    effective_width = max(84, min(probe.width, 114))
    return Console(width=effective_width, highlight=False)


def _sub(value: int) -> str:
    """Convert an integer to Unicode subscript digits (e.g., 10 -> ₁₀)."""
    return str(value).translate(_SUBSCRIPT_MAP)


def _badge(label: str, severity: str) -> str:
    """Render a color-coded pill badge using consistent palette rules."""
    palette = {
        "green": "bold black on bright_green",
        "yellow": "bold black on yellow",
        "red": "bold white on red",
        "cyan": "bold black on cyan",
    }
    style = palette.get(severity, "bold white on blue")
    return f"[{style}] {label} [/{style}]"


def _fmt_pct(val: Optional[float], decimals: int = 4) -> str:
    """Format a percentage cleanly for tables and step cards."""
    if val is None or not math.isfinite(val):
        return "—"
    if val >= 1e5:
        return f"{val:.3e}%"
    return f"{val:.{decimals}f}%"


def _fmt_ratio(val: Optional[float]) -> str:
    """Format an error ratio |ε_t|_i / |ε_t|_{i-1} cleanly."""
    if val is None or not math.isfinite(val):
        return "—"
    if val >= 1e4:
        return f"{val:.3e}"
    return f"{val:.4f}"


def print_banner(console: Console) -> None:
    """Display the Unicode box-drawn ASCII art application banner."""
    banner_body = Text()
    banner_body.append(_BANNER_ART + "\n", style="bold cyan")
    banner_body.append(
        "Simple Fixed-Point Iteration  •  One-Point / Successive Substitution Method\n",
        style="bold white",
    )
    banner_body.append(
        "Numerical Methods Project  │  SymPy Symbolic Engine  │  Plotext Visuals",
        style="dim cyan",
    )

    console.print()
    console.print(
        Panel(
            Align.center(banner_body),
            box=box.DOUBLE,
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )


def show_main_menu(console: Console) -> str:
    """Display the interactive startup menu and return the user's validated choice."""
    menu_table = Table.grid(padding=(0, 2))
    menu_table.add_column(style="bold cyan", justify="right", width=5)
    menu_table.add_column(style="bold white")
    menu_table.add_column(style="dim")

    menu_table.add_row(
        "[1]",
        "Lecture Example Preset",
        "f(x) = e^(-x) - x,  g(x) = e^(-x),  x₀ = 0,  εₛ = 0.5%,  max 10 iter",
    )
    menu_table.add_row(
        "[2]",
        "Custom Problem Mode",
        "Enter custom f(x), optional g(x) (defaults to f(x) + x), x₀, εₛ, max_iter",
    )
    menu_table.add_row(
        "[3]",
        "Method & Convergence Reference",
        "Formulas, error definitions, and the 4 g'(x) convergence cases",
    )
    menu_table.add_row(
        "[0]",
        "Exit",
        "Close the application",
    )

    console.print(
        Panel(
            menu_table,
            title="[bold cyan]Main Menu[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2),
        )
    )

    while True:
        choice = console.input("[bold cyan]Select an option [1/2/3/0]: [/bold cyan]").strip()
        if choice in {"1", "2", "3", "0"}:
            return choice
        console.print("[red]Invalid choice.[/red] Please enter 1, 2, 3, or 0.")


def show_method_reference(console: Console) -> None:
    """Display a concise presentation cheat-sheet for Simple Fixed-Point Iteration."""
    ref = Table.grid(padding=(0, 2))
    ref.add_column(style="bold cyan", width=26)
    ref.add_column(style="white")

    ref.add_row("Goal", "Find a root x* where f(x*) = 0 by rewriting as x = g(x).")
    ref.add_row("Iteration Formula", "xᵢ₊₁ = g(xᵢ),   for i = 0, 1, 2, ...")
    ref.add_row("Approx. Relative Error", "εₐ = |(xᵢ₊₁ - xᵢ) / xᵢ₊₁| × 100%")
    ref.add_row("True Relative Error", "εₜ = |(x_true - xᵢ) / x_true| × 100%")
    ref.add_row("Linear Error Propagation", "Eᵢ₊₁ ≈ g'(ξ) · Eᵢ   ⇒   |εₜ|ᵢ / |εₜ|ᵢ₋₁ ≈ |g'(x*)|")

    cases = Table(
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
        header_style="bold cyan",
        expand=True,
    )
    cases.add_column("Derivative Condition", style="bold white", width=22)
    cases.add_column("Classification", width=34)
    cases.add_column("Visual Cobweb Pattern")

    cases.add_row(
        "0 ≤ g'(x) < 1",
        "[bold green]Convergent (Monotonic)[/bold green]",
        "Staircase stepping steadily from one side toward x*",
    )
    cases.add_row(
        "-1 < g'(x) ≤ 0",
        "[bold green]Convergent (Oscillatory / Spiral)[/bold green]",
        "Inward cobweb spiral alternating around x*",
    )
    cases.add_row(
        "g'(x) > 1",
        "[bold red]Divergent (Monotonic)[/bold red]",
        "Staircase fleeing monotonically away from x*",
    )
    cases.add_row(
        "g'(x) < -1",
        "[bold red]Divergent (Oscillatory)[/bold red]",
        "Outward spiral oscillating with growing amplitude",
    )

    console.print()
    console.print(
        Panel(
            ref,
            title="[bold cyan]Simple Fixed-Point Iteration — Mathematical Summary[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            cases,
            title="[bold cyan]Sufficient Convergence Criterion: |g'(x)| < 1 Near the Root[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.input("[dim]Press Enter to return to the Main Menu...[/dim]")


def _prompt_float(
    console: Console,
    prompt_label: str,
    default: Optional[float] = None,
    allow_blank: bool = False,
    min_value: Optional[float] = None,
) -> Optional[float]:
    """Prompt for a finite float with graceful retry on invalid input."""
    while True:
        raw = console.input(f"[bold cyan]{prompt_label}[/bold cyan]").strip()
        if not raw:
            if allow_blank:
                return default
            if default is not None:
                return default
            console.print("[red]Input required.[/red] Please enter a valid real number.")
            continue

        try:
            value = float(raw)
        except ValueError:
            console.print(
                f"[red]'{raw}' is not a valid number.[/red] Try values like 0, 1.5, or -2.25."
            )
            continue

        if not math.isfinite(value):
            console.print("[red]Please enter a finite real number.[/red]")
            continue

        if min_value is not None and value < min_value:
            console.print(f"[red]Value must be ≥ {min_value:g}.[/red]")
            continue

        return value


def _prompt_int(
    console: Console,
    prompt_label: str,
    default: int,
    min_value: int = 1,
    max_value: int = 500,
) -> int:
    """Prompt for a bounded integer with graceful retry."""
    while True:
        raw = console.input(f"[bold cyan]{prompt_label}[/bold cyan]").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            console.print(
                f"[red]'{raw}' is not a valid whole number.[/red] "
                f"Enter an integer between {min_value} and {max_value}."
            )
            continue

        if not (min_value <= value <= max_value):
            console.print(
                f"[red]Value must be between {min_value} and {max_value}.[/red]"
            )
            continue

        return value


def prompt_custom_problem(
    console: Console,
) -> tuple[ParsedProblem, float, float, int, Optional[float]]:
    """Interactively collect and validate all inputs for Custom Problem Mode."""
    tips = (
        "[bold cyan]Supported math syntax:[/bold cyan] "
        "[white]exp(-x) - x[/white], [white]e^(-x) - x[/white], "
        "[white]x^2 - 2*x - 3[/white], [white]2x^2 - 5x + 1[/white], "
        "[white]cos(x) - x[/white], [white]sqrt(2x + 3)[/white], [white]ln(x + 2)[/white]\n"
        "[dim]• Implicit multiplication (e.g., 2x) and '^' for exponentiation are supported.\n"
        "• Leave g(x) blank to automatically derive g(x) = f(x) + x.\n"
        "• Leave true root blank to auto-compute a high-precision reference via SymPy nsolve.[/dim]"
    )
    console.print()
    console.print(
        Panel(
            tips,
            title="[bold cyan]Custom Problem Setup[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )

    # 1. Original function f(x)
    while True:
        f_str = console.input("[bold cyan]Enter original function f(x): [/bold cyan]").strip()
        try:
            f_expr = parse_math_expression(f_str)
            console.print(
                f"  [green]✓ Parsed:[/green] [cyan]f(x) = {format_expr_readable(f_expr)}[/cyan]"
            )
            break
        except ValueError as exc:
            console.print(f"  [red]✗ {exc}[/red]")

    # 2. Rearranged function g(x)
    while True:
        g_str = console.input(
            "[bold cyan]Enter rearranged g(x) where x = g(x) [Enter for g(x) = f(x) + x]: [/bold cyan]"
        ).strip()
        try:
            problem = build_problem(f_str, g_str if g_str else None)
            if problem.g_auto_derived:
                transform_text = (
                    f"[white]Original equation:[/white]   [cyan]f(x) = {problem.f_raw} = 0[/cyan]\n"
                    f"[white]Add x to both sides:[/white] [cyan]f(x) + x = x[/cyan]\n"
                    f"[white]Derived g(x):[/white]        [bold green]g(x) = {format_expr_readable(problem.g_expr)}[/bold green]"
                )
                console.print(
                    Panel(
                        transform_text,
                        title="[bold cyan]Automatic Algebraic Transformation[/bold cyan]",
                        box=box.ROUNDED,
                        border_style="cyan",
                        padding=(0, 2),
                    )
                )
            else:
                console.print(
                    f"  [green]✓ Parsed:[/green] [cyan]g(x) = {format_expr_readable(problem.g_expr)}[/cyan]"
                )
            break
        except ValueError as exc:
            console.print(f"  [red]✗ {exc}[/red]")

    # 3. Initial guess x0
    x0 = _prompt_float(console, "Initial guess x₀ [default 0.0]: ", default=0.0)
    assert x0 is not None

    # 4. Stopping tolerance es (%)
    es = _prompt_float(
        console,
        "Error tolerance εₛ in % [default 0.5]: ",
        default=0.5,
        min_value=0.0,
    )
    assert es is not None

    # 5. Max iterations
    max_iter = _prompt_int(
        console,
        "Maximum iterations [default 20, range 1–500]: ",
        default=20,
        min_value=1,
        max_value=500,
    )

    # 6. Optional true root
    true_root = _prompt_float(
        console,
        "Known true root x_true [Enter to auto-compute via SymPy nsolve]: ",
        default=None,
        allow_blank=True,
    )

    return problem, x0, es, max_iter, true_root


def prompt_lecture_preset_options(console: Console) -> tuple[float, int, bool]:
    """Show Lecture Preset details and confirm default textbook run or custom tolerance stop."""
    summary = (
        f"[bold cyan]Function:[/bold cyan]        f(x) = {LECTURE_PRESET['f_str']}\n"
        f"[bold cyan]Rearrangement:[/bold cyan]   g(x) = {LECTURE_PRESET['g_str']}   (from e^(-x) - x = 0 ⇒ x = e^(-x))\n"
        f"[bold cyan]Initial Guess:[/bold cyan]   x₀ = {LECTURE_PRESET['x0']}\n"
        f"[bold cyan]Stopping Rule:[/bold cyan]   εₛ = {LECTURE_PRESET['es']}%,  max_iter = {LECTURE_PRESET['max_iter']} "
        f"[dim](completes all 10 textbook iterations since εₐ,₁₀ ≈ 1.11% > 0.5%)[/dim]\n"
        f"[bold cyan]Known True Root:[/bold cyan] x_true = {LECTURE_PRESET['true_root']}"
    )
    console.print()
    console.print(
        Panel(
            summary,
            title="[bold cyan]Lecture Example Preset (Chapra & Canale Ex. 6.1)[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )
    return float(LECTURE_PRESET["es"]), int(LECTURE_PRESET["max_iter"]), True


def prompt_walkthrough_mode(console: Console) -> bool:
    """Ask whether to run in Presentation Walkthrough Mode or Full Instant Report Mode."""
    mode_table = Table.grid(padding=(0, 2))
    mode_table.add_column(style="bold cyan", width=5)
    mode_table.add_column(style="bold white")
    mode_table.add_column(style="dim")
    mode_table.add_row(
        "[1]",
        "Presentation Walkthrough Mode",
        "Pauses after each iteration for live explanation",
    )
    mode_table.add_row(
        "[2]",
        "Full Instant Report Mode",
        "Prints all iteration steps, summary table, and verdict at once",
    )

    console.print(
        Panel(
            mode_table,
            title="[bold cyan]Output Display Mode[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )

    while True:
        choice = console.input(
            "[bold cyan]Choose display mode [1 = Walkthrough / 2 = Full Report, default 2]: [/bold cyan]"
        ).strip()
        if choice == "" or choice == "2":
            return False
        if choice == "1":
            return True
        console.print("[red]Invalid choice.[/red] Enter 1 or 2.")


def _format_convergence_line(analysis: ConvergenceAnalysis) -> str:
    """Format a single convergence analysis line with badge and numeric g'(x)."""
    pt = format_number(analysis.point_value, 6)
    if analysis.derivative_value is not None:
        val_str = f"g'({analysis.point_label}) = g'({pt}) = {format_number(analysis.derivative_value, 6)}"
    else:
        val_str = f"g'({analysis.point_label}) = g'({pt}) = Undefined"
    return f"{val_str}   {_badge(analysis.classification, analysis.severity)}"


def display_problem_setup(console: Console, result: SolverResult) -> None:
    """Display the problem definition, symbolic g'(x), and pre-iteration convergence check."""
    p = result.problem
    f_display = p.f_raw or format_expr_readable(p.f_expr)
    g_display = format_expr_readable(p.g_expr) if p.g_auto_derived else (p.g_raw or format_expr_readable(p.g_expr))
    gp_display = format_expr_readable(p.g_prime_expr)

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan", width=24)
    info.add_column(style="white")

    info.add_row("Original Function", f"[bold white]f(x) = {f_display}[/bold white]")
    if p.g_auto_derived:
        info.add_row(
            "Rearranged Function",
            f"[bold white]g(x) = f(x) + x = {g_display}[/bold white] [dim](auto-derived)[/dim]",
        )
    else:
        info.add_row("Rearranged Function", f"[bold white]g(x) = {g_display}[/bold white]")
    info.add_row("Symbolic Derivative", f"[bold cyan]g'(x) = {gp_display}[/bold cyan]")
    info.add_row(
        "Parameters",
        f"x₀ = {format_number(result.x0, 6)}   │   εₛ = {result.es:g}%   │   max_iter = {result.max_iter}",
    )

    if result.true_root is not None:
        info.add_row(
            "Reference True Root",
            f"[bold green]x_true = {result.true_root:.8f}[/bold green] [dim]({result.true_root_source})[/dim]",
        )
    else:
        info.add_row(
            "Reference True Root",
            "[yellow]Unavailable (SymPy nsolve did not locate a real root nearby)[/yellow]",
        )

    if result.pre_analysis is not None:
        info.add_row("Pre-Iteration Check", _format_convergence_line(result.pre_analysis))
        info.add_row("", f"[dim]{result.pre_analysis.explanation}[/dim]")

    # Initial baseline i = 0
    init_rec = result.records[0]
    baseline_txt = f"x₀ = {format_number(result.x0, 6)}"
    if init_rec.et is not None:
        baseline_txt += (
            f"   │   εₜ,₀ = |({result.true_root:.8f} - {format_number(result.x0, 6)}) "
            f"/ {result.true_root:.8f}| × 100% = [bold]{_fmt_pct(init_rec.et)}[/bold]"
        )
    info.add_row("Initial Baseline (i = 0)", baseline_txt)

    console.print()
    console.print(
        Panel(
            info,
            title="[bold cyan]Problem Setup & Pre-Iteration Convergence Analysis[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _render_iteration_panel(
    rec: IterationRecord,
    prev_et: Optional[float],
    result: SolverResult,
) -> Panel:
    """Build a crisp bordered Panel for a single iteration i >= 1."""
    i = rec.i
    prev_idx = _sub(i - 1)
    curr_idx = _sub(i)

    assert rec.x_prev is not None
    x_prev_str = format_number(rec.x_prev, 6)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", width=22)
    grid.add_column(style="white")

    # 1. Current value x_{i-1}
    grid.add_row("1. Current Value", f"x{prev_idx} = [bold white]{x_prev_str}[/bold white]")

    # 2. Exact formula substitution
    sub_expr = rec.substitution_str or f"g({x_prev_str})"
    grid.add_row(
        "2. Substitution",
        f"x{curr_idx} = g(x{prev_idx}) = [cyan]{sub_expr}[/cyan]",
    )

    if rec.diverged:
        # 3. Divergence alert
        if rec.x_curr is not None:
            grid.add_row(
                "3. Computed Value",
                f"x{curr_idx} = [bold red]{format_number(rec.x_curr, 6)}[/bold red]  {_badge('DIVERGED', 'red')}",
            )
        else:
            grid.add_row(
                "3. Computed Value",
                f"[bold red]Evaluation halted[/bold red]  {_badge('DIVERGED', 'red')}",
            )
        grid.add_row(
            "4. Divergence Guard",
            f"[red]{rec.divergence_reason}[/red]",
        )
        grid.add_row(
            "5. Action Taken",
            "[yellow]Stopped iteration safely to prevent overflow / invalid arithmetic.[/yellow]",
        )
        return Panel(
            grid,
            title=f"[bold red]Iteration {i}  (Computing x{curr_idx} from x{prev_idx}) — DIVERGED[/bold red]",
            box=box.ROUNDED,
            border_style="red",
            padding=(0, 2),
        )

    assert rec.x_curr is not None
    x_curr_str = format_number(rec.x_curr, 6)

    # 3. Computed next estimate
    grid.add_row(
        "3. Computed Value",
        f"x{curr_idx} = [bold bright_white]{x_curr_str}[/bold bright_white]",
    )

    # 4. Explicit ε_a calculation
    if rec.ea is not None:
        ea_formula = (
            f"εₐ = |({x_curr_str} - {x_prev_str}) / {x_curr_str}| × 100% "
            f"= [bold {'green' if rec.met_tolerance else 'white'}]{_fmt_pct(rec.ea)}[/bold {'green' if rec.met_tolerance else 'white'}]"
        )
    else:
        ea_formula = f"[yellow]{rec.ea_note or 'Undefined (division by zero)'}[/yellow]"
    grid.add_row("4. Approx. Error εₐ", ea_formula)

    # 5. Explicit ε_t and |ε_t|_i / |ε_t|_{i-1}
    if result.true_root is not None:
        tr_str = f"{result.true_root:.8f}"
        if rec.et is not None:
            et_formula = (
                f"εₜ,{curr_idx} = |({tr_str} - {x_curr_str}) / {tr_str}| × 100% "
                f"= [bold]{_fmt_pct(rec.et)}[/bold]"
            )
            grid.add_row("5. True Error εₜ", et_formula)

            if rec.et_ratio is not None and prev_et is not None:
                ratio_formula = (
                    f"|εₜ|{curr_idx} / |εₜ|{prev_idx} = "
                    f"{_fmt_pct(rec.et)} / {_fmt_pct(prev_et)} = [bold cyan]{_fmt_ratio(rec.et_ratio)}[/bold cyan]"
                )
                grid.add_row("   Error Ratio", ratio_formula)
            else:
                grid.add_row("   Error Ratio", "[dim]— (previous true error is 0 or unavailable)[/dim]")
        else:
            abs_err = abs(result.true_root - rec.x_curr)
            grid.add_row(
                "5. True Error εₜ",
                f"[yellow]Relative εₜ undefined (x_true = 0); |x_true - x{curr_idx}| = {abs_err:.6e}[/yellow]",
            )
    else:
        grid.add_row("5. True Error εₜ", "[dim]N/A (reference true root unavailable)[/dim]")

    # 6. Stopping condition check
    if rec.met_tolerance:
        stop_str = (
            f"εₐ ({_fmt_pct(rec.ea)}) < εₛ ({result.es:g}%)  →  "
            f"[bold green]YES — Stopping criterion satisfied![/bold green]"
        )
        border = "green"
        title_style = "bold green"
    elif i == result.max_iter:
        ea_label = _fmt_pct(rec.ea) if rec.ea is not None else "Undef"
        stop_str = (
            f"εₐ ({ea_label}) < εₛ ({result.es:g}%)?  →  "
            f"[bold yellow]NO — Reached max iterations ({result.max_iter}).[/bold yellow]"
        )
        border = "yellow"
        title_style = "bold yellow"
    else:
        ea_label = _fmt_pct(rec.ea) if rec.ea is not None else "Undef"
        stop_str = (
            f"εₐ ({ea_label}) < εₛ ({result.es:g}%)?  →  "
            f"[dim]No — proceed to Iteration {i + 1}[/dim]"
        )
        border = "cyan"
        title_style = "bold cyan"

    grid.add_row("6. Stopping Check", stop_str)

    return Panel(
        grid,
        title=f"[{title_style}]Iteration {i}  (Computing x{curr_idx} from x{prev_idx})[/{title_style}]",
        box=box.ROUNDED,
        border_style=border,
        padding=(0, 2),
    )


def display_iterations(
    console: Console,
    result: SolverResult,
    walkthrough: bool = False,
) -> None:
    """Render every iteration panel, pausing between steps in Walkthrough Mode."""
    iter_records = [rec for rec in result.records if rec.i >= 1]
    if not iter_records:
        return

    skip_pauses = not walkthrough
    prev_et = result.records[0].et

    for idx, rec in enumerate(iter_records):
        panel = _render_iteration_panel(rec, prev_et, result)
        console.print(panel)
        prev_et = rec.et

        if not skip_pauses and idx < len(iter_records) - 1:
            reply = console.input(
                "[dim]Press Enter for next iteration (or type 's' + Enter to show all remaining)...[/dim]"
            ).strip().lower()
            if reply == "s":
                skip_pauses = True

    if walkthrough:
        console.input("[dim]Press Enter to view the Summary Table & Final Verdict...[/dim]")


def display_summary_table(console: Console, result: SolverResult) -> None:
    """Display the formatted Rich iteration summary table."""
    table = Table(
        title="[bold cyan]Fixed-Point Iteration Summary Table[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=False,
        expand=True,
    )

    table.add_column("i", justify="right", style="bold white", width=5)
    table.add_column("xᵢ", justify="right", width=18)
    table.add_column("εₐ (%)", justify="right", width=16)
    table.add_column("εₜ (%)", justify="right", width=16)
    table.add_column("|εₜ|ᵢ / |εₜ|ᵢ₋₁", justify="right", width=18)

    last_idx = len(result.records) - 1
    for idx, rec in enumerate(result.records):
        is_last = idx == last_idx

        if rec.diverged:
            x_str = (
                f"{format_number(rec.x_curr, 6)} (!)"
                if rec.x_curr is not None
                else "DIVERGED"
            )
            table.add_row(
                str(rec.i),
                x_str,
                "—",
                "—",
                "—",
                style="bold red",
            )
            continue

        assert rec.x_curr is not None
        x_str = format_number(rec.x_curr, 6)
        ea_str = (
            "—"
            if rec.i == 0
            else (
                f"{rec.ea:.4f}"
                if rec.ea is not None and rec.ea < 1e5
                else (f"{rec.ea:.3e}" if rec.ea is not None else "Undef")
            )
        )
        et_str = (
            f"{rec.et:.4f}"
            if rec.et is not None and rec.et < 1e5
            else (f"{rec.et:.3e}" if rec.et is not None else "—")
        )
        ratio_str = _fmt_ratio(rec.et_ratio)

        row_style = ""
        if is_last and rec.i >= 1:
            if result.status == "converged":
                row_style = "bold green"
            elif result.status == "max_iter_reached":
                row_style = "bold yellow"

        table.add_row(str(rec.i), x_str, ea_str, et_str, ratio_str, style=row_style)

    console.print()
    console.print(table)


def display_final_verdict(console: Console, result: SolverResult) -> None:
    """Display the final boxed verdict with root, residual, and convergence diagnosis."""
    if result.status == "converged":
        badge = _badge("CONVERGED", "green")
        border = "green"
    elif result.status == "diverged":
        badge = _badge("DIVERGED", "red")
        border = "red"
    else:
        badge = _badge("MAX ITERATIONS REACHED", "yellow")
        border = "yellow"

    verdict = Table.grid(padding=(0, 2))
    verdict.add_column(style="bold cyan", width=26)
    verdict.add_column(style="white")

    verdict.add_row("Outcome", f"{badge}  {result.status_message}")

    root_label = "Last Valid Estimate" if result.status == "diverged" else "Estimated Root"
    verdict.add_row(
        root_label,
        f"[bold {'red' if result.status == 'diverged' else 'green'}]"
        f"x = {format_number(result.estimated_root, 8)}[/bold {'red' if result.status == 'diverged' else 'green'}]",
    )

    if result.true_root is not None:
        abs_true_err = abs(result.true_root - result.estimated_root)
        verdict.add_row(
            "Reference True Root",
            f"x_true = {result.true_root:.8f}  [dim]({result.true_root_source}; "
            f"|Eₜ| = {abs_true_err:.3e})[/dim]",
        )

    verdict.add_row("Total Iterations", f"{result.total_iterations} / {result.max_iter}")

    if result.final_residual is not None:
        res_abs = abs(result.final_residual)
        res_color = "green" if res_abs < 1e-2 else ("yellow" if res_abs < 1.0 else "red")
        verdict.add_row(
            "Final Residual f(x_root)",
            f"[{res_color}]{result.final_residual:.6e}[/{res_color}]",
        )
    else:
        verdict.add_row("Final Residual f(x_root)", "[red]Undefined[/red]")

    if result.pre_analysis is not None:
        verdict.add_row(
            "Pre-Iter Convergence",
            _format_convergence_line(result.pre_analysis),
        )

    if result.post_analysis is not None:
        verdict.add_row(
            "Post-Iter Convergence",
            _format_convergence_line(result.post_analysis),
        )

    if result.true_root_analysis is not None:
        verdict.add_row(
            "At True Root g'(x_true)",
            _format_convergence_line(result.true_root_analysis),
        )

    # Demonstrate E_{i+1} ≈ g'(ξ) E_i by comparing last observed ratio vs |g'(x_final)|
    valid_ratios = [
        rec.et_ratio
        for rec in result.records
        if rec.et_ratio is not None and math.isfinite(rec.et_ratio)
    ]
    if (
        valid_ratios
        and result.post_analysis is not None
        and result.post_analysis.derivative_value is not None
    ):
        obs_ratio = valid_ratios[-1]
        theo_ratio = abs(result.post_analysis.derivative_value)
        verdict.add_row(
            "Linear Error Rate Check",
            f"Observed |εₜ|ₙ / |εₜ|ₙ₋₁ = [bold cyan]{obs_ratio:.4f}[/bold cyan]   vs.   "
            f"Theoretical |g'(x_final)| = [bold cyan]{theo_ratio:.4f}[/bold cyan]  "
            "[dim](Eᵢ₊₁ ≈ g'(ξ)Eᵢ)[/dim]",
        )

    console.print()
    console.print(
        Panel(
            verdict,
            title="[bold cyan]Final Convergence Verdict[/bold cyan]",
            box=box.DOUBLE,
            border_style=border,
            padding=(1, 2),
        )
    )


def show_graph_menu(console: Console) -> str:
    """Display the interactive terminal graph selector menu."""
    gtable = Table.grid(padding=(0, 2))
    gtable.add_column(style="bold cyan", justify="right", width=5)
    gtable.add_column(style="bold white")
    gtable.add_column(style="dim")

    gtable.add_row(
        "[1]",
        "View All Graphs Sequentially",
        "Graph A → Graph B → Graph C (pauses cleanly between each)",
    )
    gtable.add_row(
        "[2]",
        "Graph A — Original Function Root Plot",
        "y = f(x) and y = 0 with highlighted root",
    )
    gtable.add_row(
        "[3]",
        "Graph B — Two-Curve Intersection & Cobweb Plot",
        "y = x, y = g(x), and step-by-step cobweb trajectory",
    )
    gtable.add_row(
        "[4]",
        "Graph C — Error Decay Plot",
        "εₐ (%) and εₜ (%) vs. iteration number i",
    )
    gtable.add_row(
        "[0]",
        "Return to Main Menu",
        "Solve another problem or exit",
    )

    console.print()
    console.print(
        Panel(
            gtable,
            title="[bold cyan]Terminal Visualizations (Plotext ASCII/Unicode Graphs)[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )

    while True:
        choice = console.input("[bold cyan]Select visualization [1/2/3/4/0]: [/bold cyan]").strip()
        if choice in {"1", "2", "3", "4", "0"}:
            return choice
        console.print("[red]Invalid choice.[/red] Please enter 1, 2, 3, 4, or 0.")
