"""CLI entry point and interactive presentation loop for the Fixed-Point Iteration Solver.

Run interactively with:
    python main.py

Or run an instant non-interactive Lecture Example report + graphs with:
    python main.py --demo
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from solver import LECTURE_PRESET, SolverResult, build_problem, solve_fixed_point
from ui import (
    configure_utf8_console,
    display_final_verdict,
    display_iterations,
    display_problem_setup,
    display_summary_table,
    print_banner,
    prompt_custom_problem,
    prompt_lecture_preset_options,
    prompt_walkthrough_mode,
    show_graph_menu,
    show_main_menu,
    show_method_reference,
)
from visualizer import show_graph_a, show_graph_b, show_graph_c


def _run_graph_loop(console: Console, result: SolverResult) -> None:
    """Interactive visualization viewer so graphs stay cleanly visible on screen."""
    while True:
        choice = show_graph_menu(console)
        if choice == "0":
            return
        if choice == "1":
            show_graph_a(result, console)
            console.input("[dim]Press Enter to view Graph B (Two-Curve Intersection & Cobweb Plot)...[/dim]")
            show_graph_b(result, console)
            console.input("[dim]Press Enter to view Graph C (Error Decay Plot)...[/dim]")
            show_graph_c(result, console)
            console.input("[dim]Press Enter to return to the Visualization Menu...[/dim]")
        elif choice == "2":
            show_graph_a(result, console)
            console.input("[dim]Press Enter to continue...[/dim]")
        elif choice == "3":
            show_graph_b(result, console)
            console.input("[dim]Press Enter to continue...[/dim]")
        elif choice == "4":
            show_graph_c(result, console)
            console.input("[dim]Press Enter to continue...[/dim]")


def _present_solution(
    console: Console,
    result: SolverResult,
    walkthrough: bool,
    interactive_graphs: bool = True,
) -> None:
    """Display full numerical output (setup, steps, table, verdict) and visualizations."""
    display_problem_setup(console, result)
    if walkthrough:
        console.input("[dim]Press Enter to begin Iteration 1...[/dim]")

    display_iterations(console, result, walkthrough=walkthrough)
    display_summary_table(console, result)
    display_final_verdict(console, result)

    if interactive_graphs:
        _run_graph_loop(console, result)
    else:
        show_graph_a(result, console)
        show_graph_b(result, console)
        show_graph_c(result, console)


def _run_lecture_preset(console: Console, non_interactive: bool = False) -> None:
    """Execute the classic Chapra & Canale Lecture Example Preset."""
    problem = build_problem(
        str(LECTURE_PRESET["f_str"]),
        str(LECTURE_PRESET["g_str"]),
    )
    if non_interactive:
        es = float(LECTURE_PRESET["es"])
        max_iter = int(LECTURE_PRESET["max_iter"])
        stop_on_tol = True
        walkthrough = False
    else:
        es, max_iter, stop_on_tol = prompt_lecture_preset_options(console)
        walkthrough = prompt_walkthrough_mode(console)

    result = solve_fixed_point(
        problem=problem,
        x0=float(LECTURE_PRESET["x0"]),
        es=es,
        max_iter=max_iter,
        true_root=float(LECTURE_PRESET["true_root"]),
        true_root_source=str(LECTURE_PRESET["true_root_source"]),
        stop_on_tolerance=stop_on_tol,
    )
    _present_solution(
        console,
        result,
        walkthrough=walkthrough,
        interactive_graphs=not non_interactive,
    )


def _run_custom_problem(console: Console) -> None:
    """Collect a custom user problem, solve it, and display results + graphs."""
    problem, x0, es, max_iter, true_root = prompt_custom_problem(console)
    walkthrough = prompt_walkthrough_mode(console)

    result = solve_fixed_point(
        problem=problem,
        x0=x0,
        es=es,
        max_iter=max_iter,
        true_root=true_root,
        true_root_source="User-Provided" if true_root is not None else None,
        stop_on_tolerance=True,
    )
    _present_solution(console, result, walkthrough=walkthrough, interactive_graphs=True)


def run_interactive(console: Console) -> None:
    """Main interactive CLI application loop."""
    print_banner(console)

    while True:
        choice = show_main_menu(console)
        if choice == "0":
            console.print(
                "\n[bold cyan]Thank you for using the Fixed-Point Iteration Solver![/bold cyan]\n"
            )
            return
        if choice == "1":
            _run_lecture_preset(console, non_interactive=False)
        elif choice == "2":
            _run_custom_problem(console)
        elif choice == "3":
            show_method_reference(console)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse optional CLI flags."""
    parser = argparse.ArgumentParser(
        description="Terminal application demonstrating Simple Fixed-Point Iteration."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the Lecture Example Preset non-interactively (Full Report + all 3 graphs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Application entry point."""
    args = parse_args(argv)
    console = configure_utf8_console()

    try:
        if args.demo:
            print_banner(console)
            _run_lecture_preset(console, non_interactive=True)
        else:
            run_interactive(console)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Session interrupted — exiting cleanly.[/yellow]\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
