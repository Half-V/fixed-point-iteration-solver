"""Terminal-based ASCII/Unicode visualizations powered by ``plotext``.

Provides three distinct graphs tailored for live presentations in standard
100-120 column terminals:
  - Graph A: Original Function Root Plot (y = f(x), y = 0, highlighted root)
  - Graph B: Two-Curve Intersection & Cobweb Plot (y = x, y = g(x), cobweb path)
  - Graph C: Error Decay Plot (ε_a and ε_t vs. iteration number i)
"""

from __future__ import annotations

import math
import shutil
from typing import Callable, Optional

import plotext as plt
import sympy as sp
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from solver import (
    SolverResult,
    format_expr_readable,
    format_number,
    safe_eval_expr,
)


def _get_plot_dimensions(
    console: Optional[Console] = None,
    height: int = 24,
) -> tuple[int, int]:
    """Compute clean plot dimensions that fit 100-120 col terminals without wrapping."""
    if console is not None:
        term_width = console.width
    else:
        term_width = shutil.get_terminal_size((108, 30)).columns
    width = max(72, min(term_width - 4, 108))
    return width, height


def _select_focus_iterates(result: SolverResult) -> tuple[list[float], bool]:
    """Return finite iterates suitable for plotting, clamping gracefully if diverged.

    Returns ``(focus_xs, was_clamped)``.
    """
    valid_xs = [
        rec.x_curr
        for rec in result.records
        if rec.x_curr is not None and not rec.diverged and math.isfinite(rec.x_curr)
    ]
    if not valid_xs:
        return [result.x0], False

    if len(valid_xs) == 1:
        return valid_xs, False

    raw_span = max(valid_xs) - min(valid_xs)
    first_step = max(abs(valid_xs[1] - valid_xs[0]), 1.0)
    needs_clamp = (
        result.status == "diverged"
        or raw_span > max(100.0, 25.0 * first_step)
        or max(abs(v) for v in valid_xs) > 1e5
    )

    if not needs_clamp:
        return valid_xs, False

    # Keep early iterates near x0 so viewers clearly see divergence away from y = x
    radius_cap = max(8.0, 10.0 * first_step)
    focus: list[float] = [valid_xs[0]]
    for val in valid_xs[1:]:
        if len(focus) >= 7:
            break
        if abs(val - valid_xs[0]) <= radius_cap or len(focus) < 2:
            if abs(val - valid_xs[0]) <= 1e4:
                focus.append(val)
            else:
                break
        else:
            break

    clamped = len(focus) < len(valid_xs) or result.status == "diverged"
    return focus, clamped


def _sample_curve_segments(
    func: Callable[[float], object],
    expr: sp.Expr,
    x_min: float,
    x_max: float,
    num_points: int = 260,
    y_clip: Optional[tuple[float, float]] = None,
) -> list[tuple[list[float], list[float]]]:
    """Sample a curve safely into contiguous segments, skipping domain errors/asymptotes."""
    if x_max <= x_min or num_points < 2:
        return []

    step = (x_max - x_min) / (num_points - 1)
    segments: list[tuple[list[float], list[float]]] = []
    cur_x: list[float] = []
    cur_y: list[float] = []

    y_span_limit: Optional[float] = None
    if y_clip is not None:
        y_span_limit = max(abs(y_clip[1] - y_clip[0]) * 0.75, 1.0)

    for idx in range(num_points):
        xv = x_min + idx * step
        yv, err = safe_eval_expr(func, expr, xv, limit=1e9)
        valid = err is None and yv is not None and math.isfinite(yv)
        if valid and y_clip is not None:
            assert yv is not None
            if yv < y_clip[0] or yv > y_clip[1]:
                valid = False

        if valid:
            assert yv is not None
            # Break segment on extreme vertical jumps (vertical asymptotes)
            if cur_y and y_span_limit is not None and abs(yv - cur_y[-1]) > y_span_limit:
                if len(cur_x) >= 2:
                    segments.append((cur_x, cur_y))
                cur_x, cur_y = [xv], [yv]
            else:
                cur_x.append(xv)
                cur_y.append(yv)
        else:
            if len(cur_x) >= 2:
                segments.append((cur_x, cur_y))
            cur_x, cur_y = [], []

    if len(cur_x) >= 2:
        segments.append((cur_x, cur_y))

    return segments


def _robust_y_bounds(
    ys: list[float],
    anchors: list[float],
    min_half_span: float = 0.5,
) -> tuple[float, float]:
    """Compute sensible y-axis limits that include anchor values without blowing up."""
    finite_ys = sorted(v for v in ys if math.isfinite(v))
    finite_anchors = [v for v in anchors if math.isfinite(v)]

    if finite_ys:
        n = len(finite_ys)
        low_q = finite_ys[max(0, int(0.05 * (n - 1)))]
        high_q = finite_ys[min(n - 1, int(0.95 * (n - 1)))]
        iqr = max(high_q - low_q, 1.0)
        trimmed = [v for v in finite_ys if low_q - 2.0 * iqr <= v <= high_q + 2.0 * iqr]
        pool = (trimmed or finite_ys) + finite_anchors
    elif finite_anchors:
        pool = finite_anchors
    else:
        return -1.0, 1.0

    y_min = min(pool)
    y_max = max(pool)
    span = y_max - y_min
    if span < 2.0 * min_half_span:
        mid = 0.5 * (y_min + y_max)
        y_min = mid - min_half_span
        y_max = mid + min_half_span
    else:
        pad = 0.14 * span
        y_min -= pad
        y_max += pad

    return y_min, y_max


def _choose_legend_corner(
    points: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    preferred: str = "top-right",
) -> tuple[float, float, str, str]:
    """Pick the least crowded plot corner for the legend box."""
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    x_left_cut = x_min + 0.42 * x_span
    x_right_cut = x_max - 0.42 * x_span
    y_bot_cut = y_min + 0.42 * y_span
    y_top_cut = y_max - 0.42 * y_span

    counts = {
        "top-right": 0,
        "top-left": 0,
        "bottom-right": 0,
        "bottom-left": 0,
    }
    for px, py in points:
        if not (x_min <= px <= x_max and y_min <= py <= y_max):
            continue
        if px >= x_right_cut and py >= y_top_cut:
            counts["top-right"] += 1
        if px <= x_left_cut and py >= y_top_cut:
            counts["top-left"] += 1
        if px >= x_right_cut and py <= y_bot_cut:
            counts["bottom-right"] += 1
        if px <= x_left_cut and py <= y_bot_cut:
            counts["bottom-left"] += 1

    # Slight tie-break preference toward the preferred corner
    order = [preferred] + [c for c in counts if c != preferred]
    best = min(order, key=lambda key: counts[key])

    x_pad = 0.02 * x_span
    y_pad = 0.03 * y_span
    coords = {
        "top-right": (x_max - x_pad, y_max - y_pad, "right", "top"),
        "top-left": (x_min + x_pad, y_max - y_pad, "left", "top"),
        "bottom-right": (x_max - x_pad, y_min + y_pad, "right", "bottom"),
        "bottom-left": (x_min + x_pad, y_min + y_pad, "left", "bottom"),
    }
    return coords[best]


def _setup_base_figure(
    title: str,
    xlabel: str,
    ylabel: str,
    width: int,
    height: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> None:
    """Configure common styling, rulers, limits, and dotted grid on ``plotext.figure``."""
    plt.figure.clear()
    plt.figure.plot_size(width, height)
    plt.figure.theme("clear")
    plt.figure.title(title)
    plt.figure.label(xlabel, axis="x")
    plt.figure.label(ylabel, axis="y")
    plt.figure.ruler("x").lim(x_lim[0], x_lim[1]).grid(
        True, style="dotted", pixel=plt.pixel("gray")
    )
    plt.figure.ruler("y").lim(y_lim[0], y_lim[1]).grid(
        True, style="dotted", pixel=plt.pixel("gray")
    )


def render_graph_a(
    result: SolverResult,
    width: int = 104,
    height: int = 24,
    colorless: bool = False,
) -> tuple[str, str]:
    """Build Graph A: Original Function Root Plot (``y = f(x)`` and ``y = 0``).

    Returns ``(rendered_plot_str, caption_note)``.
    """
    focus_xs, clamped = _select_focus_iterates(result)
    anchors_x = list(focus_xs)

    if result.true_root is not None and math.isfinite(result.true_root):
        if abs(result.true_root - result.x0) <= max(50.0, 4.0 * max(1.0, max(focus_xs) - min(focus_xs))):
            anchors_x.append(result.true_root)

    x_lo = min(anchors_x)
    x_hi = max(anchors_x)
    span_x = x_hi - x_lo
    pad_x = max(0.35 * span_x, 0.75)
    x_min, x_max = x_lo - pad_x, x_hi + pad_x

    # Coarse pass to compute clean y-limits
    raw_segments = _sample_curve_segments(
        result.problem.f_func,
        result.problem.f_expr,
        x_min,
        x_max,
        num_points=260,
    )
    all_ys = [yv for _, seg_y in raw_segments for yv in seg_y]

    root_x = result.estimated_root
    root_y = result.final_residual
    y_anchors = [0.0]
    if root_y is not None and math.isfinite(root_y) and abs(root_y) < 1e4:
        y_anchors.append(root_y)

    y_min, y_max = _robust_y_bounds(all_ys, y_anchors, min_half_span=0.5)

    # Re-sample clipped neatly inside the y window so asymptotes don't draw vertical spikes
    margin_y = 0.05 * (y_max - y_min)
    segments = _sample_curve_segments(
        result.problem.f_func,
        result.problem.f_expr,
        x_min,
        x_max,
        num_points=280,
        y_clip=(y_min - margin_y, y_max + margin_y),
    )

    f_label = result.problem.f_raw or format_expr_readable(result.problem.f_expr)
    if len(f_label) > 28:
        f_label = f_label[:25] + "..."

    _setup_base_figure(
        title=f"Graph A — Original Function Root Plot: y = {f_label}",
        xlabel="x",
        ylabel="f(x)",
        width=width,
        height=height,
        x_lim=(x_min, x_max),
        y_lim=(y_min, y_max),
    )

    # Horizontal x-axis line y = 0
    plt.figure.line(
        0.0,
        orientation="h",
        style="default",
        pixel=plt.pixel("white"),
        label="y = 0 (x-axis)",
    )

    # Plot f(x) curve
    plotted_pts: list[tuple[float, float]] = []
    for idx, (seg_x, seg_y) in enumerate(segments):
        sig = plt.figure.signal(
            seg_x,
            seg_y,
            marker=plt.marker("braille", plt.pixel("cyan")),
        ).lines(True)
        if idx == 0:
            sig.label(f"f(x) = {f_label}")
        plt.figure.draw(sig)
        plotted_pts.extend(zip(seg_x, seg_y))

    # Highlight initial guess x0 on f(x) if inside window
    f_x0, f_x0_err = safe_eval_expr(
        result.problem.f_func, result.problem.f_expr, result.x0
    )
    if (
        f_x0_err is None
        and f_x0 is not None
        and x_min <= result.x0 <= x_max
        and y_min <= f_x0 <= y_max
    ):
        start_sig = (
            plt.figure.signal(
                [result.x0],
                [f_x0],
                marker=plt.marker("◆", plt.pixel("yellow", style="bold")),
            )
            .lines(False)
            .label(f"Start x₀ = {format_number(result.x0, 4)}")
        )
        plt.figure.draw(start_sig)
        plotted_pts.append((result.x0, f_x0))

    # Highlight estimated root (or last valid iterate)
    if (
        root_y is not None
        and math.isfinite(root_x)
        and math.isfinite(root_y)
        and x_min <= root_x <= x_max
        and y_min <= root_y <= y_max
    ):
        marker_color = "red" if result.status == "diverged" else "green"
        root_tag = "Last Est." if result.status == "diverged" else "Est. Root"
        root_sig = (
            plt.figure.signal(
                [root_x],
                [root_y],
                marker=plt.marker("●", plt.pixel(marker_color, style="bold")),
            )
            .lines(False)
            .label(f"{root_tag} x = {format_number(root_x, 6)}")
        )
        plt.figure.draw(root_sig)
        plotted_pts.append((root_x, root_y))
    elif result.true_root is not None and x_min <= result.true_root <= x_max:
        true_sig = (
            plt.figure.signal(
                [result.true_root],
                [0.0],
                marker=plt.marker("★", plt.pixel("green", style="bold")),
            )
            .lines(False)
            .label(f"Ref. Root x* = {format_number(result.true_root, 6)}")
        )
        plt.figure.draw(true_sig)
        plotted_pts.append((result.true_root, 0.0))

    lx, ly, ha, va = _choose_legend_corner(
        plotted_pts, x_min, x_max, y_min, y_max, preferred="top-right"
    )
    plt.figure.legend(x=lx, y=ly, ha=ha, va=va, relative=True, style="rounded")

    f_display = result.problem.f_raw or format_expr_readable(result.problem.f_expr)
    caption = f"Root crossing of f(x) = {f_display} at y = 0. "
    if result.final_residual is not None:
        caption += f"Residual f({format_number(root_x, 6)}) = {result.final_residual:.3e}."
    if clamped:
        caption += " (View window clamped near initial iterates due to divergence.)"

    rendered = plt.figure.build().string(colorless=colorless)
    return rendered, caption


def render_graph_b(
    result: SolverResult,
    width: int = 104,
    height: int = 26,
    colorless: bool = False,
) -> tuple[str, str]:
    """Build Graph B: Two-Curve Intersection (``y = x``, ``y = g(x)``) & Cobweb Plot.

    Overlays the step-by-step cobweb trajectory:
    ``(x_0, 0) -> (x_0, g(x_0)) -> (x_1, x_1) -> (x_1, g(x_1)) -> ...``
    """
    focus_xs, clamped = _select_focus_iterates(result)

    # Decide whether y = 0 is close enough to include (x0, 0) naturally in the view
    x_lo = min(focus_xs)
    x_hi = max(focus_xs)
    span = max(x_hi - x_lo, 0.4)
    include_zero_y = abs(focus_xs[0]) <= max(10.0, 4.0 * span)

    # Also include true root in the framing if it's reasonably near the focus iterates
    frame_xs = list(focus_xs)
    if result.true_root is not None and math.isfinite(result.true_root):
        if abs(result.true_root - focus_xs[0]) <= max(25.0, 4.0 * span):
            frame_xs.append(result.true_root)

    fx_min, fx_max = min(frame_xs), max(frame_xs)
    f_span = max(fx_max - fx_min, 0.5)
    pad_left = max(0.32 * f_span, 0.35)
    pad_right = max(0.55 * f_span, 0.55)
    x_min = fx_min - pad_left
    x_max = fx_max + pad_right

    y_candidates = list(frame_xs)
    if include_zero_y:
        y_candidates.append(0.0)

    # Sample g(x) to ensure curve fits neatly around the cobweb
    raw_g_segments = _sample_curve_segments(
        result.problem.g_func,
        result.problem.g_expr,
        x_min,
        x_max,
        num_points=260,
    )
    sampled_gy = [yv for _, seg_y in raw_g_segments for yv in seg_y]

    # Keep y-range tightly anchored to cobweb vertices + y=x diagonal + reasonable g(x)
    y_lo = min(y_candidates + [x_min])
    y_hi = max(y_candidates + [x_max])
    y_pad = max(0.14 * (y_hi - y_lo), 0.25)
    y_min = y_lo - y_pad
    y_max = y_hi + y_pad

    # Expand slightly if g(x) has mild curvature just outside y_lo/y_hi
    if sampled_gy:
        g_lo = min(sampled_gy)
        g_hi = max(sampled_gy)
        max_expand = 0.65 * (y_max - y_min)
        if y_min - max_expand <= g_lo < y_min:
            y_min = g_lo - 0.1 * (y_max - y_min)
        if y_max < g_hi <= y_max + max_expand:
            y_max = g_hi + 0.1 * (y_max - y_min)

    # Re-sample g(x) clipped cleanly to the final y-window
    margin_y = 0.04 * (y_max - y_min)
    g_segments = _sample_curve_segments(
        result.problem.g_func,
        result.problem.g_expr,
        x_min,
        x_max,
        num_points=280,
        y_clip=(y_min - margin_y, y_max + margin_y),
    )

    # Construct cobweb trajectory: (x0, y_start) -> (x0, x1) -> (x1, x1) -> (x1, x2) -> ...
    y_start = 0.0 if include_zero_y else y_min
    cobweb_x: list[float] = [focus_xs[0]]
    cobweb_y: list[float] = [y_start]
    for idx in range(len(focus_xs) - 1):
        xk = focus_xs[idx]
        x_next = focus_xs[idx + 1]
        cobweb_x.extend([xk, x_next])
        cobweb_y.extend([x_next, x_next])

    g_label = result.problem.g_raw or format_expr_readable(result.problem.g_expr)
    if len(g_label) > 26:
        g_label = g_label[:23] + "..."

    _setup_base_figure(
        title=f"Graph B — Two-Curve Intersection & Cobweb Plot: y = x  vs  y = {g_label}",
        xlabel="x",
        ylabel="y",
        width=width,
        height=height,
        x_lim=(x_min, x_max),
        y_lim=(y_min, y_max),
    )

    # Reference x-axis y = 0 if inside view
    if y_min <= 0.0 <= y_max:
        plt.figure.line(
            0.0,
            orientation="h",
            style="dotted",
            pixel=plt.pixel("gray"),
        )

    plotted_pts: list[tuple[float, float]] = []

    # 1. Diagonal line f1(x) = x clipped to [x_min, x_max] ∩ [y_min, y_max]
    diag_lo = max(x_min, y_min)
    diag_hi = min(x_max, y_max)
    if diag_hi > diag_lo:
        diag_xs = [diag_lo + (diag_hi - diag_lo) * i / 120.0 for i in range(121)]
        diag_sig = (
            plt.figure.signal(
                diag_xs,
                diag_xs,
                marker=plt.marker("braille", plt.pixel("yellow")),
            )
            .lines(True)
            .label("f₁(x) = x")
        )
        plt.figure.draw(diag_sig)
        plotted_pts.extend(zip(diag_xs, diag_xs))

    # 2. Curve f2(x) = g(x)
    for idx, (seg_x, seg_y) in enumerate(g_segments):
        g_sig = plt.figure.signal(
            seg_x,
            seg_y,
            marker=plt.marker("braille", plt.pixel("cyan")),
        ).lines(True)
        if idx == 0:
            g_sig.label(f"f₂(x) = {g_label}")
        plt.figure.draw(g_sig)
        plotted_pts.extend(zip(seg_x, seg_y))

    # 3. Cobweb path
    if len(cobweb_x) >= 2:
        cobweb_color = "red" if result.status == "diverged" else "magenta"
        cobweb_sig = (
            plt.figure.signal(
                cobweb_x,
                cobweb_y,
                marker=plt.marker("braille", plt.pixel(cobweb_color)),
            )
            .lines(True)
            .label(f"Cobweb ({len(focus_xs) - 1} steps)")
        )
        plt.figure.draw(cobweb_sig)
        plotted_pts.extend(zip(cobweb_x, cobweb_y))

    # 4. Highlight starting point (x0, y_start)
    start_sig = plt.figure.signal(
        [focus_xs[0]],
        [y_start],
        marker=plt.marker("◆", plt.pixel("yellow", style="bold")),
    ).lines(False)
    plt.figure.draw(start_sig)

    # 5. Highlight final iterate (x_n, x_n)
    end_x = focus_xs[-1]
    end_color = "red" if result.status == "diverged" else "green"
    end_tag = "Diverging" if result.status == "diverged" else "Final x"
    end_sig = (
        plt.figure.signal(
            [end_x],
            [end_x],
            marker=plt.marker("●", plt.pixel(end_color, style="bold")),
        )
        .lines(False)
        .label(f"{end_tag} = {format_number(end_x, 5)}")
    )
    plt.figure.draw(end_sig)

    lx, ly, ha, va = _choose_legend_corner(
        plotted_pts, x_min, x_max, y_min, y_max, preferred="bottom-right"
    )
    plt.figure.legend(x=lx, y=ly, ha=ha, va=va, relative=True, style="rounded")

    steps_shown = len(focus_xs) - 1
    caption = (
        f"Cobweb path: ◆ Start ({format_number(focus_xs[0], 3)}, {format_number(y_start, 1)}) "
        f"→ (x₀, g(x₀)) → (x₁, x₁) → … → ● Final ({format_number(end_x, 5)}) "
        f"[{steps_shown} step(s)]."
    )
    if clamped:
        caption += " (Window clamped around early steps so the diverging trajectory stays visible.)"

    rendered = plt.figure.build().string(colorless=colorless)
    return rendered, caption


def render_graph_c(
    result: SolverResult,
    width: int = 104,
    height: int = 22,
    colorless: bool = False,
) -> tuple[Optional[str], str]:
    """Build Graph C: Error Decay Plot (``ε_a`` and ``ε_t`` vs. iteration ``i``).

    Returns ``(rendered_plot_str_or_None, caption_note)``.
    """
    ea_i: list[float] = []
    ea_v: list[float] = []
    et_i: list[float] = []
    et_v: list[float] = []

    cap_pct = 1e6
    clipped_any = False

    for rec in result.records:
        if rec.diverged:
            continue
        if rec.ea is not None and math.isfinite(rec.ea):
            val = rec.ea
            if val > cap_pct:
                val = cap_pct
                clipped_any = True
            ea_i.append(float(rec.i))
            ea_v.append(val)
        if rec.et is not None and math.isfinite(rec.et):
            val = rec.et
            if val > cap_pct:
                val = cap_pct
                clipped_any = True
            et_i.append(float(rec.i))
            et_v.append(val)

    if not ea_i and not et_i:
        return (
            None,
            "No finite error points are available to plot (iteration stopped before computing errors).",
        )

    all_i = ea_i + et_i
    all_v = ea_v + et_v
    x_min = max(0.0, min(all_i) - 0.3)
    x_max = max(all_i) + 0.3
    if x_max - x_min < 1.0:
        x_max = x_min + 1.0

    y_max_raw = max(all_v + ([result.es] if result.es > 0 else [0.0]))
    y_min = 0.0
    y_max = max(y_max_raw * 1.12, 1.0)

    _setup_base_figure(
        title="Graph C — Relative Error Decay vs. Iteration Number",
        xlabel="Iteration (i)",
        ylabel="Percent Relative Error (%)",
        width=width,
        height=height,
        x_lim=(x_min, x_max),
        y_lim=(y_min, y_max),
    )

    # Set clean integer iteration tick marks on the x-axis
    min_tick = int(math.floor(min(all_i)))
    max_tick = int(math.ceil(max(all_i)))
    tick_span = max(1, max_tick - min_tick)
    tick_step = max(1, int(math.ceil(tick_span / 10.0)))
    tick_positions = list(range(min_tick, max_tick + 1, tick_step))
    if tick_positions[-1] != max_tick:
        tick_positions.append(max_tick)
    plt.figure.ruler("x").ticks(tick_positions, [str(t) for t in tick_positions])

    plotted_pts: list[tuple[float, float]] = []

    # Tolerance reference line ε_s
    if result.es > 0 and result.es <= y_max:
        plt.figure.line(
            result.es,
            orientation="h",
            style="dotted",
            pixel=plt.pixel("red"),
            label=f"Tolerance εₛ = {result.es:g}%",
        )

    if ea_i:
        ea_sig = (
            plt.figure.signal(
                ea_i,
                ea_v,
                marker=plt.marker("braille", plt.pixel("cyan")),
            )
            .lines(True)
            .label("Approx. Error εₐ (%)")
        )
        plt.figure.draw(ea_sig)
        ea_pts = plt.figure.signal(
            ea_i,
            ea_v,
            marker=plt.marker("◆", plt.pixel("cyan", style="bold")),
        ).lines(False)
        plt.figure.draw(ea_pts)
        plotted_pts.extend(zip(ea_i, ea_v))

    if et_i:
        et_sig = (
            plt.figure.signal(
                et_i,
                et_v,
                marker=plt.marker("braille", plt.pixel("green")),
            )
            .lines(True)
            .label("True Error εₜ (%)")
        )
        plt.figure.draw(et_sig)
        et_pts = plt.figure.signal(
            et_i,
            et_v,
            marker=plt.marker("●", plt.pixel("green", style="bold")),
        ).lines(False)
        plt.figure.draw(et_pts)
        plotted_pts.extend(zip(et_i, et_v))

    lx, ly, ha, va = _choose_legend_corner(
        plotted_pts, x_min, x_max, y_min, y_max, preferred="top-right"
    )
    plt.figure.legend(x=lx, y=ly, ha=ha, va=va, relative=True, style="rounded")

    caption = (
        "Tracks percent relative error across iterations — linear convergence appears as "
        "geometric reduction by factor ≈ |g'(ξ)| per step."
    )
    if clipped_any:
        caption += " (Extreme diverging error values clipped at 10⁶%.)"

    rendered = plt.figure.build().string(colorless=colorless)
    return rendered, caption


def _print_plot(
    console: Console,
    title: str,
    rendered: Optional[str],
    caption: str,
    border_style: str = "cyan",
) -> None:
    """Display a rendered plotext graph with a clean header & caption in ``rich``."""
    console.print()
    if rendered is None:
        console.print(
            Panel(
                f"[yellow]{caption}[/yellow]",
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="yellow",
            )
        )
        return

    console.print(Text.from_ansi(rendered))
    console.print(
        Panel(
            f"[dim]{caption}[/dim]",
            border_style=border_style,
            padding=(0, 1),
        )
    )


def show_graph_a(result: SolverResult, console: Console) -> None:
    """Render and display Graph A (Original Function Root Plot) in the terminal."""
    width, height = _get_plot_dimensions(console, height=24)
    rendered, caption = render_graph_a(result, width=width, height=height)
    _print_plot(console, "Graph A — Original Function Root Plot", rendered, caption)


def show_graph_b(result: SolverResult, console: Console) -> None:
    """Render and display Graph B (Two-Curve Intersection & Cobweb Plot) in the terminal."""
    width, height = _get_plot_dimensions(console, height=26)
    rendered, caption = render_graph_b(result, width=width, height=height)
    _print_plot(
        console,
        "Graph B — Two-Curve Intersection & Cobweb Plot",
        rendered,
        caption,
    )


def show_graph_c(result: SolverResult, console: Console) -> None:
    """Render and display Graph C (Error Decay Plot) in the terminal."""
    width, height = _get_plot_dimensions(console, height=22)
    rendered, caption = render_graph_c(result, width=width, height=height)
    _print_plot(console, "Graph C — Error Decay Plot", rendered, caption)
