"""Core mathematical engine for Simple Fixed-Point Iteration.

Implements fixed-point (one-point / successive substitution) root finding,
safe SymPy expression parsing, symbolic differentiation for convergence
classification, error analysis, divergence guards, and high-precision
reference root discovery via ``sympy.nsolve``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X_SYM = sp.Symbol("x", real=True)
DIVERGENCE_LIMIT = 1e12

LECTURE_PRESET: dict[str, object] = {
    "name": "Lecture Example: f(x) = e^(-x) - x",
    "f_str": "exp(-x) - x",
    "g_str": "exp(-x)",
    "x0": 0.0,
    "es": 0.5,
    "max_iter": 10,
    "true_root": 0.56714329,
    "true_root_source": "Lecture Reference",
}

_PARSER_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_FORBIDDEN_PATTERNS = (
    "__",
    "import",
    "lambda",
    "eval",
    "exec",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    ";",
)

_SAFE_GLOBAL_DICT: dict[str, object] = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Function": sp.Function,
    "Number": sp.Number,
}

_SAFE_LOCAL_DICT: dict[str, object] = {
    "x": X_SYM,
    "e": sp.E,
    "E": sp.E,
    "pi": sp.pi,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "log10": lambda v: sp.log(v, 10),
    "log2": lambda v: sp.log(v, 2),
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
}

_EXTRA_MATH_FUNCS: dict[str, Callable] = {
    "Abs": abs,
    "cbrt": lambda v: math.copysign(abs(v) ** (1.0 / 3.0), v)
    if isinstance(v, (int, float))
    else v ** (1.0 / 3.0),
    "sec": lambda v: 1.0 / math.cos(v),
    "csc": lambda v: 1.0 / math.sin(v),
    "cot": lambda v: 1.0 / math.tan(v),
    "sign": lambda v: (1.0 if v > 0 else (-1.0 if v < 0 else 0.0)),
}


@dataclass
class ParsedProblem:
    """Validated symbolic and numerical representation of f(x), g(x), and g'(x)."""

    x_sym: sp.Symbol
    f_raw: str
    g_raw: str
    f_expr: sp.Expr
    g_expr: sp.Expr
    g_prime_expr: sp.Expr
    g_auto_derived: bool
    f_func: Callable[[float], object]
    g_func: Callable[[float], object]
    g_prime_func: Callable[[float], object]


@dataclass
class ConvergenceAnalysis:
    """Result of evaluating g'(x) at a point and classifying fixed-point behavior."""

    point_label: str
    point_value: float
    derivative_value: Optional[float]
    classification: str
    is_convergent: Optional[bool]
    is_oscillatory: Optional[bool]
    severity: str  # "green", "yellow", or "red"
    explanation: str


@dataclass
class IterationRecord:
    """Data captured for a single step i (i = 0 is the initial guess baseline)."""

    i: int
    x_prev: Optional[float]
    x_curr: Optional[float]
    substitution_str: Optional[str] = None
    ea: Optional[float] = None
    ea_note: Optional[str] = None
    et: Optional[float] = None
    et_ratio: Optional[float] = None
    met_tolerance: bool = False
    diverged: bool = False
    divergence_reason: Optional[str] = None


@dataclass
class SolverResult:
    """Complete output of a Simple Fixed-Point Iteration run."""

    problem: ParsedProblem
    x0: float
    es: float
    max_iter: int
    records: list[IterationRecord] = field(default_factory=list)
    status: str = "max_iter_reached"  # "converged", "max_iter_reached", "diverged"
    status_message: str = ""
    estimated_root: float = 0.0
    total_iterations: int = 0
    final_residual: Optional[float] = None
    true_root: Optional[float] = None
    true_root_source: str = "Unavailable"
    pre_analysis: Optional[ConvergenceAnalysis] = None
    post_analysis: Optional[ConvergenceAnalysis] = None
    true_root_analysis: Optional[ConvergenceAnalysis] = None


def parse_math_expression(expr_str: str) -> sp.Expr:
    """Safely parse a user-supplied single-variable math expression in ``x``.

    Supports natural syntax such as ``exp(-x) - x``, ``e^(-x) - x``,
    ``x^2 - 2*x - 3``, ``2x^2 - 5x + 1``, and ``cos(x) - x``.
    Raises ``ValueError`` with a friendly explanation on invalid input.
    """
    if not expr_str or not expr_str.strip():
        raise ValueError("Expression cannot be empty.")

    cleaned = expr_str.strip()

    # Gracefully allow inputs like "f(x) = exp(-x) - x" or "g(x) = exp(-x)"
    cleaned = re.sub(
        r"^(?:[fFgGyY]\s*(?:\(\s*x\s*\))?\s*=\s*)",
        "",
        cleaned,
    ).strip()

    if not cleaned:
        raise ValueError("Expression after '=' cannot be empty.")

    lowered = cleaned.lower()
    for bad in _FORBIDDEN_PATTERNS:
        if bad in lowered:
            raise ValueError(f"Unsupported or unsafe token '{bad}' in expression.")

    if "=" in cleaned:
        raise ValueError(
            "Enter an expression only (e.g., 'exp(-x) - x'), not an equation with '='."
        )

    # Normalize Unicode minus and caret-like operators
    cleaned = cleaned.replace("−", "-").replace("×", "*").replace("÷", "/")

    try:
        expr = parse_expr(
            cleaned,
            local_dict=_SAFE_LOCAL_DICT,
            global_dict=_SAFE_GLOBAL_DICT,
            transformations=_PARSER_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not parse '{expr_str}': {exc}. "
            "Try syntax like 'exp(-x) - x', 'x^2 - 2*x - 3', or 'cos(x) - x'."
        ) from exc

    if not isinstance(expr, sp.Expr):
        raise ValueError(
            f"'{expr_str}' did not evaluate to a valid mathematical expression."
        )

    extra_symbols = {
        sym.name for sym in expr.free_symbols if sym != X_SYM
    }
    if extra_symbols:
        names = ", ".join(sorted(extra_symbols))
        raise ValueError(
            f"Unrecognized variable(s) or symbol(s): {names}. "
            "Use 'x' as the variable and standard functions (exp, sin, cos, ln, sqrt)."
        )

    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Expression contains an undefined constant (e.g., 1/0).")

    return expr


def _make_numeric_func(expr: sp.Expr) -> Callable[[float], object]:
    """Create a fast numerical evaluator with a safe SymPy fallback."""
    try:
        fast_fn = sp.lambdify(X_SYM, expr, modules=[_EXTRA_MATH_FUNCS, "math"])
    except Exception:
        fast_fn = None

    def _evaluate(val: float) -> object:
        if fast_fn is not None:
            try:
                return fast_fn(val)
            except (NameError, TypeError, AttributeError):
                pass
        return expr.evalf(subs={X_SYM: val})

    return _evaluate


def safe_eval_expr(
    func: Callable[[float], object],
    expr: sp.Expr,
    val: float,
    limit: Optional[float] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Evaluate a symbolic expression at ``val`` with comprehensive safety guards.

    Returns ``(float_result, None)`` on success, or ``(None, reason)`` if
    overflow, NaN, Inf, complex values, division by zero, or domain errors occur.
    When ``limit`` is provided and ``abs(result) > limit``, returns
    ``(result, reason)`` so callers can still display the runaway value.
    """
    try:
        raw = func(val)
    except OverflowError:
        return None, f"Numerical overflow while evaluating at x = {format_number(val)}."
    except ZeroDivisionError:
        return None, f"Division by zero encountered at x = {format_number(val)}."
    except ValueError as exc:
        # Distinguish complex domain issues (e.g. sqrt(-1), ln(-2)) from other errors
        try:
            sym_val = complex(expr.evalf(subs={X_SYM: val}))
            if abs(sym_val.imag) > 1e-12:
                return (
                    None,
                    f"Complex number encountered ({sym_val.real:.6g} "
                    f"{'+' if sym_val.imag >= 0 else '-'} {abs(sym_val.imag):.6g}i) "
                    f"at x = {format_number(val)}.",
                )
        except Exception:
            pass
        return None, f"Math domain error at x = {format_number(val)} ({exc})."
    except Exception as exc:
        return None, f"Evaluation error at x = {format_number(val)}: {exc}"

    # Handle SymPy numerical objects returned by fallback evaluation
    if isinstance(raw, sp.Basic):
        if raw.has(sp.zoo, sp.nan):
            return None, f"Undefined / NaN result at x = {format_number(val)}."
        if raw.has(sp.oo, -sp.oo):
            return None, f"Infinite result (±Inf) at x = {format_number(val)}."
        try:
            raw = complex(raw.evalf())
        except Exception:
            return None, f"Non-numeric result '{raw}' at x = {format_number(val)}."

    if isinstance(raw, complex):
        if math.isnan(raw.real) or math.isnan(raw.imag):
            return None, f"NaN encountered at x = {format_number(val)}."
        if math.isinf(raw.real) or math.isinf(raw.imag):
            return None, f"Infinite value encountered at x = {format_number(val)}."
        if abs(raw.imag) > 1e-12:
            sign = "+" if raw.imag >= 0 else "-"
            return (
                None,
                f"Complex number encountered ({raw.real:.6g} {sign} {abs(raw.imag):.6g}i) "
                f"at x = {format_number(val)}.",
            )
        result = float(raw.real)
    else:
        try:
            result = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None, f"Could not convert result to float at x = {format_number(val)}."

    if math.isnan(result):
        return None, f"NaN encountered at x = {format_number(val)}."
    if math.isinf(result):
        return None, f"Infinite value (±Inf) encountered at x = {format_number(val)}."

    if limit is not None and abs(result) > limit:
        return (
            result,
            f"Runaway magnitude: |x| = {abs(result):.3e} exceeded divergence limit ({limit:.0e}).",
        )

    return result, None


def format_number(val: float, decimals: int = 6) -> str:
    """Format a float cleanly, switching to scientific notation for extreme scales."""
    if abs(val) == 0.0:
        return f"{0.0:.{decimals}f}"
    if abs(val) >= 1e7 or (0 < abs(val) < 1e-5):
        return f"{val:.{max(4, decimals - 2)}e}"
    return f"{val:.{decimals}f}"


def format_expr_readable(expr: sp.Expr) -> str:
    """Format a SymPy expression into clean, human-readable mathematical notation."""
    text = str(expr)
    text = text.replace("**", "^")
    text = re.sub(r"\bAbs\(", "abs(", text)
    text = re.sub(r"\bE\b", "e", text)
    return text


def format_substitution(expr: sp.Expr, x_val: float, decimals: int = 6) -> str:
    """Return an explicit formula substitution string with ``x`` replaced by ``x_val``."""
    expr_str = format_expr_readable(expr)
    val_str = format_number(x_val, decimals=decimals)
    # Avoid double parentheses when x is the sole argument, e.g. cos(x) -> cos(0.500000)
    substituted = re.sub(r"\(x\)", f"({val_str})", expr_str)
    # Replace remaining standalone occurrences of variable x
    substituted = re.sub(r"\bx\b", f"({val_str})", substituted)
    return substituted


def build_problem(f_str: str, g_str: Optional[str] = None) -> ParsedProblem:
    """Parse ``f(x)`` and ``g(x)`` (defaulting to ``g(x) = f(x) + x``) and compute ``g'(x)``."""
    f_expr = parse_math_expression(f_str)

    if g_str is None or not g_str.strip():
        g_expr = f_expr + X_SYM
        g_auto_derived = True
        g_raw = format_expr_readable(g_expr)
    else:
        g_expr = parse_math_expression(g_str)
        g_auto_derived = False
        g_raw = g_str.strip()

    g_prime_expr = sp.diff(g_expr, X_SYM)

    return ParsedProblem(
        x_sym=X_SYM,
        f_raw=f_str.strip(),
        g_raw=g_raw,
        f_expr=f_expr,
        g_expr=g_expr,
        g_prime_expr=g_prime_expr,
        g_auto_derived=g_auto_derived,
        f_func=_make_numeric_func(f_expr),
        g_func=_make_numeric_func(g_expr),
        g_prime_func=_make_numeric_func(g_prime_expr),
    )


def classify_derivative(
    deriv_val: Optional[float],
    eval_error: Optional[str],
    point_label: str,
    point_value: float,
) -> ConvergenceAnalysis:
    """Classify local fixed-point behavior based on the value of g'(x)."""
    if deriv_val is None or eval_error is not None:
        return ConvergenceAnalysis(
            point_label=point_label,
            point_value=point_value,
            derivative_value=None,
            classification="Undefined / Domain Error",
            is_convergent=None,
            is_oscillatory=None,
            severity="red",
            explanation=eval_error or "g'(x) could not be evaluated at this point.",
        )

    tol = 1e-12
    if abs(deriv_val) <= tol:
        return ConvergenceAnalysis(
            point_label=point_label,
            point_value=point_value,
            derivative_value=deriv_val,
            classification="Convergent (Monotonic)",
            is_convergent=True,
            is_oscillatory=False,
            severity="green",
            explanation="g'(x) = 0 satisfies 0 <= g'(x) < 1 (rapid / superlinear local convergence).",
        )

    if 0.0 < deriv_val < 1.0 - tol:
        slow = deriv_val >= 0.85
        return ConvergenceAnalysis(
            point_label=point_label,
            point_value=point_value,
            derivative_value=deriv_val,
            classification="Convergent (Monotonic)",
            is_convergent=True,
            is_oscillatory=False,
            severity="yellow" if slow else "green",
            explanation=(
                "0 <= g'(x) < 1: errors decrease monotonically without changing sign"
                + (" (near 1, so convergence will be slow)." if slow else ".")
            ),
        )

    if -1.0 + tol < deriv_val < 0.0:
        slow = abs(deriv_val) >= 0.85
        return ConvergenceAnalysis(
            point_label=point_label,
            point_value=point_value,
            derivative_value=deriv_val,
            classification="Convergent (Oscillatory / Spiral)",
            is_convergent=True,
            is_oscillatory=True,
            severity="yellow" if slow else "green",
            explanation=(
                "-1 < g'(x) <= 0: estimates spiral/oscillate with decreasing amplitude toward the root"
                + (" (near -1, so convergence will be slow)." if slow else ".")
            ),
        )

    if abs(abs(deriv_val) - 1.0) <= tol:
        sign_text = "+1" if deriv_val > 0 else "-1"
        return ConvergenceAnalysis(
            point_label=point_label,
            point_value=point_value,
            derivative_value=deriv_val,
            classification=f"Borderline / Marginal (g'(x) = {sign_text})",
            is_convergent=None,
            is_oscillatory=deriv_val < 0,
            severity="yellow",
            explanation=(
                f"|g'({point_label})| = 1 lies on the stability boundary; "
                "convergence requires |g'(x)| < 1 in a neighborhood of the root."
            ),
        )

    if deriv_val > 1.0 + tol:
        return ConvergenceAnalysis(
            point_label=point_label,
            point_value=point_value,
            derivative_value=deriv_val,
            classification="Divergent (Monotonic)",
            is_convergent=False,
            is_oscillatory=False,
            severity="red",
            explanation="g'(x) > 1: errors amplify monotonically away from the fixed point.",
        )

    return ConvergenceAnalysis(
        point_label=point_label,
        point_value=point_value,
        derivative_value=deriv_val,
        classification="Divergent (Oscillatory)",
        is_convergent=False,
        is_oscillatory=True,
        severity="red",
        explanation="g'(x) < -1: estimates oscillate with growing amplitude away from the fixed point.",
    )


def compute_reference_root(
    problem: ParsedProblem,
    preferred_guess: float,
    x0: float,
) -> Optional[float]:
    """Attempt to locate a high-precision real root of ``f(x) = 0`` via ``sympy.nsolve``."""
    raw_candidates: list[float] = []
    for candidate in (
        preferred_guess,
        x0,
        preferred_guess + 0.25,
        preferred_guess - 0.25,
        x0 + 0.5,
        x0 - 0.5,
        0.5,
        1.0,
        -1.0,
        0.0,
    ):
        if math.isfinite(candidate) and abs(candidate) < 1e6:
            if all(abs(candidate - existing) > 1e-6 for existing in raw_candidates):
                raw_candidates.append(candidate)

    for guess in raw_candidates:
        try:
            sol = sp.nsolve(problem.f_expr, problem.x_sym, guess, tol=1e-14, maxsteps=60)
            sol_c = complex(sol)
            if abs(sol_c.imag) > 1e-10:
                continue
            root_val = float(sol_c.real)
            if not math.isfinite(root_val):
                continue
            residual, err = safe_eval_expr(problem.f_func, problem.f_expr, root_val)
            if err is None and residual is not None and abs(residual) < 1e-6:
                return root_val
        except Exception:
            continue

    return None


def _compute_true_error(x_val: float, true_root: Optional[float]) -> Optional[float]:
    """Compute true percent relative error |(x_true - x_i) / x_true| * 100% safely."""
    if true_root is None or abs(true_root) < 1e-15:
        return None
    return abs((true_root - x_val) / true_root) * 100.0


def solve_fixed_point(
    problem: ParsedProblem,
    x0: float,
    es: float = 0.5,
    max_iter: int = 10,
    true_root: Optional[float] = None,
    true_root_source: Optional[str] = None,
    stop_on_tolerance: bool = True,
) -> SolverResult:
    """Execute Simple Fixed-Point Iteration ``x_{i+1} = g(x_i)`` with full diagnostics."""
    # Pre-iteration convergence analysis at x0
    g_prime_x0, g_prime_x0_err = safe_eval_expr(
        problem.g_prime_func, problem.g_prime_expr, x0
    )
    pre_analysis = classify_derivative(g_prime_x0, g_prime_x0_err, "x₀", x0)

    records: list[IterationRecord] = [
        IterationRecord(i=0, x_prev=None, x_curr=x0)
    ]

    x_curr = x0
    last_valid_x = x0
    status = "max_iter_reached"
    status_message = (
        f"Reached maximum of {max_iter} iteration(s) without satisfying εₐ < {es:g}%."
    )
    total_iterations = 0

    for step in range(1, max_iter + 1):
        sub_str = format_substitution(problem.g_expr, x_curr, decimals=6)
        x_next, eval_err = safe_eval_expr(
            problem.g_func, problem.g_expr, x_curr, limit=DIVERGENCE_LIMIT
        )

        if eval_err is not None:
            total_iterations = step
            status = "diverged"
            status_message = f"Method diverged at iteration {step}: {eval_err}"
            records.append(
                IterationRecord(
                    i=step,
                    x_prev=x_curr,
                    x_curr=x_next,
                    substitution_str=sub_str,
                    diverged=True,
                    divergence_reason=eval_err,
                )
            )
            break

        assert x_next is not None
        total_iterations = step
        last_valid_x = x_next

        # Approximate percent relative error ε_a = |(x_{i+1} - x_i) / x_{i+1}| * 100%
        ea: Optional[float]
        ea_note: Optional[str] = None
        if abs(x_next) < 1e-15:
            if abs(x_next - x_curr) < 1e-15:
                ea = 0.0
            else:
                ea = None
                ea_note = "Undefined (division by x_{i+1} = 0)"
        else:
            ea = abs((x_next - x_curr) / x_next) * 100.0

        met_tol = ea is not None and ea < es

        records.append(
            IterationRecord(
                i=step,
                x_prev=x_curr,
                x_curr=x_next,
                substitution_str=sub_str,
                ea=ea,
                ea_note=ea_note,
                met_tolerance=met_tol,
            )
        )

        x_curr = x_next

        if stop_on_tolerance and met_tol:
            status = "converged"
            status_message = (
                f"Converged in {step} iteration(s): εₐ = {ea:.4f}% < εₛ = {es:g}%."
            )
            break

    # Resolve reference true root (either provided or discovered via sympy.nsolve)
    resolved_true_root = true_root
    resolved_source = true_root_source or (
        "User-Provided" if true_root is not None else "Unavailable"
    )
    if resolved_true_root is None:
        discovered = compute_reference_root(problem, last_valid_x, x0)
        if discovered is not None:
            resolved_true_root = discovered
            resolved_source = "SymPy nsolve"

    # Populate true relative error ε_t and linear convergence ratio |ε_t|_i / |ε_t|_{i-1}
    if resolved_true_root is not None:
        prev_et: Optional[float] = None
        for rec in records:
            if rec.x_curr is not None and not rec.diverged:
                rec.et = _compute_true_error(rec.x_curr, resolved_true_root)
                if rec.i >= 1 and rec.et is not None and prev_et is not None and prev_et > 1e-15:
                    rec.et_ratio = rec.et / prev_et
                prev_et = rec.et
            else:
                prev_et = None

    # Final residual f(x_root)
    final_residual, _ = safe_eval_expr(problem.f_func, problem.f_expr, last_valid_x)

    # Post-iteration convergence analysis at x_final (and x_true if known)
    g_prime_final, g_prime_final_err = safe_eval_expr(
        problem.g_prime_func, problem.g_prime_expr, last_valid_x
    )
    post_analysis = classify_derivative(
        g_prime_final, g_prime_final_err, "x_final", last_valid_x
    )

    true_root_analysis: Optional[ConvergenceAnalysis] = None
    if resolved_true_root is not None:
        g_prime_true, g_prime_true_err = safe_eval_expr(
            problem.g_prime_func, problem.g_prime_expr, resolved_true_root
        )
        true_root_analysis = classify_derivative(
            g_prime_true, g_prime_true_err, "x_true", resolved_true_root
        )

    # Enrich divergence message if |g'(x)| > 1 explains the behavior
    if status == "diverged" and pre_analysis.derivative_value is not None:
        if abs(pre_analysis.derivative_value) > 1.0:
            status_message += (
                f" Note: |g'(x₀)| = {abs(pre_analysis.derivative_value):.4f} > 1 "
                f"({pre_analysis.classification})."
            )

    return SolverResult(
        problem=problem,
        x0=x0,
        es=es,
        max_iter=max_iter,
        records=records,
        status=status,
        status_message=status_message,
        estimated_root=last_valid_x,
        total_iterations=total_iterations,
        final_residual=final_residual,
        true_root=resolved_true_root,
        true_root_source=resolved_source,
        pre_analysis=pre_analysis,
        post_analysis=post_analysis,
        true_root_analysis=true_root_analysis,
    )
