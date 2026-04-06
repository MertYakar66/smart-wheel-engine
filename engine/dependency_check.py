"""
Environment Parity Gate

Ensures required dependencies are installed and available before running
critical modules. This prevents runtime errors from missing packages and
ensures test/production environment parity.

Usage:
    from engine.dependency_check import check_dependencies, require_dependencies

    # Check all dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {missing}")

    # Require specific dependencies (raises ImportError if missing)
    require_dependencies(["pandas", "numpy", "scipy"])

    # Decorator for functions requiring specific deps
    @require_dependencies(["pydantic"])
    def my_function():
        from pydantic import BaseModel
        ...
"""

import importlib.util
import sys
from functools import wraps
from typing import Callable


# Core dependencies required for all engine modules
CORE_DEPENDENCIES = [
    "pandas",
    "numpy",
    "scipy",
]

# Extended dependencies by module
MODULE_DEPENDENCIES = {
    "option_pricer": ["pandas", "numpy", "scipy"],
    "risk_manager": ["pandas", "numpy", "scipy"],
    "monte_carlo": ["pandas", "numpy", "scipy"],
    "volatility_surface": ["pandas", "numpy", "scipy"],
    "stress_testing": ["pandas", "numpy", "scipy"],
    "portfolio_tracker": ["pandas", "numpy"],
    "event_calendar": ["pandas"],
    "regime_detector": ["pandas", "numpy"],
    "signals": ["pandas", "numpy"],
    "data_pipeline": ["pandas", "numpy", "pydantic"],
    "financial_news": ["pydantic", "requests"],
    "backtests": ["pandas", "numpy", "scipy"],
    "dashboard": ["pandas", "numpy", "streamlit"],
}

# Optional dependencies (warn but don't fail)
OPTIONAL_DEPENDENCIES = {
    "yfinance": "Live market data fetching",
    "arch": "GARCH volatility modeling",
    "statsmodels": "Statistical modeling",
    "scikit-learn": "Machine learning features",
    "matplotlib": "Plotting and visualization",
    "seaborn": "Statistical visualization",
    "streamlit": "Dashboard interface",
}


def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed and importable."""
    # Handle package name variations (e.g., scikit-learn vs sklearn)
    import_name = package_name.replace("-", "_")
    if package_name == "scikit-learn":
        import_name = "sklearn"

    spec = importlib.util.find_spec(import_name)
    return spec is not None


def check_dependencies(
    required: list[str] | None = None,
    include_optional: bool = False,
) -> dict[str, list[str]]:
    """
    Check which dependencies are missing.

    Args:
        required: List of package names to check. If None, checks CORE_DEPENDENCIES.
        include_optional: Also check OPTIONAL_DEPENDENCIES.

    Returns:
        Dict with 'missing_required' and 'missing_optional' lists.
    """
    if required is None:
        required = CORE_DEPENDENCIES

    result = {
        "missing_required": [],
        "missing_optional": [],
    }

    for pkg in required:
        if not is_package_installed(pkg):
            result["missing_required"].append(pkg)

    if include_optional:
        for pkg in OPTIONAL_DEPENDENCIES:
            if not is_package_installed(pkg):
                result["missing_optional"].append(pkg)

    return result


def check_module_dependencies(module_name: str) -> list[str]:
    """
    Check dependencies for a specific module.

    Args:
        module_name: Name of the module (e.g., 'option_pricer')

    Returns:
        List of missing package names.
    """
    deps = MODULE_DEPENDENCIES.get(module_name, CORE_DEPENDENCIES)
    return [pkg for pkg in deps if not is_package_installed(pkg)]


def require_dependencies(
    packages: list[str] | None = None,
    module_name: str | None = None,
) -> Callable:
    """
    Decorator that checks dependencies before function execution.

    Args:
        packages: List of required packages.
        module_name: If provided, use MODULE_DEPENDENCIES[module_name].

    Raises:
        ImportError: If any required package is missing.

    Example:
        @require_dependencies(["pydantic", "requests"])
        def fetch_news():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            required = packages
            if module_name:
                required = MODULE_DEPENDENCIES.get(module_name, [])
            if required is None:
                required = CORE_DEPENDENCIES

            missing = [pkg for pkg in required if not is_package_installed(pkg)]
            if missing:
                raise ImportError(
                    f"Missing required dependencies for {func.__name__}: {missing}. "
                    f"Install with: pip install {' '.join(missing)}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_environment(
    strict: bool = True,
    modules: list[str] | None = None,
) -> bool:
    """
    Validate that the environment has all required dependencies.

    This should be called at application startup or before running tests.

    Args:
        strict: If True, raise ImportError on missing required deps.
        modules: List of module names to check. If None, checks all.

    Returns:
        True if all required dependencies are present.

    Raises:
        ImportError: If strict=True and dependencies are missing.
    """
    all_missing = set()

    if modules:
        for module in modules:
            missing = check_module_dependencies(module)
            all_missing.update(missing)
    else:
        # Check all module dependencies
        for module, deps in MODULE_DEPENDENCIES.items():
            for pkg in deps:
                if not is_package_installed(pkg):
                    all_missing.add(pkg)

    if all_missing:
        msg = (
            f"Environment validation failed. Missing packages: {sorted(all_missing)}\n"
            f"Install with: pip install {' '.join(sorted(all_missing))}"
        )
        if strict:
            raise ImportError(msg)
        else:
            print(f"WARNING: {msg}")
            return False

    return True


def get_environment_report() -> dict:
    """
    Generate a detailed environment report.

    Returns:
        Dict with installed/missing packages and versions.
    """
    report = {
        "python_version": sys.version,
        "core_packages": {},
        "optional_packages": {},
        "all_installed": True,
    }

    # Check core dependencies
    for pkg in CORE_DEPENDENCIES:
        installed = is_package_installed(pkg)
        report["core_packages"][pkg] = {
            "installed": installed,
            "version": _get_package_version(pkg) if installed else None,
        }
        if not installed:
            report["all_installed"] = False

    # Check optional dependencies
    for pkg, description in OPTIONAL_DEPENDENCIES.items():
        installed = is_package_installed(pkg)
        report["optional_packages"][pkg] = {
            "installed": installed,
            "version": _get_package_version(pkg) if installed else None,
            "description": description,
        }

    return report


def _get_package_version(package_name: str) -> str | None:
    """Get installed version of a package."""
    try:
        import_name = package_name.replace("-", "_")
        if package_name == "scikit-learn":
            import_name = "sklearn"

        module = importlib.import_module(import_name)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return None


# Convenience function for pytest conftest.py
def pytest_environment_check():
    """
    Check environment at pytest startup.

    Add to conftest.py:
        from engine.dependency_check import pytest_environment_check
        pytest_environment_check()
    """
    try:
        validate_environment(strict=True)
    except ImportError as e:
        import pytest
        pytest.exit(str(e), returncode=1)
