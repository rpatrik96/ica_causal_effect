"""Pytest configuration and fixtures."""

import os
import sys
import types

# Add parent directory to path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _install_torch_stub() -> None:
    """Install a minimal torch stub when torch is not available.

    The real torch package has no wheel for Python 3.13 on macOS, so we build
    a lightweight stub that satisfies import-time attribute lookups used by the
    codebase (nn, nn.functional, optim, utils.data, cuda, autograd,
    distributions).  The stub uses real :class:`types.ModuleType` objects so
    that :func:`issubclass` checks in ``scipy._lib.array_api_compat`` work
    correctly (MagicMock objects break those checks).
    """
    try:
        import torch  # noqa: F401 — already installed, nothing to do

        return
    except ModuleNotFoundError:
        pass

    torch_mod = types.ModuleType("torch")

    class Tensor:  # minimal stand-in
        pass

    torch_mod.Tensor = Tensor
    torch_mod.is_tensor = lambda x: False
    sys.modules["torch"] = torch_mod

    _submodules = [
        "torch.nn",
        "torch.nn.functional",
        "torch.optim",
        "torch.utils",
        "torch.utils.data",
        "torch.cuda",
        "torch.autograd",
        "torch.distributions",
    ]
    for sub in _submodules:
        m = types.ModuleType(sub)
        sys.modules[sub] = m
        parts = sub.split(".")
        parent = sys.modules[".".join(parts[:-1])]
        setattr(parent, parts[-1], m)


_install_torch_stub()


def _install_seaborn_stub() -> None:
    """Install a minimal seaborn stub when seaborn is not available."""
    try:
        import seaborn  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    sns_mod = types.ModuleType("seaborn")
    sns_mod.set_theme = lambda *a, **kw: None
    sns_mod.set_context = lambda *a, **kw: None
    sns_mod.set_style = lambda *a, **kw: None
    sns_mod.set_palette = lambda *a, **kw: None
    sns_mod.color_palette = lambda *a, **kw: ["#1f77b4", "#ff7f0e", "#2ca02c"]
    sns_mod.heatmap = lambda *a, **kw: None
    sns_mod.barplot = lambda *a, **kw: None
    sns_mod.boxplot = lambda *a, **kw: None
    sns_mod.lineplot = lambda *a, **kw: None
    sns_mod.scatterplot = lambda *a, **kw: None
    sns_mod.kdeplot = lambda *a, **kw: None
    sns_mod.histplot = lambda *a, **kw: None
    sns_mod.despine = lambda *a, **kw: None
    sys.modules["seaborn"] = sns_mod


_install_seaborn_stub()


def _install_tueplots_stub() -> None:
    """Install a minimal tueplots stub for plotting modules that import it."""
    try:
        import tueplots  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    tp = types.ModuleType("tueplots")
    bundles = types.ModuleType("tueplots.bundles")
    for name in (
        "icml2022",
        "icml2024",
        "neurips2021",
        "neurips2023",
        "neurips2024",
        "aistats2025",
        "iclr2024",
        "beamer_moml",
        "beamer_dark",
        "jmlr2001",
    ):
        setattr(bundles, name, lambda *a, **kw: {})
    fonts = types.ModuleType("tueplots.fonts")
    for name in ("neurips2022", "icml2022"):
        setattr(fonts, name, lambda *a, **kw: {})
    figsizes = types.ModuleType("tueplots.figsizes")
    for name in ("icml2022", "neurips2024"):
        setattr(figsizes, name, lambda *a, **kw: {})
    tp.bundles = bundles
    tp.fonts = fonts
    tp.figsizes = figsizes
    sys.modules["tueplots"] = tp
    sys.modules["tueplots.bundles"] = bundles
    sys.modules["tueplots.fonts"] = fonts
    sys.modules["tueplots.figsizes"] = figsizes


_install_tueplots_stub()


# ---------------------------------------------------------------------------
# Joblib backend override
# ---------------------------------------------------------------------------
# Several tests trigger ``joblib.Parallel(n_jobs=-1, ...)`` deep in the
# experiment dispatcher.  The default ``loky`` backend forks worker
# processes that do NOT inherit the stubs above (torch / seaborn /
# tueplots).  Forcing the threading backend keeps everything in-process
# so the stubs apply.

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _force_joblib_threading_backend():
    """Force joblib to use the threading backend for every test."""
    try:
        import joblib

        with joblib.parallel_backend("threading", n_jobs=1):
            yield
    except Exception:
        yield
