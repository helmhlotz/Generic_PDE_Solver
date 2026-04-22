import numpy as np

from inference_engine import InferenceEngine, SolveResult, SolverOption


_BC = {
    "left": {"type": "dirichlet", "value": "0"},
    "right": {"type": "dirichlet", "value": "0"},
    "bottom": {"type": "dirichlet", "value": "0"},
    "top": {"type": "dirichlet", "value": "0"},
}


def _dummy_result(method: str) -> SolveResult:
    zeros = np.zeros((4, 4), dtype=np.float32)
    return SolveResult(
        u=zeros,
        xx=zeros,
        yy=zeros,
        residual=0.0,
        bc_error=0.0,
        method=method,
    )


def test_fd_solver_type_routes_directly(monkeypatch):
    engine = InferenceEngine(SolverOption(solver_type="fd"))

    monkeypatch.setattr(engine, "_fd_path", lambda *args, **kwargs: _dummy_result("fd"))

    result = engine.solve("u_xx + u_yy = 0", _BC)

    assert result.method == "fd"
    assert result.route_reason == "solver_type=fd"


def test_deeponet_missing_artifact_falls_back_to_fd(monkeypatch):
    option = SolverOption(solver_type="deeponet")
    engine = InferenceEngine(option)

    monkeypatch.setattr(engine, "_fd_path", lambda *args, **kwargs: _dummy_result("fd"))

    result = engine.solve("u_xx + u_yy = 0", _BC)

    assert result.method == "fd_missing_artifact"
    assert "DeepONet artifact unavailable" in result.route_reason


def test_deeponet_ood_falls_back_to_fd(monkeypatch):
    option = SolverOption(solver_type="deeponet")
    engine = InferenceEngine(option)
    engine._has_deeponet_artifact = True
    engine._deeponet_model = object()
    engine._deeponet_n_points = engine.options.grid.n_points

    class _FakeOOD:
        def check(self, parsed_pde, bc_specs):
            return True, "outside manifest"

    engine._ood_detector = _FakeOOD()

    monkeypatch.setattr(engine, "_fd_path", lambda *args, **kwargs: _dummy_result("fd"))

    result = engine.solve("u_xx + u_yy = 0", _BC)

    assert result.method == "fd_fallback"
    assert result.is_ood is True
    assert result.ood_reason == "outside manifest"
    assert "OOD detected" in result.route_reason


def test_deeponet_resolution_mismatch_falls_back_to_fd(monkeypatch):
    option = SolverOption(solver_type="deeponet")
    engine = InferenceEngine(option)
    engine._has_deeponet_artifact = True
    engine._deeponet_model = object()
    engine._deeponet_n_points = 8
    engine._ood_detector = None

    monkeypatch.setattr(engine, "_fd_path", lambda *args, **kwargs: _dummy_result("fd"))

    result = engine.solve("u_xx + u_yy = 0", _BC, n_points=4)

    assert result.method == "fd_resolution_mismatch"
    assert "resolution mismatch" in result.route_reason.lower()


def test_deeponet_offline_route_uses_loaded_operator(monkeypatch):
    option = SolverOption(solver_type="deeponet")
    engine = InferenceEngine(option)
    engine._has_deeponet_artifact = True
    engine._deeponet_model = object()
    engine._deeponet_n_points = engine.options.grid.n_points
    engine._ood_detector = None

    monkeypatch.setattr(engine, "_deeponet_path", lambda *args, **kwargs: _dummy_result("deeponet"))

    result = engine.solve("u_xx + u_yy = 0", _BC)

    assert result.method == "deeponet"
    assert result.route_reason == "Using loaded DeepONet operator artifact."
