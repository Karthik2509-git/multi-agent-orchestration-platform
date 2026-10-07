"""Tests for the public API surface used by the hosted demo."""

import asyncio

from src.app import main
from src.app.core.config import Settings


def _route_paths(app):
    return {route.path for route in app.routes}


def test_production_docs_are_disabled_by_default(monkeypatch):
    settings = Settings(
        app_env="production",
        public_docs_enabled=False,
        checkpoint_backend="memory",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    app = main.create_app()

    paths = _route_paths(app)
    assert "/docs" not in paths
    assert "/redoc" not in paths
    assert "/openapi.json" in paths
    assert "/" in paths
    assert "/health" in paths


def test_public_docs_can_be_explicitly_enabled(monkeypatch):
    settings = Settings(
        app_env="production",
        public_docs_enabled=True,
        checkpoint_backend="memory",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    app = main.create_app()

    paths = _route_paths(app)
    assert "/docs" in paths
    assert "/redoc" in paths
    assert "/openapi.json" in paths


def test_root_landing_page_contains_live_api_links(monkeypatch):
    settings = Settings(
        app_env="production",
        public_docs_enabled=True,
        checkpoint_backend="memory",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    app = main.create_app()
    root_route = next(route for route in app.routes if route.path == "/")
    html = asyncio.run(root_route.endpoint())

    assert "Multi-Agent AI Orchestration Platform" in html
    assert 'href="/docs"' in html
    assert 'href="/redoc"' in html
    assert 'href="/openapi.json"' in html
    assert 'href="/health"' in html
