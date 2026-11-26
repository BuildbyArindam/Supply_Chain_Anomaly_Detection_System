"""Sidebar navigation with the live connection status (Day 18)."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.theme import MUTED_TEXT, PANEL_DARK


def build_sidebar() -> html.Div:
    """Return the sidebar: brand, one link per registered page, live status.

    Call after the Dash app exists with ``use_pages=True`` so that ``dash.page_registry`` is full.
    """
    pages = sorted(dash.page_registry.values(), key=lambda p: (p.get("order", 99), p["name"]))
    links = [
        dbc.NavLink(
            page["name"],
            href=page["relative_path"],
            active="exact",
            id=f"nav-{page['name'].lower().replace(' ', '-')}",
        )
        for page in pages
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.H5("Supply Chain", className="mb-0"),
                    html.Small("Anomaly Monitor", style={"color": MUTED_TEXT}),
                ],
                className="mb-4",
            ),
            dbc.Nav(links, vertical=True, pills=True, id="sidebar-nav"),
            html.Hr(),
            html.Div(id="status-strip", children="Connecting…", className="small"),
            html.Div(
                id="sse-status",
                children="Live alerts: connecting…",
                className="small mt-2",
                style={"color": MUTED_TEXT},
            ),
        ],
        id="sidebar",
        className="sidebar",
        style={"backgroundColor": PANEL_DARK},
    )
