"""Page building blocks shared by every dashboard page (Day 18)."""

from __future__ import annotations

from dash import html

from src.dashboard.components.theme import MUTED_TEXT


def page_header(title: str, subtitle: str = "") -> html.Div:
    """Heading of a page: title plus a muted one-line description."""
    return html.Div(
        [
            html.H2(title, id="page-title", className="mb-1"),
            html.P(subtitle, className="mb-3", style={"color": MUTED_TEXT}) if subtitle else None,
        ],
        className="page-header",
    )


def coming_soon(day: int, what: str) -> html.Div:
    """Build the placeholder body of a page that a later build day fills in."""
    return html.Div(
        f"{what} arrive on Day {day}. The live data layer behind this page is already running.",
        className="alert alert-secondary",
        id="page-placeholder",
    )
