"""Model Health page (Day 21): the ops view of the ML system.

Four sections, all rendered from the :class:`DashboardDataService` snapshot by
:func:`build_content`:

* **Model performance**: ensemble F1 / precision / recall gauges, F1 per model, and today's alerts
  by anomaly type.
* **Feedback analytics**: false-positive rate per day against its 10 % target, analyst activity by
  day and hour, and the noisiest entities (threshold candidates).
* **Retraining history**: a timeline of runs, the audit-log table and the change since the last
  promotion.
* **Champion / challenger**: shown only while a challenger waits for a decision; Promote and
  Reject call ``POST /api/v1/models/promote`` and ``/reject`` after a confirmation.

A drift banner tops the page while the feedback drift detector has fired within the last hour.

The layout is a function (the first paint already holds data), so every callback uses
``prevent_initial_call=True``. After that the ``live-interval`` tick re-renders the page, and only
when the snapshot version changed, so an idle system costs the browser nothing. The Promote and
Reject buttons live in the static layout and only their section is shown or hidden, so a refresh
never resets a click.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import flask
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from src.dashboard.components.layout import page_header
from src.dashboard.components.model_health_widgets import (
    anomaly_type_chart,
    challenger_summary,
    comparison_table,
    drift_banner,
    feedback_heatmap,
    fp_trend_chart,
    gauge,
    live_precision_caption,
    model_f1_chart,
    pr_overlay,
    retrain_delta_card,
    retraining_log,
    retraining_notice,
    retraining_timeline,
    top_fp_table,
)
from src.dashboard.components.overview_widgets import card_with_title
from src.dashboard.components.theme import MUTED_TEXT
from src.dashboard.data_service import DashboardDataService

dash.register_page(
    __name__,
    path="/model-health",
    name="Model Health",
    title="Model Health · Supply Chain Anomaly Monitor",
    order=5,
)

GAUGE_IDS = ("mh-gauge-f1", "mh-gauge-precision", "mh-gauge-recall")


def _service() -> DashboardDataService:
    """Return the dashboard's data service (``create_dash_app`` attaches it to the app)."""
    return flask.current_app.extensions["scad_service"]


def build_content(service: DashboardDataService) -> dict[str, Any]:
    """Render every dynamic part of the page from the current snapshot.

    Returns:
        A dict with the ``version`` and one entry per dynamic element: ``drift``, ``gauges``
        (F1, precision, recall figures), ``caption``, ``f1_chart``, ``type_chart``, ``fp_trend``,
        ``heatmap``, ``top_fp``, ``notice``, ``timeline``, ``log``, ``delta``, ``has_challenger``,
        ``cc_title``, ``cc_table`` and ``cc_pr``.
    """
    snap = service.snapshot()
    scores = service.get_model_scores()
    ensemble = scores["ensemble"]
    health = service.model_health()
    last_alert = (snap.feedback.get("drift") or {}).get("last_alert")
    retraining = service.get_retraining()
    challenger = retraining["challenger"]
    return {
        "version": snap.version,
        "drift": drift_banner(health["drift_alert_active"], last_alert),
        "gauges": (
            gauge("Ensemble F1", ensemble["f1"]),
            gauge("Ensemble precision", ensemble["precision"]),
            gauge("Ensemble recall", ensemble["recall"]),
        ),
        "caption": live_precision_caption(scores["live"]),
        "f1_chart": model_f1_chart(scores["f1"], scores["versions"]),
        "type_chart": anomaly_type_chart(service.get_anomaly_type_breakdown()),
        "fp_trend": fp_trend_chart(service.get_feedback_trend()),
        "heatmap": feedback_heatmap(service.get_feedback_heatmap()),
        "top_fp": top_fp_table(service.get_top_fp_entities()),
        "notice": retraining_notice(retraining["enabled"], retraining["runs"]) or "",
        "timeline": retraining_timeline(retraining["runs"]),
        "log": retraining_log(retraining["runs"]),
        "delta": retrain_delta_card(service.get_retraining_delta()),
        "has_challenger": challenger is not None,
        "cc_title": challenger_summary(challenger) if challenger else "",
        "cc_table": comparison_table(challenger) if challenger else None,
        "cc_pr": pr_overlay(challenger) if challenger else go.Figure(),
    }


def _graph(ident: str, figure: go.Figure, height: int = 300) -> dcc.Graph:
    """Return a :class:`dcc.Graph` without a mode bar (fast to render and resize)."""
    return dcc.Graph(
        id=ident,
        figure=figure,
        config={"displayModeBar": False, "responsive": True},
        style={"height": f"{height}px"},
    )


def _section(title: str) -> html.H4:
    return html.H4(title, className="mt-3 mb-2 h5")


def _col(card: Any, **widths: int) -> dbc.Col:
    return dbc.Col(card, className="mb-3", **widths)


def layout() -> html.Div:
    """Page layout, filled from the latest snapshot at the time of navigation."""
    content = build_content(_service())
    f1_fig, precision_fig, recall_fig = content["gauges"]
    return html.Div(
        [
            dcc.Store(id="mh-version", data=content["version"]),
            dcc.Store(id="mh-pending-action"),
            dcc.ConfirmDialog(id="mh-confirm", message=""),
            page_header("Model Health", "Model quality, feedback and retraining · times in UTC"),
            html.Div(content["drift"], id="mh-drift"),
            _section("Model performance"),
            dbc.Row(
                [
                    _col(
                        card_with_title("Ensemble F1", _graph("mh-gauge-f1", f1_fig, 210)),
                        xs=12,
                        md=4,
                    ),
                    _col(
                        card_with_title(
                            "Ensemble precision", _graph("mh-gauge-precision", precision_fig, 210)
                        ),
                        xs=12,
                        md=4,
                    ),
                    _col(
                        card_with_title(
                            "Ensemble recall", _graph("mh-gauge-recall", recall_fig, 210)
                        ),
                        xs=12,
                        md=4,
                    ),
                ]
            ),
            html.Div(
                content["caption"],
                id="mh-live-caption",
                className="small mb-3",
                style={"color": MUTED_TEXT},
            ),
            dbc.Row(
                [
                    _col(
                        card_with_title(
                            "F1 by model (registry evaluation)",
                            _graph("mh-f1-chart", content["f1_chart"]),
                        ),
                        xs=12,
                        lg=7,
                    ),
                    _col(
                        card_with_title(
                            "Anomalies detected today, by type",
                            _graph("mh-type-chart", content["type_chart"]),
                        ),
                        xs=12,
                        lg=5,
                    ),
                ]
            ),
            _section("Feedback analytics"),
            dbc.Row(
                [
                    _col(
                        card_with_title(
                            "False-positive rate, last 7 days",
                            _graph("mh-fp-trend", content["fp_trend"]),
                        ),
                        xs=12,
                        lg=6,
                    ),
                    _col(
                        card_with_title(
                            "Feedback volume by day and hour",
                            _graph("mh-heatmap", content["heatmap"]),
                        ),
                        xs=12,
                        lg=6,
                    ),
                ]
            ),
            dbc.Row(
                _col(
                    card_with_title(
                        "Noisiest entities (candidates for a threshold adjustment)",
                        html.Div(content["top_fp"], id="mh-top-fp"),
                    ),
                    xs=12,
                )
            ),
            _section("Retraining history"),
            html.Div(
                content["notice"],
                id="mh-retrain-notice",
                className="alert alert-secondary py-2 small",
                hidden=not content["notice"],
            ),
            dbc.Row(
                _col(
                    card_with_title(
                        "Retraining runs", _graph("mh-timeline", content["timeline"], 300)
                    ),
                    xs=12,
                )
            ),
            dbc.Row(
                [
                    _col(
                        card_with_title("Retraining log", html.Div(content["log"], id="mh-log")),
                        xs=12,
                        xl=8,
                    ),
                    _col(
                        card_with_title(
                            "Current vs baseline", html.Div(content["delta"], id="mh-delta-wrap")
                        ),
                        xs=12,
                        xl=4,
                    ),
                ]
            ),
            html.Div(
                [
                    _section("Champion / challenger"),
                    html.Div(
                        content["cc_title"],
                        id="mh-cc-title",
                        className="small mb-2",
                        style={"color": MUTED_TEXT},
                    ),
                    dbc.Row(
                        [
                            _col(
                                card_with_title(
                                    "Metrics", html.Div(content["cc_table"], id="mh-cc-table-wrap")
                                ),
                                xs=12,
                                lg=5,
                            ),
                            _col(
                                card_with_title(
                                    "Precision-recall curves", _graph("mh-cc-pr", content["cc_pr"])
                                ),
                                xs=12,
                                lg=7,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Promote challenger",
                                id="mh-promote",
                                color="success",
                                size="sm",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Reject challenger", id="mh-reject", color="danger", size="sm"
                            ),
                        ],
                        className="mb-2",
                    ),
                ],
                id="mh-cc-section",
                style={} if content["has_challenger"] else {"display": "none"},
            ),
            html.Div(id="mh-review-notice"),
        ],
        id="mh-page",
    )


# --------------------------------------------------------------------------- #
# Live refresh
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("mh-drift", "children"),
    Output("mh-gauge-f1", "figure"),
    Output("mh-gauge-precision", "figure"),
    Output("mh-gauge-recall", "figure"),
    Output("mh-live-caption", "children"),
    Output("mh-f1-chart", "figure"),
    Output("mh-type-chart", "figure"),
    Output("mh-fp-trend", "figure"),
    Output("mh-heatmap", "figure"),
    Output("mh-top-fp", "children"),
    Output("mh-retrain-notice", "children"),
    Output("mh-retrain-notice", "hidden"),
    Output("mh-timeline", "figure"),
    Output("mh-log", "children"),
    Output("mh-delta-wrap", "children"),
    Output("mh-cc-section", "style"),
    Output("mh-cc-title", "children"),
    Output("mh-cc-table-wrap", "children"),
    Output("mh-cc-pr", "figure"),
    Output("mh-version", "data"),
    Input("live-interval", "n_intervals"),
    State("mh-version", "data"),
    prevent_initial_call=True,
)
def refresh_model_health(_n: int | None, shown_version: int | None):
    """Re-render the page on each tick, unless the snapshot is the one already shown."""
    service = _service()
    if shown_version is not None and service.snapshot().version == shown_version:
        return (no_update,) * 20
    content = build_content(service)
    f1_fig, precision_fig, recall_fig = content["gauges"]
    return (
        content["drift"],
        f1_fig,
        precision_fig,
        recall_fig,
        content["caption"],
        content["f1_chart"],
        content["type_chart"],
        content["fp_trend"],
        content["heatmap"],
        content["top_fp"],
        content["notice"],
        not content["notice"],
        content["timeline"],
        content["log"],
        content["delta"],
        {} if content["has_challenger"] else {"display": "none"},
        content["cc_title"],
        content["cc_table"],
        content["cc_pr"],
        content["version"],
    )


# --------------------------------------------------------------------------- #
# Promote / reject (with a confirmation: promotion swaps the serving model)
# --------------------------------------------------------------------------- #
CONFIRM_TEXT = {
    "promote": "Promote the challenger? It will replace the champion and serve live traffic.",
    "reject": "Reject the challenger? It will be discarded and the champion keeps serving.",
}


@dash.callback(
    Output("mh-confirm", "message"),
    Output("mh-confirm", "displayed"),
    Output("mh-pending-action", "data"),
    Input("mh-promote", "n_clicks"),
    Input("mh-reject", "n_clicks"),
    prevent_initial_call=True,
)
def ask_confirmation(_promote: int | None, _reject: int | None):
    """Open the confirmation dialog for the button that was pressed."""
    action = {"mh-promote": "promote", "mh-reject": "reject"}.get(str(dash.ctx.triggered_id))
    if action is None:
        return no_update, no_update, no_update
    return CONFIRM_TEXT[action], True, action


@dash.callback(
    Output("mh-review-notice", "children"),
    Input("mh-confirm", "submit_n_clicks"),
    State("mh-pending-action", "data"),
    running=[
        (Output("mh-promote", "disabled"), True, False),
        (Output("mh-reject", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def review_challenger(_submitted: int | None, action: str | None):
    """Send the confirmed decision to the API and report the outcome."""
    if action not in ("promote", "reject"):
        return no_update
    result = _service().review_challenger(action)
    return dbc.Alert(
        result["message"],
        color="success" if result["ok"] else "danger",
        id="mh-review-alert",
        dismissable=True,
        duration=8000,
        className="mt-2",
    )
