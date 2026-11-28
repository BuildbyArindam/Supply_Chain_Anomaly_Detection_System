"""Alerts page (Day 20): filterable table, bulk feedback, detail modal and CSV export.

* Filters run on the server (``DashboardDataService.query_alerts``), so the browser only gets the
  matching rows.
* The table follows the live data. While rows are selected it stops updating, so a selection
  is never lost to a refresh; it resumes when the selection is cleared.
* "Mark as false positive" sends one ``POST /api/v1/feedback`` per selected alert.
* A click on a row opens its full record with the SHAP explanation.
* "Export CSV" downloads every field of the filtered alerts (not only the visible columns).

The layout is a function, so every callback uses ``prevent_initial_call=True``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import flask
from dash import Input, Output, State, dcc, html, no_update

from src.dashboard.components.alert_widgets import (
    DEFAULT_ANALYST,
    DEFAULT_RANGE_HOURS,
    ENTITY_OPTIONS,
    RANGE_OPTIONS,
    SEVERITY_OPTIONS,
    STATUS_OPTIONS,
    TYPE_OPTIONS,
    alert_detail,
    alerts_to_csv,
    column_defs,
    feedback_notice,
    grid_options,
    grid_rows,
)
from src.dashboard.components.layout import page_header
from src.dashboard.components.theme import MUTED_TEXT
from src.dashboard.data_service import DashboardDataService

dash.register_page(
    __name__, path="/alerts", name="Alerts", title="Alerts · Supply Chain Anomaly Monitor", order=4
)

FILTER_INPUTS = (
    Input("alerts-severity", "value"),
    Input("alerts-type", "value"),
    Input("alerts-entity-type", "value"),
    Input("alerts-range", "value"),
    Input("alerts-status", "value"),
)
FILTER_STATES = tuple(State(i.component_id, i.component_property) for i in FILTER_INPUTS)


def _service() -> DashboardDataService:
    """Return the dashboard's data service (``create_dash_app`` attaches it to the app)."""
    return flask.current_app.extensions["scad_service"]


def _filtered(
    service: DashboardDataService,
    severities: list[str] | None,
    types: list[str] | None,
    entity_types: list[str] | None,
    hours: float | None,
    status: str | None,
) -> list[dict[str, Any]]:
    return service.query_alerts(
        severities=severities,
        anomaly_types=types,
        entity_types=entity_types,
        window_hours=hours or None,
        status=status,
    )


def _summary(shown: int, total: int, hours: float | None) -> str:
    scope = f"last {hours:g} h" if hours else "all stored"
    return f"Showing {shown:,} alert{'s' if shown != 1 else ''} ({scope}) of {total:,} stored"


def _signature(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [[r["event_id"], r["severity"], r["status"]] for r in rows]


def _dropdown(ident: str, label: str, options: Any, *, multi: bool, value: Any = None) -> dbc.Col:
    return dbc.Col(
        [
            dbc.Label(label, html_for=ident, className="small"),
            dcc.Dropdown(
                id=ident,
                options=options,
                value=value,
                multi=multi,
                clearable=multi,
                placeholder="Any",
                className="dash-dark-dropdown",
            ),
        ],
        xs=12,
        sm=6,
        lg=True,
        className="mb-2",
    )


def layout() -> html.Div:
    """Page layout, filled from the latest snapshot at the time of navigation."""
    service = _service()
    rows = grid_rows(_filtered(service, None, None, None, DEFAULT_RANGE_HOURS, "all"))
    total = len(service.snapshot().alerts)
    return html.Div(
        [
            dcc.Store(id="alerts-sig", data=_signature(rows)),
            dcc.Store(id="alert-modal-event"),
            dcc.Download(id="alerts-download"),
            page_header("Alerts", "Filter, review and give feedback on alerts · times in UTC"),
            dbc.Row(
                [
                    _dropdown("alerts-severity", "Severity", SEVERITY_OPTIONS, multi=True),
                    _dropdown("alerts-type", "Anomaly type", TYPE_OPTIONS, multi=True),
                    _dropdown("alerts-entity-type", "Entity type", ENTITY_OPTIONS, multi=True),
                    _dropdown(
                        "alerts-range",
                        "Time range",
                        RANGE_OPTIONS,
                        multi=False,
                        value=DEFAULT_RANGE_HOURS,
                    ),
                    _dropdown("alerts-status", "Status", STATUS_OPTIONS, multi=False, value="all"),
                ],
                className="mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                _summary(len(rows), total, DEFAULT_RANGE_HOURS),
                                id="alerts-summary",
                                className="small",
                                style={"color": MUTED_TEXT},
                            ),
                            html.Div(
                                "",
                                id="alerts-pause-note",
                                className="small text-warning",
                            ),
                        ],
                        xs=12,
                        lg=5,
                        className="mb-2",
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Input(
                                    id="alerts-analyst",
                                    value=DEFAULT_ANALYST,
                                    maxlength=64,
                                    size="sm",
                                    placeholder="Analyst id",
                                    style={"width": "170px"},
                                    className="me-2",
                                    debounce=True,
                                ),
                                dbc.Button(
                                    "Mark as false positive",
                                    id="alerts-bulk-fp",
                                    color="warning",
                                    size="sm",
                                    disabled=True,
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "Export CSV",
                                    id="alerts-export",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                ),
                            ],
                            className="d-flex flex-wrap justify-content-lg-end align-items-center",
                        ),
                        xs=12,
                        lg=7,
                        className="mb-2",
                    ),
                ]
            ),
            html.Div(id="alerts-feedback"),
            dag.AgGrid(
                id="alerts-grid",
                rowData=rows,
                columnDefs=column_defs(),
                defaultColDef={
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                    "flex": 1,
                },
                dashGridOptions=grid_options(),
                getRowId="params.data.event_id",
                style={"height": "620px", "width": "100%"},
                dangerously_allow_code=True,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Alert details", id="alert-modal-title")),
                    dbc.ModalBody(id="alert-modal-body"),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Mark false positive",
                                id="alert-modal-fp",
                                color="warning",
                                size="sm",
                            ),
                            dbc.Button(
                                "Confirm true positive",
                                id="alert-modal-tp",
                                color="success",
                                size="sm",
                            ),
                            dbc.Button(
                                "Close", id="alert-modal-close", color="secondary", size="sm"
                            ),
                        ]
                    ),
                ],
                id="alert-modal",
                size="xl",
                scrollable=True,
                is_open=False,
            ),
        ],
        id="alerts-page",
    )


# --------------------------------------------------------------------------- #
# Table content
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("alerts-grid", "rowData"),
    Output("alerts-summary", "children"),
    Output("alerts-sig", "data"),
    *FILTER_INPUTS,
    Input("live-interval", "n_intervals"),
    State("alerts-grid", "selectedRows"),
    State("alerts-sig", "data"),
    prevent_initial_call=True,
)
def update_grid(
    severities: list[str] | None,
    types: list[str] | None,
    entity_types: list[str] | None,
    hours: float | None,
    status: str | None,
    _n: int | None,
    selected: list[dict[str, Any]] | None,
    previous: list[list[str]] | None,
):
    """Filter on the server and refresh the table; skip the refresh while rows are selected."""
    by_interval = dash.ctx.triggered_id == "live-interval"
    if by_interval and selected:
        return no_update, no_update, no_update
    service = _service()
    rows = grid_rows(_filtered(service, severities, types, entity_types, hours, status))
    signature = _signature(rows)
    total = len(service.snapshot().alerts)
    summary = _summary(len(rows), total, hours)
    if by_interval and signature == previous:
        return no_update, summary, no_update
    return rows, summary, signature


@dash.callback(
    Output("alerts-bulk-fp", "disabled"),
    Output("alerts-bulk-fp", "children"),
    Output("alerts-pause-note", "children"),
    Input("alerts-grid", "selectedRows"),
    prevent_initial_call=True,
)
def show_selection(selected: list[dict[str, Any]] | None):
    """Enable the bulk button and tell the user that live updates pause during a selection."""
    count = len(selected or [])
    if not count:
        return True, "Mark as false positive", ""
    return (
        False,
        f"Mark {count} as false positive",
        f"{count} selected · the table does not update while rows are selected",
    )


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("alerts-feedback", "children"),
    Output("alerts-grid", "deselectAll"),
    Input("alerts-bulk-fp", "n_clicks"),
    State("alerts-grid", "selectedRows"),
    State("alerts-analyst", "value"),
    running=[(Output("alerts-bulk-fp", "disabled"), True, False)],
    prevent_initial_call=True,
)
def bulk_false_positive(
    _clicks: int | None, selected: list[dict[str, Any]] | None, analyst: str | None
):
    """Send a false-positive verdict for every selected alert that is not one already."""
    wanted = [r["event_id"] for r in selected or [] if r.get("status") != "False positive"]
    skipped = len(selected or []) - len(wanted)
    if not wanted:
        return feedback_notice({"requested": 0}, "false positive"), True
    result = _service().submit_feedback(
        wanted,
        "false_positive",
        analyst_id=(analyst or DEFAULT_ANALYST).strip() or DEFAULT_ANALYST,
        notes="Marked as a false positive from the dashboard (bulk action).",
    )
    return feedback_notice(result, "false positive", skipped), True


# --------------------------------------------------------------------------- #
# Detail modal
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("alert-modal", "is_open"),
    Output("alert-modal-body", "children"),
    Output("alert-modal-title", "children"),
    Output("alert-modal-event", "data"),
    Input("alerts-grid", "cellClicked"),
    Input("alert-modal-close", "n_clicks"),
    Input("alert-modal-fp", "n_clicks"),
    Input("alert-modal-tp", "n_clicks"),
    State("alert-modal-event", "data"),
    prevent_initial_call=True,
)
def toggle_modal(
    click: dict[str, Any] | None,
    _close: int | None,
    _fp: int | None,
    _tp: int | None,
    current: str | None,
):
    """Open the details of a clicked row; close on Close or after a verdict."""
    trigger = dash.ctx.triggered_id
    if trigger != "alerts-grid":
        return False, no_update, no_update, no_update
    column = str((click or {}).get("colId") or "")
    event_id = (click or {}).get("rowId")
    if not event_id or column.startswith("ag-Grid"):  # the checkbox column only selects
        return no_update, no_update, no_update, no_update
    alert = _service().get_alert(str(event_id))
    if alert is None:
        alert = next(
            (a for a in _service().snapshot().alerts if a.get("event_id") == event_id), None
        )
    title = f"Alert · {alert.get('entity_id')}" if alert else "Alert details"
    return True, alert_detail(alert), title, str(event_id)


@dash.callback(
    Output("alerts-feedback", "children", allow_duplicate=True),
    Input("alert-modal-fp", "n_clicks"),
    Input("alert-modal-tp", "n_clicks"),
    State("alert-modal-event", "data"),
    State("alerts-analyst", "value"),
    prevent_initial_call=True,
)
def modal_feedback(_fp: int | None, _tp: int | None, event_id: str | None, analyst: str | None):
    """Send a verdict for the alert shown in the modal."""
    if not event_id:
        return no_update
    verdict = "false_positive" if dash.ctx.triggered_id == "alert-modal-fp" else "true_positive"
    result = _service().submit_feedback(
        [event_id],
        verdict,
        analyst_id=(analyst or DEFAULT_ANALYST).strip() or DEFAULT_ANALYST,
        notes="Verdict given from the alert detail window.",
    )
    return feedback_notice(result, verdict.replace("_", " "))


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@dash.callback(
    Output("alerts-download", "data"),
    Input("alerts-export", "n_clicks"),
    *FILTER_STATES,
    prevent_initial_call=True,
)
def export_csv(
    _clicks: int | None,
    severities: list[str] | None,
    types: list[str] | None,
    entity_types: list[str] | None,
    hours: float | None,
    status: str | None,
):
    """Download every field of the alerts that match the current filters."""
    alerts = _filtered(_service(), severities, types, entity_types, hours, status)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return dcc.send_string(alerts_to_csv(alerts), f"alerts-{stamp}.csv")
