"""Tests of the Day 20 pages: vehicles, inventory and alerts (widgets, data service)."""

from __future__ import annotations

import csv
import io
from datetime import timedelta
from typing import Any

from src.dashboard.components import alert_widgets as aw
from src.dashboard.components import inventory_widgets as iw
from src.dashboard.components import vehicle_widgets as vw
from src.dashboard.data_service import DashboardDataService
from tests.unit.test_dashboard_data_service import NOW, FakeSource, alert


def iso(minutes_ago: float = 0, seconds_ago: float = 0) -> str:
    return (NOW - timedelta(minutes=minutes_ago, seconds=seconds_ago)).isoformat()


def vehicle(vid: str = "v1", distance: float = 10.0, adherence: float = 0.99) -> dict[str, Any]:
    return {
        "entity_id": vid,
        "source_type": "vehicle",
        "value": 60.0,
        "timestamp": iso(),
        "details": {
            "latitude": 41.0,
            "longitude": -87.0,
            "route_id": "R1",
            "driver_id": "d1",
            "distance_from_route_m": distance,
            "route_adherence_score": adherence,
        },
    }


# ------------------------------ vehicles ------------------------------ #
def test_vehicle_status_rules() -> None:
    assert vw.vehicle_status(vehicle(), []) == vw.ON_ROUTE
    assert vw.vehicle_status(vehicle(distance=800), []) == vw.DEVIATION
    assert vw.vehicle_status(vehicle(), [alert(1)]) == vw.DELAYED
    # deviation wins over delayed
    assert vw.vehicle_status(vehicle(distance=800), [alert(1)]) == vw.DEVIATION


def test_build_fleet_orders_by_urgency_and_ignores_dismissed_alerts() -> None:
    fleet = vw.build_fleet(
        [vehicle("a"), vehicle("b", distance=900), vehicle("c")],
        [
            {**alert(1, entity="c"), "entity_type": "vehicle"},
            {**alert(2, entity="a", fp=True), "entity_type": "vehicle"},
        ],
        NOW,
    )
    assert [v["vehicle_id"] for v in fleet] == ["b", "c", "a"]
    assert [v["status"] for v in fleet] == ["deviation", "delayed", "on-route"]
    assert fleet[2]["active_count"] == 0 and fleet[1]["active_count"] == 1


def test_old_alert_does_not_make_a_vehicle_delayed() -> None:
    old = {**alert(1, entity="a", minutes_ago=60), "entity_type": "vehicle"}
    assert vw.build_fleet([vehicle("a")], [old], NOW)[0]["status"] == vw.ON_ROUTE


def test_deviation_markers_use_nearest_gps_fix() -> None:
    trails = {
        "v1": [
            {"t": iso(seconds_ago=60), "lat": 1.0, "lon": 2.0},
            {"t": iso(), "lat": 3.0, "lon": 4.0},
        ]
    }
    marks = vw.deviation_markers(
        [
            {**alert(1, entity="v1", minutes_ago=0), "entity_type": "vehicle"},
            {**alert(2, entity="v1", minutes_ago=30), "entity_type": "vehicle"},  # no fix nearby
            {**alert(3, entity="v1", minutes_ago=0, fp=True), "entity_type": "vehicle"},
        ],
        trails,
    )
    assert [(m["latitude"], m["longitude"]) for m in marks] == [(3.0, 4.0)]


def test_fleet_map_uses_open_street_map_and_names_every_status() -> None:
    fleet = vw.build_fleet([vehicle("a"), vehicle("b", distance=900)], [], NOW)
    routes = {
        "R1": {"waypoints": [(40.0, -88.0), (42.0, -86.0)], "origin": "A", "destination": "B"}
    }
    fig = vw.fleet_map(
        fleet,
        routes,
        [
            {
                "vehicle_id": "b",
                "latitude": 41,
                "longitude": -87,
                "timestamp": iso(),
                "severity": "CRITICAL",
            }
        ],
        {},
        "all",
    )
    assert fig.layout.map.style == "open-street-map"
    names = {t.name for t in fig.data}
    assert {"Planned route", "Alert position (✕)", "On route", "Deviation"} <= names
    assert fig.layout.uirevision == "all"
    focused = vw.fleet_map(fleet, routes, [], {}, "b")
    assert focused.layout.map.zoom == vw.FOCUS_ZOOM and focused.layout.uirevision == "b"


def test_fleet_map_without_positions_is_a_message() -> None:
    fig = vw.fleet_map([], {}, [], {}, "all")
    assert not fig.data and fig.layout.annotations


def test_info_card_and_table_render_without_a_vehicle() -> None:
    assert vw.vehicle_info_card(None) is not None
    assert vw.fleet_table([]) is not None


# ------------------------------ inventory ------------------------------ #
def sku(sid: str, level: float = 500, warehouse: str = "W1", zone: str = "A") -> dict[str, Any]:
    return {
        "entity_id": sid,
        "source_type": "inventory",
        "value": level,
        "warehouse_id": warehouse,
        "zone": zone,
        "details": {"min_stock": 50, "max_stock": 1000, "reorder_point": 300},
    }


def pts(*changes: float, start: float = 500) -> list[dict[str, Any]]:
    out, level = [], start
    for i, change in enumerate(changes):
        level += change
        out.append({"t": iso(seconds_ago=(len(changes) - i) * 70), "v": level, "d": change})
    return out


def test_top_active_skus_ranks_by_transactions_and_caps_at_ten() -> None:
    entities = [sku(f"S{i:02d}") for i in range(12)]
    series = {f"S{i:02d}": pts(*([10] * i)) for i in range(12)}
    rows = iw.top_active_skus(entities, series, [])
    assert len(rows) == 10 and rows[0]["sku_id"] == "S11"
    assert rows[-1]["transactions"] == 2


def test_top_active_skus_flags_open_alerts_only() -> None:
    rows = iw.top_active_skus(
        [sku("A"), sku("B")],
        {},
        [{**alert(1, entity="A")}, {**alert(2, entity="B", fp=True)}],
    )
    assert {r["sku_id"]: r["has_alert"] for r in rows} == {"A": True, "B": False}


def test_stock_gauges_one_indicator_per_sku_with_reorder_threshold() -> None:
    rows = iw.top_active_skus([sku("A"), sku("B")], {}, [])
    fig = iw.stock_gauges(rows)
    assert [t.type for t in fig.data] == ["indicator", "indicator"]
    assert fig.data[0].gauge.threshold.value == 300


def test_sku_selection_cascades_and_keeps_valid_choices() -> None:
    entities = [
        sku("A", warehouse="W1", zone="A"),
        sku("B", warehouse="W1", zone="B"),
        sku("C", warehouse="W2", zone="A"),
    ]
    first = iw.resolve_sku_selection(entities)
    assert (first["warehouse"], first["zone"], first["sku"]) == ("W1", "A", "A")
    moved = iw.resolve_sku_selection(entities, "W2", "B", "A")  # zone B does not exist in W2
    assert (moved["warehouse"], moved["zone"], moved["sku"]) == ("W2", "A", "C")
    assert iw.resolve_sku_selection([])["sku"] is None


def test_waterfall_buckets_chain_and_mark_anomalies() -> None:
    points = pts(-40, 60, -30)  # all within the last 4 minutes
    buckets = iw.transaction_buckets(points, [])
    assert sum(b["net"] for b in buckets) == -10
    for earlier, later in zip(buckets, buckets[1:], strict=False):
        assert earlier["end"] == later["begin"]
    flagged = iw.transaction_buckets(points, [alert(1, minutes_ago=3, entity="A")])
    assert any(b["kind"] == "anomaly" for b in flagged)
    dismissed = iw.transaction_buckets(points, [alert(1, minutes_ago=3, fp=True)])
    assert all(b["kind"] != "anomaly" for b in dismissed)


def test_waterfall_figure_has_floating_bars_and_empty_state() -> None:
    fig = iw.waterfall_figure(
        iw.transaction_buckets(pts(-40, 60, -30)), sku_id="A", reorder_point=300
    )
    assert any(t.type == "bar" and t.base is not None for t in fig.data)
    assert any(t.name == "Reorder point" for t in fig.data)
    assert not iw.waterfall_figure([]).data


def test_inventory_anomaly_rows_find_the_magnitude() -> None:
    event = {**alert(1, minutes_ago=2, entity="A"), "timestamp": iso(minutes_ago=2)}
    series = {"A": [{"t": iso(minutes_ago=2), "v": 200, "d": -300}]}
    row = iw.inventory_anomaly_rows([event], series)[0]
    assert row["change"] == -300 and row["event"] == "Sudden drop"
    assert iw.inventory_anomaly_rows([event], {})[0]["change"] is None


# ------------------------------- alerts ------------------------------- #
def service_with(alerts: list[dict[str, Any]]) -> tuple[DashboardDataService, FakeSource]:
    source = FakeSource()
    source.alert_rows = alerts
    service = DashboardDataService(source, clock=lambda: NOW)  # type: ignore[call-arg]
    service.refresh()
    return service, source


def test_query_alerts_filters_on_the_server() -> None:
    rows = [
        alert(1, "CRITICAL", 5),
        alert(2, "INFO", 5),
        {**alert(3, "CRITICAL", 5), "entity_type": "vehicle", "anomaly_type": "contextual"},
        alert(4, "CRITICAL", 300),
        alert(5, "CRITICAL", 5, fp=True),
    ]
    service, _ = service_with(rows)
    ids = lambda **kw: [a["event_id"] for a in service.query_alerts(**kw)]  # noqa: E731
    assert ids(severities=["critical"], window_hours=1, status="open") == ["e1", "e3"]
    assert ids(entity_types=["vehicle"]) == ["e3"]
    assert ids(anomaly_types=["contextual"]) == ["e3"]
    assert ids(status="false_positive") == ["e5"]
    assert ids(window_hours=1) != ids()
    assert ids(limit=2) == ["e1", "e2"]


def test_submit_feedback_posts_each_alert_and_reports_failures() -> None:
    service, source = service_with([alert(i) for i in range(4)])
    source.reject = {"e2"}
    result = service.submit_feedback(
        ["e0", "e1", "e2", "e1", ""], "false_positive", analyst_id="me"
    )
    assert result["requested"] == 3 and result["ok"] == 2
    assert result["failed"] == [{"event_id": "e2", "error": "alert no longer stored"}]
    assert sorted(b["anomaly_event_id"] for b in source.posted) == ["e0", "e1"]
    assert all(
        b["analyst_id"] == "me" and b["feedback_type"] == "false_positive" for b in source.posted
    )


def test_grid_rows_and_status_labels() -> None:
    rows = aw.grid_rows([alert(1, resolved=True), alert(2, fp=True), alert(3)])
    assert [r["status"] for r in rows] == ["Resolved", "False positive", "Open"]
    assert rows[0]["event_id"] == "e1" and rows[0]["severity"] == "WARNING"
    assert {c["field"] for c in aw.column_defs()} >= {
        "timestamp",
        "entity_id",
        "entity_type",
        "severity",
        "anomaly_type",
        "value",
        "status",
        "actions",
    }


def test_shap_figure_takes_top_five_by_strength() -> None:
    feats = [
        {"feature": f"f{i}", "shap_value": v}
        for i, v in enumerate([0.1, -0.9, 0.3, 0.05, -0.2, 0.6, 0.4])
    ]
    fig = aw.shap_figure(feats)
    assert len(fig.data[0].y) == 5 and set(fig.data[0].y) == {"f1", "f5", "f6", "f2", "f4"}
    assert fig.data[0].y[-1] == "f1"  # strongest drawn at the top
    assert not aw.shap_figure(None).data


def test_detail_shows_a_missing_alert_without_failing() -> None:
    assert aw.alert_detail(None).id == "alert-detail-missing"
    assert aw.alert_detail({**alert(1), "recommended_action": "Check it"}).id == "alert-detail"


def test_csv_has_every_field_and_is_formula_safe() -> None:
    row = {
        **alert(1),
        "entity_id": "=HYPERLINK(1)",
        "extra_field": "x",
        "individual_scores": {"if": 0.5},
    }
    text = aw.alerts_to_csv([row, alert(2, fp=True)])
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert len(parsed) == 2
    assert parsed[0]["entity_id"] == "'=HYPERLINK(1)"
    assert parsed[0]["expected_min"] == "2.0" and parsed[0]["expected_max"] == "4.0"
    assert parsed[0]["extra_field"] == "x" and '"if"' in parsed[0]["individual_scores"]
    assert parsed[1]["status"] == "false_positive"
    assert set(aw.CSV_COLUMNS) <= set(parsed[0])
    assert aw.alerts_to_csv([]).splitlines() == [",".join(aw.CSV_COLUMNS)]


def test_feedback_notice_reports_partial_failure_and_skips() -> None:
    ok = aw.feedback_notice({"requested": 2, "ok": 2, "failed": []}, "false positive", skipped=1)
    assert "skipped" in str(ok.children)
    partial = aw.feedback_notice(
        {"requested": 2, "ok": 1, "failed": [{"event_id": "abcdefghij", "error": "HTTP 500"}]},
        "false positive",
    )
    assert partial.color == "warning" and "HTTP 500" in str(partial.children)
    assert aw.feedback_notice({"requested": 0}, "x").color == "secondary"
