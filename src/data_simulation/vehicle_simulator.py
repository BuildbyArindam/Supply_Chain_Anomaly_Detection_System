"""GPS vehicle tracking simulation (Day 3).

Every vehicle drives back and forth along a fixed route (a polyline of lat/lon
waypoints) and reports a GPS reading every ``reading_interval_s`` simulated
seconds (10 by default). Normal readings carry a little Gaussian GPS noise, a
plausible cruising speed and a high *route adherence score*. Three anomaly kinds
are injected with exact ground-truth labels:

* **route_deviation** (``point``) - the vehicle is more than
  ``deviation_threshold_m`` (500 m) away from its route corridor for a few
  readings (detour, wrong turn, GPS spoofing).
* **sudden_stop / excessive_speed** (``point``) - the speed collapses to ~0 or
  jumps to 1.5-1.9x cruising speed for a few readings.
* **gradual_delay** (``collective``) - the speed decays smoothly over 12-30
  readings (traffic jam, engine trouble). Early readings look normal; the
  *sequence* is the anomaly, just like the warehouse drift.

Normal readings are guaranteed to stay inside the corridor and inside the normal
speed band (0.6x-1.2x cruise), so ``is_anomaly`` is the exact ground truth.
Every reading is a plain, JSON-serialisable dict::

    {
        "vehicle_id": "veh-001", "timestamp": "2026-11-20T08:00:00Z",
        "latitude": 51.924412, "longitude": 4.477719, "speed_kmh": 63.41, "heading": 187.2,
        "is_anomaly": False, "anomaly_type": None,        # None | point | collective
        "route_adherence_score": 0.9987,                  # 1.0 on the route, ~0.2 at 500 m
        "metadata": {"route_id": "R-RTM-BRU", "driver_id": "drv-001", "trip": 1,
                     "progress_m": 412.7, "distance_from_route_m": 3.1,
                     "cruise_speed_kmh": 65.0, "anomaly_subtype": None,
                     "episode_id": None, "episode_step": None},
    }

Example::

    from src.data_simulation.vehicle_simulator import GPSSimulator

    sim = GPSSimulator(seed=42)                          # 5 vehicles on 5 freight routes
    for reading in sim.simulate_all_vehicles(duration_s=60):
        ...                                              # 5 vehicles x 6 ticks = 30 readings

    sim.inject_route_deviation("veh-001")                # force a detour on the next tick

The simulator is deterministic for a given ``seed`` (each vehicle has its own
random stream) and depends only on the standard library and the project logger.
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

from src.data_simulation._common import (
    default_start,
    isoformat,
    mean_length,
    to_utc,
    validate_duration,
    validate_length_range,
    validate_mix,
    validate_non_negative,
    validate_positive,
    validate_rate,
)
from src.data_simulation.warehouse_simulator import COLLECTIVE, POINT, AnomalyType
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config import VehicleSimConfig

Side = Literal[-1, 1]
SpeedMode = Literal["stop", "excessive"]

EARTH_RADIUS_M: Final = 6_371_008.8

ROUTE_DEVIATION: Final = "route_deviation"
SUDDEN_STOP: Final = "sudden_stop"
EXCESSIVE_SPEED: Final = "excessive_speed"
GRADUAL_DELAY: Final = "gradual_delay"
ANOMALY_SUBTYPES: Final[tuple[str, ...]] = (
    ROUTE_DEVIATION,
    SUDDEN_STOP,
    EXCESSIVE_SPEED,
    GRADUAL_DELAY,
)

# Normal speed is a Gaussian around the cruising speed, truncated to this band
# (fractions of cruise). Stops and excessive speed sit clearly outside it.
_NORMAL_SPEED_BAND: Final = (0.6, 1.2)
_SPEED_NOISE_FRACTION: Final = 0.06
_STOP_MAX_KMH: Final = 2.0
_EXCESSIVE_FACTOR: Final = (1.5, 1.9)
_EXCESSIVE_FLOOR_FACTOR: Final = 1.45
_DELAY_FINAL_FACTOR: Final = (0.35, 0.65)  # speed at the end of a gradual delay (x cruise)

# Route deviations start at 1.25-3x the corridor width and are never allowed to
# fall below _DEVIATION_MARGIN x the threshold, which leaves headroom for the small
# difference between the flat-earth maths used here and true geodesic distance.
_DEVIATION_OFFSET: Final = (1.25, 3.0)
_DEVIATION_MARGIN: Final = 1.15
_HEADING_TURN_DEG: Final = (25.0, 75.0)
_HEADING_NOISE_DEG: Final = 1.5
_NOISE_CLIP_SIGMAS: Final = 4.0
_MAX_RESAMPLES: Final = 25
_MIN_ROUTE_LENGTH_M: Final = 1_000.0


class GPSReading(TypedDict):
    """One labelled GPS reading (the schema Day 4's Kafka models mirror)."""

    vehicle_id: str
    timestamp: str
    latitude: float
    longitude: float
    speed_kmh: float
    heading: float
    is_anomaly: bool
    anomaly_type: AnomalyType | None
    route_adherence_score: float
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Route geometry
# --------------------------------------------------------------------------- #
class _Route:
    """A polyline in a local flat-earth plane (x east, y north, metres).

    The plane is anchored at the first waypoint and scaled at the mean latitude,
    which is accurate to well under 1 % for routes a few hundred kilometres long -
    far better than the 500 m corridor needs. All interpolation, offsets and
    distances are done in this plane so they are mutually consistent.
    """

    def __init__(self, waypoints: Sequence[tuple[float, float]]) -> None:
        self._lat0 = sum(lat for lat, _ in waypoints) / len(waypoints)
        self._lon0 = waypoints[0][1]
        self._kx = EARTH_RADIUS_M * math.cos(math.radians(self._lat0))
        self._points = [self.to_xy(lat, lon) for lat, lon in waypoints]
        self._cumulative = [0.0]
        self._headings: list[float] = []
        for (x0, y0), (x1, y1) in zip(self._points, self._points[1:], strict=False):
            self._cumulative.append(self._cumulative[-1] + math.hypot(x1 - x0, y1 - y0))
            self._headings.append(math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0)

    @property
    def length(self) -> float:
        """Total route length in metres."""
        return self._cumulative[-1]

    @property
    def min_segment_length(self) -> float:
        """Length of the shortest segment in metres."""
        return min(b - a for a, b in zip(self._cumulative, self._cumulative[1:], strict=False))

    def to_xy(self, latitude: float, longitude: float) -> tuple[float, float]:
        """Project lat/lon (degrees) to plane coordinates (metres)."""
        x = math.radians(longitude - self._lon0) * self._kx
        y = math.radians(latitude - self._lat0) * EARTH_RADIUS_M
        return x, y

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of :meth:`to_xy`."""
        latitude = self._lat0 + math.degrees(y / EARTH_RADIUS_M)
        longitude = self._lon0 + math.degrees(x / self._kx)
        return latitude, longitude

    def point_at(self, distance: float) -> tuple[float, float, float]:
        """Return ``(x, y, heading_deg)`` at ``distance`` metres along the route."""
        distance = min(max(distance, 0.0), self.length)
        index = min(bisect.bisect_right(self._cumulative, distance) - 1, len(self._headings) - 1)
        index = max(index, 0)
        start, end = self._cumulative[index], self._cumulative[index + 1]
        fraction = (distance - start) / (end - start)
        (x0, y0), (x1, y1) = self._points[index], self._points[index + 1]
        return x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0), self._headings[index]

    def distance_to(self, x: float, y: float) -> float:
        """Return the shortest distance in metres from ``(x, y)`` to the polyline."""
        best = math.inf
        for (x0, y0), (x1, y1) in zip(self._points, self._points[1:], strict=False):
            dx, dy = x1 - x0, y1 - y0
            fraction = ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)
            fraction = min(max(fraction, 0.0), 1.0)
            best = min(best, math.hypot(x - (x0 + fraction * dx), y - (y0 + fraction * dy)))
        return best


def _fold(travelled: float, length: float) -> tuple[float, int, int]:
    """Map unbounded travelled distance to ``(position, direction, trip)`` on a ping-pong route.

    The vehicle drives origin -> destination (direction ``+1``), then back
    (direction ``-1``), and so on. ``trip`` is the 1-based leg number.
    """
    period = 2.0 * length
    phase = travelled % period
    trip = int(travelled // length) + 1
    if phase < length:
        return phase, 1, trip
    return period - phase, -1, trip


# --------------------------------------------------------------------------- #
# Vehicle definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VehicleTracker:
    """Static description of one tracked vehicle and its planned route.

    Attributes:
        vehicle_id: Unique id, e.g. ``"veh-001"``.
        driver_id: Driver currently assigned, e.g. ``"drv-001"``.
        route_id: Id of the planned route (several vehicles may share one).
        origin: Human-readable start of the route.
        destination: Human-readable end of the route.
        waypoints: ``(latitude, longitude)`` pairs, at least two, in driving order.
            Lists are accepted and normalised to a tuple of float tuples.
        cruise_speed_kmh: Typical speed on the route (20-100 km/h).
    """

    vehicle_id: str
    driver_id: str
    route_id: str
    origin: str
    destination: str
    waypoints: tuple[tuple[float, float], ...]
    cruise_speed_kmh: float = 65.0

    def __post_init__(self) -> None:
        """Validate the definition and normalise ``waypoints``."""
        for name in ("vehicle_id", "driver_id", "route_id", "origin", "destination"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        points = tuple((float(lat), float(lon)) for lat, lon in self.waypoints)
        if len(points) < 2:
            raise ValueError("a route needs at least two waypoints")
        for lat, lon in points:
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(f"waypoint {(lat, lon)} is not a valid latitude/longitude")
        if not 20.0 <= self.cruise_speed_kmh <= 100.0:
            raise ValueError(
                f"cruise_speed_kmh must be within [20, 100], got {self.cruise_speed_kmh}"
            )
        route = _Route(points)
        if route.min_segment_length <= 1.0:
            raise ValueError("consecutive waypoints must be more than 1 m apart")
        if route.length < _MIN_ROUTE_LENGTH_M:
            raise ValueError(f"route must be at least {_MIN_ROUTE_LENGTH_M:.0f} m long")
        object.__setattr__(self, "waypoints", points)


@dataclass(frozen=True)
class _RouteTemplate:
    route_id: str
    origin: str
    destination: str
    waypoints: tuple[tuple[float, float], ...]


#: Approximate freight corridors used by :func:`build_default_vehicles`.
_ROUTE_TEMPLATES: Final[tuple[_RouteTemplate, ...]] = (
    _RouteTemplate(
        "R-RTM-BRU",
        "Rotterdam",
        "Brussels",
        ((51.9244, 4.4777), (51.5719, 4.7683), (51.2194, 4.4025), (50.8503, 4.3517)),
    ),
    _RouteTemplate(
        "R-CHI-IND",
        "Chicago",
        "Indianapolis",
        ((41.8781, -87.6298), (41.5934, -87.3464), (40.4167, -86.8753), (39.7684, -86.1581)),
    ),
    _RouteTemplate(
        "R-LAX-SAN",
        "Los Angeles",
        "San Diego",
        ((34.0522, -118.2437), (33.8366, -117.9143), (33.6846, -117.8265), (32.7157, -117.1611)),
    ),
    _RouteTemplate(
        "R-FRA-MUC",
        "Frankfurt",
        "Munich",
        ((50.1109, 8.6821), (49.7913, 9.9534), (49.4521, 11.0767), (48.1351, 11.5820)),
    ),
    _RouteTemplate(
        "R-ATL-SAV",
        "Atlanta",
        "Savannah",
        ((33.7490, -84.3880), (32.8407, -83.6324), (32.0809, -81.0912)),
    ),
)


def default_routes() -> dict[str, dict[str, Any]]:
    """Return the built-in freight routes by id (the planned routes of the default fleet).

    The dashboard draws these as the planned route polylines. Each value is
    ``{"origin": str, "destination": str, "waypoints": [(lat, lon), ...]}``.
    """
    return {
        t.route_id: {
            "origin": t.origin,
            "destination": t.destination,
            "waypoints": [tuple(p) for p in t.waypoints],
        }
        for t in _ROUTE_TEMPLATES
    }


def build_default_vehicles(n_vehicles: int = 5, *, seed: int | None = None) -> list[VehicleTracker]:
    """Build a fleet on the built-in freight routes.

    Vehicle ``k`` drives route template ``k mod 5``, so fleets larger than five
    put several vehicles on the same route. Ids look like ``veh-001`` / ``drv-001``.

    Args:
        n_vehicles: Fleet size.
        seed: Seeds the cruising-speed choices.

    Raises:
        ValueError: ``n_vehicles`` is not positive.
    """
    if n_vehicles < 1:
        raise ValueError(f"n_vehicles must be >= 1, got {n_vehicles}")
    rng = random.Random(seed)
    fleet: list[VehicleTracker] = []
    for index in range(n_vehicles):
        template = _ROUTE_TEMPLATES[index % len(_ROUTE_TEMPLATES)]
        fleet.append(
            VehicleTracker(
                vehicle_id=f"veh-{index + 1:03d}",
                driver_id=f"drv-{index + 1:03d}",
                route_id=template.route_id,
                origin=template.origin,
                destination=template.destination,
                waypoints=template.waypoints,
                cruise_speed_kmh=round(rng.uniform(55.0, 80.0), 1),
            )
        )
    return fleet


# --------------------------------------------------------------------------- #
# Internal state
# --------------------------------------------------------------------------- #
@dataclass
class _Episode:
    """An in-progress anomaly on one vehicle."""

    subtype: str
    kind: AnomalyType
    episode_id: str
    total: int
    magnitude: float  # deviation: offset (m); excessive: speed factor; delay: final speed factor
    side: int = 1  # deviation only: +1 left of travel, -1 right
    heading_turn: float = 0.0  # deviation only: degrees the heading is rotated
    index: int = 0


@dataclass
class _VehicleState:
    """Mutable per-vehicle simulation state."""

    tracker: VehicleTracker
    route: _Route
    rng: random.Random
    travelled: float  # unbounded distance driven along the ping-pong route, metres
    episode: _Episode | None = None
    episodes_started: int = 0


def _clamp_noise(rng: random.Random, sigma: float) -> float:
    """Draw zero-mean Gaussian noise clipped to +/- 4 sigma."""
    if sigma == 0.0:
        return 0.0
    bound = _NOISE_CLIP_SIGMAS * sigma
    return min(max(rng.gauss(0.0, sigma), -bound), bound)


def _normal_speed(rng: random.Random, cruise: float) -> float:
    """Draw a normal speed: Gaussian around ``cruise`` truncated to the normal band."""
    low, high = _NORMAL_SPEED_BAND[0] * cruise, _NORMAL_SPEED_BAND[1] * cruise
    for _ in range(_MAX_RESAMPLES):
        value = rng.gauss(cruise, _SPEED_NOISE_FRACTION * cruise)
        if low <= value <= high:
            return value
    return cruise


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class GPSSimulator:
    """Generate labelled GPS readings for a fleet of vehicles.

    All vehicles tick together on a simulated clock (``reading_interval_s``,
    default 10 s). Anomalies are *episodes*: once one starts on a vehicle it runs
    for its full length and no other episode starts on that vehicle meanwhile.
    Episodes start randomly at the configured rate, or on demand via the
    ``inject_*`` methods (used by the demo and tests).

    ``anomaly_rate`` is the approximate fraction of **all** readings labelled
    anomalous, split between route deviations, speed anomalies and gradual delays
    by ``anomaly_mix``. Because a vehicle cannot start a new episode while one is
    active, the realised fraction is ``rate / (1 + rate)`` - a couple of percent
    below target at the default 2 %.
    """

    def __init__(
        self,
        vehicles: Sequence[VehicleTracker] | None = None,
        *,
        deviation_threshold_m: float = 500.0,
        anomaly_rate: float = 0.02,
        anomaly_mix: Sequence[float] = (0.4, 0.3, 0.3),
        deviation_length_range: tuple[int, int] = (3, 12),
        speed_length_range: tuple[int, int] = (2, 6),
        delay_length_range: tuple[int, int] = (12, 30),
        gps_noise_m: float = 4.0,
        reading_interval_s: float = 10.0,
        start_time: datetime | None = None,
        seed: int | None = None,
    ) -> None:
        """Create a simulator.

        Args:
            vehicles: Fleet to simulate. Defaults to :func:`build_default_vehicles`.
            deviation_threshold_m: Distance from the route that counts as a deviation.
            anomaly_rate: Target fraction of readings labelled anomalous.
            anomaly_mix: Shares of ``(route_deviation, speed anomaly, gradual_delay)``
                among anomalies; must sum to 1.
            deviation_length_range: Inclusive readings per route deviation.
            speed_length_range: Inclusive readings per sudden stop / excessive speed.
            delay_length_range: Inclusive readings per gradual delay.
            gps_noise_m: Standard deviation of GPS position noise (clipped at 4 sigma).
            reading_interval_s: Simulated seconds between ticks.
            start_time: Simulated clock start (naive values are UTC). Defaults to now.
            seed: Makes the stream reproducible. ``None`` is non-deterministic.

        Raises:
            ValueError: Invalid rates, ranges, interval, noise, vehicles or start time.
        """
        self._threshold = validate_positive("deviation_threshold_m", deviation_threshold_m)
        self._rate = validate_rate("anomaly_rate", anomaly_rate)
        self._mix = validate_mix("anomaly_mix", anomaly_mix, 3)
        self._deviation_range = validate_length_range(
            "deviation_length_range", deviation_length_range
        )
        self._speed_range = validate_length_range("speed_length_range", speed_length_range)
        self._delay_range = validate_length_range("delay_length_range", delay_length_range)
        self._noise = validate_non_negative("gps_noise_m", gps_noise_m)
        self._interval = validate_positive("reading_interval_s", reading_interval_s)
        self._start = to_utc(start_time) if start_time else default_start()
        self._tick_index = 0
        self._seed = seed
        self._log = get_logger(__name__)

        fleet = list(vehicles) if vehicles is not None else build_default_vehicles(seed=seed)
        if not fleet:
            raise ValueError("at least one vehicle is required")
        self._states: dict[str, _VehicleState] = {}
        for tracker in fleet:
            if tracker.vehicle_id in self._states:
                raise ValueError(f"duplicate vehicle_id {tracker.vehicle_id!r}")
            rng = self._make_rng(tracker.vehicle_id)
            route = _Route(tracker.waypoints)
            self._states[tracker.vehicle_id] = _VehicleState(
                tracker=tracker,
                route=route,
                rng=rng,
                travelled=rng.uniform(0.0, 2.0 * route.length),
            )

        self._p_deviation = self._rate * self._mix[0] / mean_length(self._deviation_range)
        self._p_speed = self._rate * self._mix[1] / mean_length(self._speed_range)
        self._p_delay = self._rate * self._mix[2] / mean_length(self._delay_range)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg: VehicleSimConfig,
        *,
        seed: int | None = None,
        start_time: datetime | None = None,
        vehicles: Sequence[VehicleTracker] | None = None,
    ) -> GPSSimulator:
        """Build a simulator from ``config.simulation.vehicle``.

        Args:
            cfg: The ``simulation.vehicle`` section of the project config.
            seed: Usually ``config.simulation.seed``.
            start_time: Simulated clock start (defaults to now, UTC).
            vehicles: Custom fleet; defaults to ``cfg.n_vehicles`` built-in vehicles.
        """
        fleet = (
            list(vehicles)
            if vehicles is not None
            else build_default_vehicles(cfg.n_vehicles, seed=seed)
        )
        return cls(
            fleet,
            deviation_threshold_m=cfg.route_deviation_m,
            anomaly_rate=cfg.anomaly_rate,
            reading_interval_s=cfg.reading_interval_s,
            start_time=start_time,
            seed=seed,
        )

    def _make_rng(self, vehicle_id: str) -> random.Random:
        """Return the vehicle's private random stream (stable across fleet changes)."""
        if self._seed is None:
            return random.Random()
        return random.Random(f"{self._seed}:{vehicle_id}")

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def vehicles(self) -> tuple[VehicleTracker, ...]:
        """All simulated vehicles, in tick order."""
        return tuple(state.tracker for state in self._states.values())

    @property
    def reading_interval_s(self) -> float:
        """Simulated seconds between ticks."""
        return self._interval

    @property
    def deviation_threshold_m(self) -> float:
        """Distance from the route (metres) beyond which a reading is a deviation."""
        return self._threshold

    @property
    def current_time(self) -> datetime:
        """Simulated timestamp of the *next* tick."""
        return self._time_at(self._tick_index)

    def get_vehicle(self, vehicle_id: str) -> VehicleTracker:
        """Return a vehicle by id.

        Raises:
            KeyError: Unknown vehicle id.
        """
        return self._state(vehicle_id).tracker

    def distance_from_route_m(self, vehicle_id: str, latitude: float, longitude: float) -> float:
        """Return how far (metres) a position is from the vehicle's planned route.

        This is the quantity the ground truth for route deviations is defined on,
        so downstream code and tests can re-derive labels from raw coordinates.

        Raises:
            KeyError: Unknown vehicle id.
        """
        route = self._state(vehicle_id).route
        return route.distance_to(*route.to_xy(latitude, longitude))

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def tick(self) -> list[GPSReading]:
        """Advance the clock one interval and return one reading per vehicle."""
        moment = self._time_at(self._tick_index)
        self._tick_index += 1
        stamp = isoformat(moment)
        return [self._build_reading(state, stamp) for state in self._states.values()]

    def simulate_all_vehicles(
        self, n_ticks: int | None = None, duration_s: float | None = None
    ) -> Iterator[GPSReading]:
        """Yield readings for every vehicle, tick by tick, in timestamp order.

        With neither argument the stream is endless. Each tick yields one reading
        per vehicle, so ``n_ticks`` ticks produce ``n_ticks * len(vehicles)`` readings.

        Args:
            n_ticks: Number of ticks to simulate.
            duration_s: Simulated seconds to cover (rounded up to whole ticks).

        Raises:
            ValueError: Both arguments given, or a negative value.
        """
        if n_ticks is not None and duration_s is not None:
            raise ValueError("pass either n_ticks or duration_s, not both")
        if duration_s is not None:
            if duration_s < 0:
                raise ValueError(f"duration_s must be >= 0, got {duration_s}")
            n_ticks = math.ceil(duration_s / self._interval)
        if n_ticks is not None and n_ticks < 0:
            raise ValueError(f"n_ticks must be >= 0, got {n_ticks}")
        produced = 0
        while n_ticks is None or produced < n_ticks:
            yield from self.tick()
            produced += 1

    # ------------------------------------------------------------------ #
    # Anomaly injection (manual; the random path uses the same builders)
    # ------------------------------------------------------------------ #
    def inject_route_deviation(
        self, vehicle_id: str, duration: int | None = None, side: Side | None = None
    ) -> str:
        """Start a route deviation on the vehicle's next reading.

        The vehicle appears more than ``deviation_threshold_m`` away from its route
        for ``duration`` readings, then rejoins it. Replaces any running episode.

        Args:
            vehicle_id: Target vehicle.
            duration: Readings to last; random in ``deviation_length_range`` if omitted.
            side: ``1`` to leave the route to the left of travel, ``-1`` to the right;
                random if omitted.

        Returns:
            The new episode id (also reported in each reading's metadata).

        Raises:
            KeyError: Unknown vehicle id.
            ValueError: Invalid duration or side.
        """
        state = self._state(vehicle_id)
        if side is not None and side not in (-1, 1):
            raise ValueError(f"side must be -1 or 1, got {side}")
        return self._start_deviation(
            state, validate_duration("duration", duration), side, manual=True
        ).episode_id

    def inject_speed_anomaly(
        self, vehicle_id: str, duration: int | None = None, mode: SpeedMode | None = None
    ) -> str:
        """Start a sudden stop or excessive-speed episode on the next reading.

        Args:
            vehicle_id: Target vehicle.
            duration: Readings to last; random in ``speed_length_range`` if omitted.
            mode: ``"stop"`` (speed drops to ~0) or ``"excessive"`` (1.5-1.9x cruise);
                random if omitted.

        Returns:
            The new episode id.

        Raises:
            KeyError: Unknown vehicle id.
            ValueError: Invalid duration or mode.
        """
        state = self._state(vehicle_id)
        if mode is not None and mode not in ("stop", "excessive"):
            raise ValueError(f"mode must be 'stop' or 'excessive', got {mode!r}")
        return self._start_speed(
            state, validate_duration("duration", duration), mode, manual=True
        ).episode_id

    def inject_gradual_delay(self, vehicle_id: str, length: int | None = None) -> str:
        """Start a gradual slowdown (traffic / breakdown) on the next reading.

        The vehicle's speed decays linearly to 35-65 % of cruise over ``length``
        readings, then recovers. Replaces any running episode.

        Args:
            vehicle_id: Target vehicle.
            length: Readings to last; random in ``delay_length_range`` if omitted.

        Returns:
            The new episode id.

        Raises:
            KeyError: Unknown vehicle id.
            ValueError: Invalid length.
        """
        state = self._state(vehicle_id)
        return self._start_delay(state, validate_duration("length", length), manual=True).episode_id

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _state(self, vehicle_id: str) -> _VehicleState:
        try:
            return self._states[vehicle_id]
        except KeyError:
            raise KeyError(f"unknown vehicle_id {vehicle_id!r}") from None

    def _time_at(self, tick_index: int) -> datetime:
        return self._start + timedelta(seconds=tick_index * self._interval)

    def _new_episode(
        self,
        state: _VehicleState,
        subtype: str,
        kind: AnomalyType,
        total: int,
        magnitude: float,
        *,
        side: int = 1,
        heading_turn: float = 0.0,
        manual: bool,
    ) -> _Episode:
        state.episodes_started += 1
        episode = _Episode(
            subtype=subtype,
            kind=kind,
            episode_id=f"{state.tracker.vehicle_id}#{state.episodes_started}",
            total=total,
            magnitude=magnitude,
            side=side,
            heading_turn=heading_turn,
        )
        state.episode = episode
        self._log.debug(
            "anomaly_episode_started",
            vehicle_id=state.tracker.vehicle_id,
            anomaly_type=kind,
            anomaly_subtype=subtype,
            episode_id=episode.episode_id,
            length=total,
            manual=manual,
        )
        return episode

    def _start_deviation(
        self, state: _VehicleState, duration: int | None, side: int | None, *, manual: bool = False
    ) -> _Episode:
        rng = state.rng
        total = duration if duration is not None else rng.randint(*self._deviation_range)
        chosen = side if side is not None else rng.choice((-1, 1))
        offset = self._threshold * rng.uniform(*_DEVIATION_OFFSET)
        turn = -chosen * rng.uniform(*_HEADING_TURN_DEG)
        return self._new_episode(
            state,
            ROUTE_DEVIATION,
            POINT,
            total,
            offset,
            side=chosen,
            heading_turn=turn,
            manual=manual,
        )

    def _start_speed(
        self,
        state: _VehicleState,
        duration: int | None,
        mode: SpeedMode | None,
        *,
        manual: bool = False,
    ) -> _Episode:
        rng = state.rng
        total = duration if duration is not None else rng.randint(*self._speed_range)
        chosen = mode if mode is not None else rng.choice(("stop", "excessive"))
        if chosen == "stop":
            return self._new_episode(state, SUDDEN_STOP, POINT, total, 0.0, manual=manual)
        factor = rng.uniform(*_EXCESSIVE_FACTOR)
        return self._new_episode(state, EXCESSIVE_SPEED, POINT, total, factor, manual=manual)

    def _start_delay(
        self, state: _VehicleState, length: int | None, *, manual: bool = False
    ) -> _Episode:
        rng = state.rng
        total = length if length is not None else rng.randint(*self._delay_range)
        final_factor = rng.uniform(*_DELAY_FINAL_FACTOR)
        return self._new_episode(
            state, GRADUAL_DELAY, COLLECTIVE, total, final_factor, manual=manual
        )

    def _maybe_start_random_episode(self, state: _VehicleState) -> _Episode | None:
        """Possibly start a random episode on a free vehicle (one draw picks the type)."""
        draw = state.rng.random()
        if draw < self._p_deviation:
            return self._start_deviation(state, None, None)
        draw -= self._p_deviation
        if draw < self._p_speed:
            return self._start_speed(state, None, None)
        draw -= self._p_speed
        if draw < self._p_delay:
            return self._start_delay(state, None)
        return None

    def _speed_for(self, state: _VehicleState, episode: _Episode | None) -> float:
        rng, cruise = state.rng, state.tracker.cruise_speed_kmh
        base = _normal_speed(rng, cruise)
        if episode is None or episode.subtype == ROUTE_DEVIATION:
            return base
        if episode.subtype == SUDDEN_STOP:
            return min(abs(rng.gauss(0.0, 0.8)), _STOP_MAX_KMH)
        if episode.subtype == EXCESSIVE_SPEED:
            factor = max(episode.magnitude + rng.gauss(0.0, 0.03), _EXCESSIVE_FLOOR_FACTOR)
            return cruise * factor
        progress = (episode.index + 1) / episode.total
        return base * (1.0 - (1.0 - episode.magnitude) * progress)

    def _build_reading(self, state: _VehicleState, stamp: str) -> GPSReading:
        tracker, route, rng = state.tracker, state.route, state.rng
        episode = state.episode
        if episode is None:
            episode = self._maybe_start_random_episode(state)

        speed = self._speed_for(state, episode)
        state.travelled += speed / 3.6 * self._interval
        position, direction, trip = _fold(state.travelled, route.length)
        x, y, segment_heading = route.point_at(position)
        heading = segment_heading if direction > 0 else segment_heading + 180.0
        x += _clamp_noise(rng, self._noise)
        y += _clamp_noise(rng, self._noise)

        if episode is not None and episode.subtype == ROUTE_DEVIATION:
            heading += episode.heading_turn
            left_normal = segment_heading - 90.0 * direction  # compass heading of "left of travel"
            x, y = self._deviate(route, episode, x, y, left_normal=left_normal)
        distance = route.distance_to(x, y)
        adherence = 1.0 / (1.0 + (distance / (0.5 * self._threshold)) ** 2)

        if episode is None:
            anomaly_type: AnomalyType | None = None
            subtype: str | None = None
            episode_id: str | None = None
            step: int | None = None
        else:
            anomaly_type, subtype, episode_id = episode.kind, episode.subtype, episode.episode_id
            step = episode.index
            episode.index += 1
            if episode.index >= episode.total:
                state.episode = None

        latitude, longitude = route.to_latlon(x, y)
        heading = round((heading + rng.gauss(0.0, _HEADING_NOISE_DEG)) % 360.0, 1) % 360.0
        return {
            "vehicle_id": tracker.vehicle_id,
            "timestamp": stamp,
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "speed_kmh": round(speed, 2),
            "heading": heading,
            "is_anomaly": anomaly_type is not None,
            "anomaly_type": anomaly_type,
            "route_adherence_score": round(adherence, 4),
            "metadata": {
                "route_id": tracker.route_id,
                "driver_id": tracker.driver_id,
                "trip": trip,
                "progress_m": round(position, 1),
                "distance_from_route_m": round(distance, 1),
                "cruise_speed_kmh": tracker.cruise_speed_kmh,
                "anomaly_subtype": subtype,
                "episode_id": episode_id,
                "episode_step": step,
            },
        }

    def _deviate(
        self, route: _Route, episode: _Episode, x: float, y: float, *, left_normal: float
    ) -> tuple[float, float]:
        """Push ``(x, y)`` sideways until it is safely outside the route corridor.

        ``left_normal`` is the compass heading (degrees) pointing to the left of the
        direction of travel; the episode's ``side`` picks left (+1) or right (-1). The
        offset starts at the episode's magnitude and grows by 50 % at a time while the
        true distance to the route is still short of the required margin (this can
        happen near sharp corners), so the ground-truth label is exact.
        """
        required = _DEVIATION_MARGIN * self._threshold
        theta = math.radians(left_normal)
        unit_x, unit_y = math.sin(theta) * episode.side, math.cos(theta) * episode.side
        offset = episode.magnitude
        while True:
            candidate = (x + unit_x * offset, y + unit_y * offset)
            if route.distance_to(*candidate) >= required:
                return candidate
            offset *= 1.5
