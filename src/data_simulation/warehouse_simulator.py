"""Warehouse temperature / humidity sensor simulation (Day 2).

Generates a ground-truth-labelled stream of readings, one per sensor per
simulated second, with three kinds of injected anomalies:

* **point** - a sudden spike *outside* the sensor's acceptable range that lasts
  1-5 readings (door left open, compressor trip, sensor glitch).
* **contextual** - a reading that is *inside* the acceptable range but wrong for
  the time of day: during off-hours (00:00-05:00 by default) the warehouse should
  be quiet, yet the sensor shows a business-hours activity signature (level shift
  plus extra noise). A plain range check cannot catch these.
* **collective** - a gradual drift over 10-30 readings that simulates equipment
  degradation. Individual readings may look unremarkable; the *sequence* is the
  anomaly.

Normal readings follow a Gaussian around ``normal_mean`` with a mild diurnal
pattern (slightly warmer/more humid during business hours, 06:00-22:00) and are
always kept inside ``acceptable_range``, so ``is_anomaly`` is the exact ground
truth. Every reading is a plain, JSON-serialisable dict::

    {
        "sensor_id": "wh1-A-temp-01", "warehouse_id": "wh1",
        "timestamp": "2026-11-20T00:00:00Z", "reading_type": "temperature",
        "value": 3.02, "unit": "°C",
        "is_anomaly": False, "anomaly_type": None,   # None | point | contextual | collective
        "metadata": {"zone": "A", "firmware_version": "2.3.1", "battery_level": 87.41,
                     "last_calibration_date": "2026-08-14", "episode_id": None, ...},
    }

Example::

    from src.data_simulation.warehouse_simulator import WarehouseSimulator

    sim = WarehouseSimulator(seed=42)                     # 2 warehouses x 3 zones x 2 sensors
    for reading in sim.simulate_all_sensors(duration_s=60):
        ...                                               # 12 sensors x 60 ticks = 720 readings

    sim.inject_point_anomaly("wh1-A-temp-01")             # force an anomaly on the next tick

The simulator is deterministic for a given ``seed`` (each sensor has its own
random stream, so adding a sensor does not change the others) and has no
dependency beyond the standard library and the project logger.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config import WarehouseSimConfig

SensorType = Literal["temperature", "humidity"]
AnomalyType = Literal["point", "contextual", "collective"]
Direction = Literal[-1, 1]

TEMPERATURE: Final[SensorType] = "temperature"
HUMIDITY: Final[SensorType] = "humidity"
SENSOR_TYPES: Final[tuple[SensorType, ...]] = (TEMPERATURE, HUMIDITY)
UNITS: Final[dict[str, str]] = {TEMPERATURE: "°C", HUMIDITY: "%RH"}
#: Physically possible values per sensor type; injected anomalies are clamped to these.
PHYSICAL_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    TEMPERATURE: (-50.0, 80.0),
    HUMIDITY: (0.0, 100.0),
}

POINT: Final[AnomalyType] = "point"
CONTEXTUAL: Final[AnomalyType] = "contextual"
COLLECTIVE: Final[AnomalyType] = "collective"
ANOMALY_TYPES: Final[tuple[AnomalyType, ...]] = (POINT, CONTEXTUAL, COLLECTIVE)

_TYPE_ABBREVIATION: Final[dict[str, str]] = {TEMPERATURE: "temp", HUMIDITY: "hum"}
_FIRMWARE_VERSIONS: Final[tuple[str, ...]] = ("2.3.1", "2.4.0", "3.0.2")

# Diurnal model: business hours raise the mean by up to _ACTIVITY_SHIFT_SIGMA sigmas.
_BUSINESS_START_HOUR: Final = 6.0
_BUSINESS_END_HOUR: Final = 22.0
_ACTIVITY_SHIFT_SIGMA: Final = 0.4

# Anomaly signatures, in units of the sensor's normal_std unless stated otherwise.
_POINT_OVERSHOOT_SPAN: Final = (0.15, 0.60)  # fraction of the acceptable range width
_POINT_JITTER_SPAN: Final = 0.05  # fraction of the acceptable range width
_CONTEXTUAL_SHIFT_SIGMA: Final = (1.5, 3.0)
_CONTEXTUAL_NOISE_FACTOR: Final = 1.5
_COLLECTIVE_DRIFT_SIGMA: Final = (2.5, 4.5)
_COLLECTIVE_UP_PROBABILITY_TEMPERATURE: Final = 0.8  # cooling/HVAC failures mostly warm up

_BATTERY_DRAIN_PER_READING: Final = 1.0e-4  # percentage points
_MAX_RESAMPLES: Final = 25


class WarehouseReading(TypedDict):
    """One labelled sensor reading (the schema Day 4's Kafka models mirror)."""

    sensor_id: str
    warehouse_id: str
    timestamp: str
    reading_type: SensorType
    value: float
    unit: str
    is_anomaly: bool
    anomaly_type: AnomalyType | None
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Sensor definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WarehouseSensor:
    """Static description of one warehouse sensor.

    Attributes:
        sensor_id: Unique id, e.g. ``"wh1-A-temp-01"``.
        warehouse_id: Owning warehouse, e.g. ``"wh1"``.
        location: Zone inside the warehouse (``"A"``, ``"B"``, ``"C"``).
        sensor_type: ``"temperature"`` or ``"humidity"``.
        normal_mean: Centre of the normal Gaussian reading distribution.
        normal_std: Standard deviation of normal readings (must be positive).
        acceptable_range: ``(min, max)`` the value must stay within when healthy.
        firmware_version: Reported in the reading metadata.
        battery_level: Initial battery charge in percent (drains slowly).
        last_calibration_date: Date of the last calibration, if known.
        storage_type: Free-form storage class, e.g. ``"cold_storage"``.
    """

    sensor_id: str
    warehouse_id: str
    location: str
    sensor_type: SensorType
    normal_mean: float
    normal_std: float
    acceptable_range: tuple[float, float]
    firmware_version: str = "1.0.0"
    battery_level: float = 100.0
    last_calibration_date: date | None = None
    storage_type: str = "ambient"

    def __post_init__(self) -> None:
        """Validate the definition and normalise ``acceptable_range`` to a float tuple."""
        for name in ("sensor_id", "warehouse_id", "location"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.sensor_type not in SENSOR_TYPES:
            raise ValueError(f"sensor_type must be one of {SENSOR_TYPES}, got {self.sensor_type!r}")
        if self.normal_std <= 0:
            raise ValueError(f"normal_std must be positive, got {self.normal_std}")
        low, high = (float(self.acceptable_range[0]), float(self.acceptable_range[1]))
        if low >= high:
            raise ValueError(f"acceptable_range must satisfy min < max, got {(low, high)}")
        if not low <= self.normal_mean <= high:
            raise ValueError(
                f"normal_mean {self.normal_mean} must lie inside acceptable_range {(low, high)}"
            )
        if not 0.0 <= self.battery_level <= 100.0:
            raise ValueError(f"battery_level must be within [0, 100], got {self.battery_level}")
        phys_min, phys_max = PHYSICAL_BOUNDS[self.sensor_type]
        if low < phys_min or high > phys_max:
            raise ValueError(
                f"acceptable_range {(low, high)} exceeds physical bounds {(phys_min, phys_max)}"
            )
        object.__setattr__(self, "acceptable_range", (low, high))
        if not self.can_spike_up and not self.can_spike_down:
            raise ValueError("acceptable_range leaves no room for out-of-range anomalies")

    @property
    def unit(self) -> str:
        """Measurement unit for this sensor type."""
        return UNITS[self.sensor_type]

    @property
    def span(self) -> float:
        """Width of the acceptable range."""
        return self.acceptable_range[1] - self.acceptable_range[0]

    @property
    def edge_margin(self) -> float:
        """Minimum distance anomalies keep from the range edge (inside or outside)."""
        return max(0.02 * self.span, 0.01)

    @property
    def can_spike_up(self) -> bool:
        """Whether a value above the range is physically possible."""
        return self.acceptable_range[1] + self.edge_margin < PHYSICAL_BOUNDS[self.sensor_type][1]

    @property
    def can_spike_down(self) -> bool:
        """Whether a value below the range is physically possible."""
        return self.acceptable_range[0] - self.edge_margin > PHYSICAL_BOUNDS[self.sensor_type][0]


@dataclass(frozen=True)
class _ZoneProfile:
    """Normal operating envelope of a storage class: (mean, std, min, max) per sensor type."""

    storage_type: str
    temperature: tuple[float, float, float, float]
    humidity: tuple[float, float, float, float]


#: Profiles are assigned to zones by position: cold storage, dry goods, then ambient.
_ZONE_PROFILES: Final[tuple[_ZoneProfile, ...]] = (
    _ZoneProfile("cold_storage", (3.0, 0.25, 2.0, 4.0), (88.0, 1.8, 80.0, 95.0)),
    _ZoneProfile("dry_goods", (20.0, 0.5, 18.0, 22.0), (45.0, 2.5, 35.0, 55.0)),
    _ZoneProfile("ambient", (21.0, 1.2, 15.0, 27.0), (50.0, 3.0, 30.0, 65.0)),
)


def build_default_sensors(
    n_warehouses: int = 2,
    zones: Sequence[str] = ("A", "B", "C"),
    sensors_per_zone: int = 2,
    *,
    seed: int | None = None,
    reference_date: date | None = None,
) -> list[WarehouseSensor]:
    """Build a realistic sensor fleet.

    Sensors alternate temperature / humidity inside each zone, and ids look like
    ``wh1-A-temp-01``. Zone ``A`` is cold storage (2-4 °C), ``B`` dry goods
    (18-22 °C) and ``C`` ambient storage; further zones cycle through the profiles.

    Args:
        n_warehouses: Number of warehouses (ids ``wh1``, ``wh2``, ...).
        zones: Zone names inside every warehouse.
        sensors_per_zone: Sensors per zone, alternating temperature then humidity.
        seed: Seeds firmware / battery / calibration-date choices (not the readings).
        reference_date: "Today" for calibration dates. Defaults to the current UTC date.

    Raises:
        ValueError: A count is not positive or ``zones`` is empty.
    """
    if n_warehouses < 1:
        raise ValueError(f"n_warehouses must be >= 1, got {n_warehouses}")
    if sensors_per_zone < 1:
        raise ValueError(f"sensors_per_zone must be >= 1, got {sensors_per_zone}")
    if not zones:
        raise ValueError("zones must not be empty")
    rng = random.Random(seed)
    today = reference_date or datetime.now(UTC).date()
    sensors: list[WarehouseSensor] = []
    for w in range(1, n_warehouses + 1):
        for zone_index, zone in enumerate(zones):
            profile = _ZONE_PROFILES[zone_index % len(_ZONE_PROFILES)]
            counters: dict[SensorType, int] = {TEMPERATURE: 0, HUMIDITY: 0}
            for i in range(sensors_per_zone):
                sensor_type: SensorType = TEMPERATURE if i % 2 == 0 else HUMIDITY
                counters[sensor_type] += 1
                mean, std, low, high = getattr(profile, sensor_type)
                abbreviation = _TYPE_ABBREVIATION[sensor_type]
                sensors.append(
                    WarehouseSensor(
                        sensor_id=f"wh{w}-{zone}-{abbreviation}-{counters[sensor_type]:02d}",
                        warehouse_id=f"wh{w}",
                        location=zone,
                        sensor_type=sensor_type,
                        normal_mean=mean,
                        normal_std=std,
                        acceptable_range=(low, high),
                        firmware_version=rng.choice(_FIRMWARE_VERSIONS),
                        battery_level=round(rng.uniform(55.0, 100.0), 1),
                        last_calibration_date=today - timedelta(days=rng.randint(5, 240)),
                        storage_type=profile.storage_type,
                    )
                )
    return sensors


# --------------------------------------------------------------------------- #
# Internal state
# --------------------------------------------------------------------------- #
@dataclass
class _Episode:
    """An in-progress anomaly on one sensor."""

    kind: AnomalyType
    episode_id: str
    total: int
    direction: int
    magnitude: float  # point: spike level; contextual: level shift; collective: final drift
    index: int = 0


@dataclass
class _SensorState:
    """Mutable per-sensor simulation state."""

    sensor: WarehouseSensor
    rng: random.Random
    battery: float
    episode: _Episode | None = None
    episodes_started: int = 0


def _validate_length_range(name: str, value: tuple[int, int]) -> tuple[int, int]:
    low, high = value
    if low < 1 or high < low:
        raise ValueError(f"{name} must satisfy 1 <= low <= high, got {value}")
    return int(low), int(high)


def _validate_rate(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")
    return float(value)


def _validate_duration(name: str, value: int | None) -> int | None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _validate_direction(value: int | None) -> int | None:
    if value is not None and value not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1, got {value}")
    return value


def _to_utc(moment: datetime) -> datetime:
    """Return ``moment`` as an aware UTC datetime (naive values are assumed to be UTC)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _isoformat(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _activity_level(hour: float) -> float:
    """Return business activity in [0, 1]: 0 off-hours, peaking at 14:00."""
    if not _BUSINESS_START_HOUR <= hour < _BUSINESS_END_HOUR:
        return 0.0
    return math.sin(
        math.pi * (hour - _BUSINESS_START_HOUR) / (_BUSINESS_END_HOUR - _BUSINESS_START_HOUR)
    )


def _sample_within(rng: random.Random, mean: float, sigma: float, low: float, high: float) -> float:
    """Draw from a Gaussian truncated to ``[low, high]`` by rejection (clamped as a fallback)."""
    for _ in range(_MAX_RESAMPLES):
        value = rng.gauss(mean, sigma)
        if low <= value <= high:
            return value
    return min(max(mean, low), high)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class WarehouseSimulator:
    """Generate labelled warehouse sensor readings with injected anomalies.

    All sensors tick together on a simulated clock (``reading_interval_s``,
    default one second). Anomalies are *episodes*: once one starts on a sensor it
    runs for its full length and no other episode starts on that sensor
    meanwhile. Episodes start randomly at the configured rates, or on demand via
    the ``inject_*`` methods (used by the demo and tests).

    Anomaly rates are the approximate fraction of **all** readings that carry each
    label over a full simulated day (the contextual rate is concentrated into the
    off-hours window, so a daytime-only run produces none). Starting probabilities
    are ``rate / mean episode length``; because a sensor cannot start a new episode
    while one is active, realised fractions run a few percent below the target.
    """

    def __init__(
        self,
        sensors: Sequence[WarehouseSensor] | None = None,
        *,
        point_anomaly_rate: float = 0.02,
        contextual_anomaly_rate: float = 0.01,
        collective_anomaly_rate: float = 0.005,
        point_duration_range: tuple[int, int] = (1, 5),
        contextual_duration_range: tuple[int, int] = (3, 10),
        collective_length_range: tuple[int, int] = (10, 30),
        off_hours_start: int = 0,
        off_hours_end: int = 5,
        reading_interval_s: float = 1.0,
        start_time: datetime | None = None,
        seed: int | None = None,
    ) -> None:
        """Create a simulator.

        Args:
            sensors: Sensors to simulate. Defaults to :func:`build_default_sensors`.
            point_anomaly_rate: Target fraction of readings labelled ``point``.
            contextual_anomaly_rate: Target fraction of readings labelled ``contextual``.
            collective_anomaly_rate: Target fraction of readings labelled ``collective``.
            point_duration_range: Inclusive ``(min, max)`` readings per point anomaly.
            contextual_duration_range: Inclusive ``(min, max)`` readings per contextual
                anomaly (cut short if the off-hours window closes).
            collective_length_range: Inclusive ``(min, max)`` readings per drift.
            off_hours_start: First off-hours hour (0-23, inclusive).
            off_hours_end: Hour the off-hours window ends (0-23, exclusive). The window
                may wrap midnight (22 -> 6); equal start and end means no window.
            reading_interval_s: Simulated seconds between ticks.
            start_time: Simulated clock start (naive values are UTC). Defaults to now.
            seed: Makes the stream reproducible. ``None`` is non-deterministic.

        Raises:
            ValueError: Invalid rates, ranges, hours, interval, sensors or start time.
        """
        self._point_rate = _validate_rate("point_anomaly_rate", point_anomaly_rate)
        self._contextual_rate = _validate_rate("contextual_anomaly_rate", contextual_anomaly_rate)
        self._collective_rate = _validate_rate("collective_anomaly_rate", collective_anomaly_rate)
        self._point_range = _validate_length_range("point_duration_range", point_duration_range)
        self._contextual_range = _validate_length_range(
            "contextual_duration_range", contextual_duration_range
        )
        self._collective_range = _validate_length_range(
            "collective_length_range", collective_length_range
        )
        for name, hour in (("off_hours_start", off_hours_start), ("off_hours_end", off_hours_end)):
            if not 0 <= hour <= 23:
                raise ValueError(f"{name} must be within [0, 23], got {hour}")
        if reading_interval_s <= 0:
            raise ValueError(f"reading_interval_s must be positive, got {reading_interval_s}")

        self._off_start = off_hours_start
        self._off_hours_len = (off_hours_end - off_hours_start) % 24
        self._interval = float(reading_interval_s)
        self._start = (
            _to_utc(start_time) if start_time else datetime.now(UTC).replace(microsecond=0)
        )
        self._tick_index = 0
        self._seed = seed
        self._log = get_logger(__name__)

        fleet = (
            list(sensors)
            if sensors is not None
            else build_default_sensors(seed=seed, reference_date=self._start.date())
        )
        if not fleet:
            raise ValueError("at least one sensor is required")
        self._states: dict[str, _SensorState] = {}
        for sensor in fleet:
            if sensor.sensor_id in self._states:
                raise ValueError(f"duplicate sensor_id {sensor.sensor_id!r}")
            self._states[sensor.sensor_id] = _SensorState(
                sensor=sensor, rng=self._make_rng(sensor.sensor_id), battery=sensor.battery_level
            )

        self._p_point = self._point_rate / _mean(self._point_range)
        self._p_collective = self._collective_rate / _mean(self._collective_range)
        if self._off_hours_len:
            self._p_contextual = (
                self._contextual_rate * 24 / self._off_hours_len / _mean(self._contextual_range)
            )
        else:
            self._p_contextual = 0.0
        if self._p_point + self._p_contextual + self._p_collective > 1.0:
            raise ValueError("anomaly rates are too high for the configured episode lengths")

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg: WarehouseSimConfig,
        *,
        seed: int | None = None,
        start_time: datetime | None = None,
        sensors: Sequence[WarehouseSensor] | None = None,
    ) -> WarehouseSimulator:
        """Build a simulator from ``config.simulation.warehouse``.

        Args:
            cfg: The ``simulation.warehouse`` section of the project config.
            seed: Usually ``config.simulation.seed``.
            start_time: Simulated clock start (defaults to now, UTC).
            sensors: Custom sensor fleet; defaults to one built from ``cfg``.
        """
        start = _to_utc(start_time) if start_time else datetime.now(UTC).replace(microsecond=0)
        fleet = (
            list(sensors)
            if sensors is not None
            else build_default_sensors(
                cfg.n_warehouses,
                cfg.zones,
                cfg.sensors_per_zone,
                seed=seed,
                reference_date=start.date(),
            )
        )
        return cls(
            fleet,
            point_anomaly_rate=cfg.point_anomaly_rate,
            contextual_anomaly_rate=cfg.contextual_anomaly_rate,
            collective_anomaly_rate=cfg.collective_anomaly_rate,
            point_duration_range=cfg.point_duration_range,
            collective_length_range=cfg.collective_length_range,
            off_hours_start=cfg.off_hours_start,
            off_hours_end=cfg.off_hours_end,
            reading_interval_s=cfg.reading_interval_s,
            start_time=start,
            seed=seed,
        )

    def _make_rng(self, sensor_id: str) -> random.Random:
        """Return the sensor's private random stream (stable across fleet changes)."""
        if self._seed is None:
            return random.Random()
        return random.Random(f"{self._seed}:{sensor_id}")

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def sensors(self) -> tuple[WarehouseSensor, ...]:
        """All simulated sensors, in tick order."""
        return tuple(state.sensor for state in self._states.values())

    @property
    def reading_interval_s(self) -> float:
        """Simulated seconds between ticks."""
        return self._interval

    @property
    def current_time(self) -> datetime:
        """Simulated timestamp of the *next* tick."""
        return self._time_at(self._tick_index)

    def get_sensor(self, sensor_id: str) -> WarehouseSensor:
        """Return a sensor by id.

        Raises:
            KeyError: Unknown sensor id.
        """
        return self._state(sensor_id).sensor

    def is_off_hours(self, moment: datetime) -> bool:
        """Return whether ``moment`` (UTC hour) falls inside the off-hours window."""
        return self._in_off_hours(_to_utc(moment).hour)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def tick(self) -> list[WarehouseReading]:
        """Advance the clock one interval and return one reading per sensor."""
        moment = self._time_at(self._tick_index)
        self._tick_index += 1
        hour = moment.hour + moment.minute / 60 + moment.second / 3600
        activity = _activity_level(hour)
        off_hours = self._in_off_hours(moment.hour)
        stamp = _isoformat(moment)
        return [
            self._build_reading(state, stamp, activity, off_hours)
            for state in self._states.values()
        ]

    def simulate_all_sensors(
        self, n_ticks: int | None = None, duration_s: float | None = None
    ) -> Iterator[WarehouseReading]:
        """Yield readings for every sensor, tick by tick, in timestamp order.

        With neither argument the stream is endless. Each tick yields one reading
        per sensor, so ``n_ticks`` ticks produce ``n_ticks * len(sensors)`` readings.

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
    def inject_point_anomaly(
        self, sensor_id: str, duration: int | None = None, direction: Direction | None = None
    ) -> str:
        """Start a point anomaly on the sensor's next reading.

        The sensor jumps outside ``acceptable_range`` for ``duration`` readings.
        Replaces any episode already running on that sensor.

        Args:
            sensor_id: Target sensor.
            duration: Readings to last; random in ``point_duration_range`` if omitted.
            direction: ``1`` spike up, ``-1`` spike down; random if omitted.

        Returns:
            The new episode id (also reported in each reading's metadata).

        Raises:
            KeyError: Unknown sensor id.
            ValueError: Invalid duration/direction, or that direction is physically
                impossible for the sensor.
        """
        state = self._state(sensor_id)
        return self._start_point(
            state,
            _validate_duration("duration", duration),
            _validate_direction(direction),
            manual=True,
        ).episode_id

    def inject_contextual_anomaly(
        self, sensor_id: str, duration: int | None = None, direction: Direction | None = None
    ) -> str:
        """Start a contextual (out-of-hours activity) anomaly on the next reading.

        The value stays inside ``acceptable_range`` but shows a business-hours
        activity signature while the warehouse should be quiet. The episode ends
        early if the off-hours window closes. Replaces any running episode.

        Args:
            sensor_id: Target sensor.
            duration: Readings to last; random in ``contextual_duration_range`` if omitted.
            direction: ``1`` shift up, ``-1`` shift down; random if omitted.

        Returns:
            The new episode id.

        Raises:
            KeyError: Unknown sensor id.
            ValueError: Invalid arguments, or the next reading is not in off-hours.
        """
        state = self._state(sensor_id)
        if not self._in_off_hours(self.current_time.hour):
            raise ValueError(
                f"contextual anomalies need off-hours; next reading is at {self.current_time}"
            )
        return self._start_contextual(
            state,
            _validate_duration("duration", duration),
            _validate_direction(direction),
            manual=True,
        ).episode_id

    def inject_collective_anomaly(
        self, sensor_id: str, length: int | None = None, direction: Direction | None = None
    ) -> str:
        """Start a gradual equipment-degradation drift on the next reading.

        The sensor drifts linearly away from normal over ``length`` readings, then
        snaps back (maintenance). Replaces any episode already running.

        Args:
            sensor_id: Target sensor.
            length: Readings to last; random in ``collective_length_range`` if omitted.
            direction: ``1`` drift up, ``-1`` drift down; random if omitted
                (temperature favours upward drift).

        Returns:
            The new episode id.

        Raises:
            KeyError: Unknown sensor id.
            ValueError: Invalid length or direction.
        """
        state = self._state(sensor_id)
        return self._start_collective(
            state,
            _validate_duration("length", length),
            _validate_direction(direction),
            manual=True,
        ).episode_id

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _state(self, sensor_id: str) -> _SensorState:
        try:
            return self._states[sensor_id]
        except KeyError:
            raise KeyError(f"unknown sensor_id {sensor_id!r}") from None

    def _time_at(self, tick_index: int) -> datetime:
        return self._start + timedelta(seconds=tick_index * self._interval)

    def _in_off_hours(self, hour: int) -> bool:
        return self._off_hours_len > 0 and (hour - self._off_start) % 24 < self._off_hours_len

    def _new_episode(
        self, state: _SensorState, kind: AnomalyType, total: int, direction: int, magnitude: float
    ) -> _Episode:
        state.episodes_started += 1
        episode = _Episode(
            kind=kind,
            episode_id=f"{state.sensor.sensor_id}#{state.episodes_started}",
            total=total,
            direction=direction,
            magnitude=magnitude,
        )
        state.episode = episode
        return episode

    def _log_start(self, state: _SensorState, episode: _Episode, manual: bool) -> None:
        self._log.debug(
            "anomaly_episode_started",
            sensor_id=state.sensor.sensor_id,
            anomaly_type=episode.kind,
            episode_id=episode.episode_id,
            length=episode.total,
            manual=manual,
        )

    def _start_point(
        self,
        state: _SensorState,
        duration: int | None,
        direction: int | None = None,
        *,
        manual: bool = False,
    ) -> _Episode:
        sensor, rng = state.sensor, state.rng
        total = duration if duration is not None else rng.randint(*self._point_range)
        up = self._pick_spike_direction(state, direction)
        low, high = sensor.acceptable_range
        phys_min, phys_max = PHYSICAL_BOUNDS[sensor.sensor_type]
        overshoot = sensor.span * rng.uniform(*_POINT_OVERSHOOT_SPAN)
        if up > 0:
            level = max(min(high + overshoot, phys_max), high + sensor.edge_margin)
        else:
            level = min(max(low - overshoot, phys_min), low - sensor.edge_margin)
        episode = self._new_episode(state, POINT, total, up, level)
        self._log_start(state, episode, manual)
        return episode

    def _pick_spike_direction(self, state: _SensorState, direction: int | None) -> int:
        sensor = state.sensor
        if direction is not None:
            feasible = sensor.can_spike_up if direction > 0 else sensor.can_spike_down
            if not feasible:
                raise ValueError(
                    f"{sensor.sensor_id}: a spike with direction {direction} would leave the "
                    f"physical bounds of a {sensor.sensor_type} sensor"
                )
            return direction
        options = [d for d, ok in ((1, sensor.can_spike_up), (-1, sensor.can_spike_down)) if ok]
        return state.rng.choice(options)

    def _start_contextual(
        self,
        state: _SensorState,
        duration: int | None,
        direction: int | None = None,
        *,
        manual: bool = False,
    ) -> _Episode:
        sensor, rng = state.sensor, state.rng
        total = duration if duration is not None else rng.randint(*self._contextual_range)
        sign = direction if direction is not None else rng.choice((-1, 1))
        shift = sign * sensor.normal_std * rng.uniform(*_CONTEXTUAL_SHIFT_SIGMA)
        episode = self._new_episode(state, CONTEXTUAL, total, sign, shift)
        self._log_start(state, episode, manual)
        return episode

    def _start_collective(
        self,
        state: _SensorState,
        length: int | None,
        direction: int | None = None,
        *,
        manual: bool = False,
    ) -> _Episode:
        sensor, rng = state.sensor, state.rng
        total = length if length is not None else rng.randint(*self._collective_range)
        if direction is not None:
            sign = direction
        elif sensor.sensor_type == TEMPERATURE:
            sign = 1 if rng.random() < _COLLECTIVE_UP_PROBABILITY_TEMPERATURE else -1
        else:
            sign = rng.choice((-1, 1))
        drift = sign * sensor.normal_std * rng.uniform(*_COLLECTIVE_DRIFT_SIGMA)
        episode = self._new_episode(state, COLLECTIVE, total, sign, drift)
        self._log_start(state, episode, manual)
        return episode

    def _maybe_start_random_episode(self, state: _SensorState, off_hours: bool) -> _Episode | None:
        """Possibly start a random episode on a free sensor (one draw picks the type)."""
        draw = state.rng.random()
        p_contextual = self._p_contextual if off_hours else 0.0
        if draw < self._p_point:
            return self._start_point(state, None)
        draw -= self._p_point
        if draw < p_contextual:
            return self._start_contextual(state, None)
        draw -= p_contextual
        if draw < self._p_collective:
            return self._start_collective(state, None)
        return None

    def _build_reading(
        self, state: _SensorState, stamp: str, activity: float, off_hours: bool
    ) -> WarehouseReading:
        sensor = state.sensor
        episode = state.episode
        if episode is not None and episode.kind == CONTEXTUAL and not off_hours:
            state.episode = episode = None  # the quiet window closed: no longer contextual
        if episode is None:
            episode = self._maybe_start_random_episode(state, off_hours)

        mean = sensor.normal_mean + _ACTIVITY_SHIFT_SIGMA * sensor.normal_std * activity
        low, high = sensor.acceptable_range
        if episode is None:
            value = _sample_within(state.rng, mean, sensor.normal_std, low, high)
            value = min(max(round(value, 3), low), high)
            anomaly_type: AnomalyType | None = None
            episode_id: str | None = None
            step: int | None = None
        else:
            value = self._anomalous_value(state, episode, mean)
            anomaly_type, episode_id, step = episode.kind, episode.episode_id, episode.index
            episode.index += 1
            if episode.index >= episode.total:
                state.episode = None

        state.battery = max(state.battery - _BATTERY_DRAIN_PER_READING, 0.0)
        return {
            "sensor_id": sensor.sensor_id,
            "warehouse_id": sensor.warehouse_id,
            "timestamp": stamp,
            "reading_type": sensor.sensor_type,
            "value": value,
            "unit": sensor.unit,
            "is_anomaly": anomaly_type is not None,
            "anomaly_type": anomaly_type,
            "metadata": {
                "zone": sensor.location,
                "storage_type": sensor.storage_type,
                "firmware_version": sensor.firmware_version,
                "battery_level": round(state.battery, 2),
                "last_calibration_date": (
                    sensor.last_calibration_date.isoformat()
                    if sensor.last_calibration_date
                    else None
                ),
                "acceptable_range": [low, high],
                "episode_id": episode_id,
                "episode_step": step,
            },
        }

    def _anomalous_value(self, state: _SensorState, episode: _Episode, mean: float) -> float:
        sensor, rng = state.sensor, state.rng
        low, high = sensor.acceptable_range
        phys_min, phys_max = PHYSICAL_BOUNDS[sensor.sensor_type]
        margin = sensor.edge_margin
        if episode.kind == POINT:
            value = episode.magnitude + rng.gauss(0.0, _POINT_JITTER_SPAN * sensor.span)
            if episode.direction > 0:
                value = min(max(value, high + margin), phys_max)
            else:
                value = max(min(value, low - margin), phys_min)
        elif episode.kind == CONTEXTUAL:
            value = _sample_within(
                rng,
                mean + episode.magnitude,
                sensor.normal_std * _CONTEXTUAL_NOISE_FACTOR,
                low + margin,
                high - margin,
            )
        else:
            base = _sample_within(rng, mean, sensor.normal_std, low, high)
            progress = (episode.index + 1) / episode.total
            value = min(max(base + episode.magnitude * progress, phys_min), phys_max)
        return round(value, 3)


def _mean(bounds: tuple[int, int]) -> float:
    return (bounds[0] + bounds[1]) / 2
