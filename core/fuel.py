"""Fuel for normal flight and the pulse engine."""

import config
from core.mathutil import clamp


class FuelSystem:
    def __init__(self):
        self.fuel = float(config.PULSE_FUEL_MAX)
        self.pulse_fuel = float(config.PULSE_FUEL_MAX)
        self.max_fuel = float(config.PULSE_FUEL_MAX)
        self.max_pulse = float(config.PULSE_FUEL_MAX)

    def regen(self, dt, pulsing):
        if not pulsing:
            self.pulse_fuel = clamp(
                self.pulse_fuel + config.PULSE_FUEL_REGEN * dt, 0.0,
                self.max_pulse)
            self.fuel = clamp(self.fuel + config.PULSE_FUEL_REGEN * 0.5 * dt,
                             0.0, self.max_fuel)

    def drain_pulse(self, dt):
        self.pulse_fuel = clamp(
            self.pulse_fuel - config.PULSE_FUEL_DRAIN * dt, 0.0,
            self.max_pulse)

    def can_pulse(self):
        return self.pulse_fuel >= config.PULSE_FUEL_MIN_START

    def to_dict(self):
        return {"fuel": self.fuel, "pulse_fuel": self.pulse_fuel}

    @classmethod
    def from_dict(cls, data):
        fs = cls()
        if isinstance(data, dict):
            fs.fuel = float(data.get("fuel", fs.fuel))
            fs.pulse_fuel = float(data.get("pulse_fuel", fs.pulse_fuel))
        return fs
