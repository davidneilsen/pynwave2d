"""
Gauge choices for the BSSN solver.

To add a gauge, write a class in lapse.py, shift.py or a new module imported
here, with the matching register decorator; nothing else needs editing.  To select one, set
`type` in the [lapse] or [shift_damping] table of the parameter file.
"""

from .registry import LAPSE_GAUGES, SHIFT_DAMPING, build_gauges
from .lapse import LapseGauge, BonaMasso, OnePlusLog
from .telegrapher import Telegrapher
from .shift import ShiftDamping, ConstantEta
