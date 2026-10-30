"""
Gauge choices for the BSSN solver.

To add a gauge, write a class in lapse.py or shift.py with the matching
register decorator; nothing else needs editing.  To select one, set
`type` in the [lapse] or [shift_damping] table of the parameter file.
"""

from .registry import LAPSE_GAUGES, SHIFT_DAMPING, build_gauges
from .lapse import BonaMasso, OnePlusLog
from .shift import ShiftDamping, ConstantEta
