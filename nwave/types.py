from enum import Enum

CompactFilterTypes = ["JTT6", "JTP6", "JTT8", "JTP8", "KP6"]


class BCType(Enum):
    """
    Enum for boundary condition types
    """

    NONE = 0
    RHS = 1
    FUNCTION = 2


class FilterType(Enum):
    """
    Enum for filter types
    """

    NONE = 1
    KREISS_OLIGER_O6 = 2
    KREISS_OLIGER_O8 = 3
    JTT6 = 4
    JTP6 = 5
    JTT8 = 6
    JTP8 = 7
    KP4 = 8


class FilterApply(Enum):
    """
    Enum for filter application
    """

    NONE = 1
    RHS = 2
    APPLY_VARS = 3
    APPLY_DERIVS = 4


class DerivType(Enum):
    NONE = 0
    D1_E44 = 1
    D2_E44 = 2
    D1_E642 = 3
    D2_E642 = 4
    D1_JP6 = 5
    D2_JP6 = 6
    D1_JT4 = 7
    D2_JT4 = 8
    D1_JT6 = 9
    D2_JT6 = 10
    D1_KP4 = 20
    D1_SP4 = 21
    D1_DE4 = 22


class CFDSolve(Enum):
    NONE = 0
    SCIPY = 1
    LUSOLVE = 2
    PENTAPY = 3
    D_INV = 4
    D_LU = 5
