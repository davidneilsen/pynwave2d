"""
Registries mapping parameter-file gauge names to gauge classes.

A gauge class registers itself with a decorator,

    @register_lapse("1+log")
    class OnePlusLog(BonaMasso):
        ...

and is then selectable from a parameter file by its name:

    [lapse]
    type = "1+log"
    k = 2.0

Every other key in the table is passed to the class constructor, so the
constructor signature is the list of accepted parameters.
"""

LAPSE_GAUGES = {}
SHIFT_DAMPING = {}


def _register(registry, name):
    def deco(cls):
        if name in registry:
            raise ValueError(f"Gauge '{name}' is registered twice.")
        cls.name = name
        registry[name] = cls
        return cls

    return deco


def register_lapse(name):
    return _register(LAPSE_GAUGES, name)


def register_shift(name):
    return _register(SHIFT_DAMPING, name)


def _build(registry, table, section, default):
    spec = dict(default if table is None else table)
    name = spec.pop("type", None)
    if name not in registry:
        raise ValueError(
            f"[{section}] type = {name!r} not recognised. "
            f"Options: {', '.join(sorted(registry))}"
        )
    try:
        return registry[name](**spec)
    except TypeError as err:
        # unknown or missing keys surface here as constructor TypeErrors
        raise ValueError(f"[{section}] type = '{name}': {err}") from err


def build_gauges(params):
    """
    Construct the (lapse, shift damping) gauge objects from a parameter dict.
    Defaults, used when a table is absent: 1+log with k = 2, constant eta = 2.
    A top-level 'eta' (the pre-table format) means constant eta, so older
    parameter files still run.
    """
    shift_table = params.get("shift_damping")
    if "eta" in params:
        if shift_table is not None:
            raise ValueError(
                "Both a top-level 'eta' and a [shift_damping] table are given; "
                "remove the top-level 'eta'."
            )
        shift_table = {"type": "constant", "eta": params["eta"]}
    lapse = _build(LAPSE_GAUGES, params.get("lapse"), "lapse", {"type": "1+log"})
    shift = _build(SHIFT_DAMPING, shift_table, "shift_damping", {"type": "constant"})
    return lapse, shift
