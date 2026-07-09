"""User transforms loaded from plain Python scripts — in-memory only, never persisted.

Point projector at a `.py` file and it imports it (fresh module namespace) and
discovers everything usable as a per-frame transform:

- plain functions — `(pts) -> labels|pts` or `(pts, labels) -> (pts, labels)|labels`;
- classes with such a `__call__` (instantiated with their default kwargs; numeric
  constructor kwargs become UI parameters);
- apairo-style preprocessors — anything exposing a `process` method.

Nothing is written to disk: a transform producing integer per-point labels becomes a
*virtual* LABELS channel (confusion / hover / class chips work on it immediately); a
transform returning a cloud replaces its target cloud channel for display.
Persistent derived channels remain `apairo_preprocess.run_preprocess`'s job.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.labels import LabelSet
from ..core.source import ChannelKind, ChannelSpec, Frame

_NUMERIC = (int, float)


@dataclass
class TransformSpec:
    id: str
    name: str
    source: str  # script path
    factory: object  # callable(**params) -> per-frame callable, or the fn
    params: dict = field(default_factory=dict)  # name -> default (numeric)
    is_factory: bool = False  # True: instantiate with params before use
    doc: str = ""


@dataclass
class ActiveTransform:
    spec: TransformSpec
    cloud: str  # target cloud channel
    params: dict
    fn: object = None  # bound per-frame callable
    last_error: str | None = None

    def bind(self) -> None:
        from functools import partial

        self.last_error = None
        try:
            if self.spec.is_factory:
                self.fn = self.spec.factory(**self.params)
            elif self.params:
                self.fn = partial(self.spec.factory, **self.params)
            else:
                self.fn = self.spec.factory
        except Exception as e:  # noqa: BLE001 — user code
            self.fn = None
            self.last_error = f"init failed: {e}"


def _numeric_kwargs(sig: inspect.Signature) -> dict:
    out = {}
    for p in sig.parameters.values():
        if (
            p.default is not inspect.Parameter.empty
            and isinstance(p.default, _NUMERIC)
            and not isinstance(p.default, bool)
        ):
            out[p.name] = p.default
    return out


def _positional_arity(fn) -> int | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            n += 1
        elif p.kind is p.VAR_POSITIONAL:
            return 2
    return n


def load_script(path: str) -> list[TransformSpec]:
    """Import a script (fresh namespace) and list its transform candidates."""
    p = Path(path).expanduser().resolve()
    if not p.is_file() or p.suffix != ".py":
        raise ValueError(f"not a python script: {p}")
    mod_name = f"projector_plugin_{abs(hash(str(p)))}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # allows dataclasses/pickle inside the script
    spec.loader.exec_module(module)  # user code runs here (their own script)

    found: list[TransformSpec] = []

    def add(name, factory, params, is_factory, doc=""):
        found.append(
            TransformSpec(
                id=f"{p.name}::{name}",
                name=name,
                source=str(p),
                factory=factory,
                params=params,
                is_factory=is_factory,
                doc=(doc or "").strip().splitlines()[0] if doc else "",
            )
        )

    # Explicit registry wins: TRANSFORMS = {"name": callable_or_class, ...}
    explicit = getattr(module, "TRANSFORMS", None)
    members = (
        list(explicit.items())
        if isinstance(explicit, dict)
        else [(n, v) for n, v in vars(module).items() if not n.startswith("_")]
    )

    for name, obj in members:
        if inspect.isclass(obj) and getattr(obj, "__module__", None) == mod_name:
            # a class is a candidate when its instances are per-frame callables
            has_call = "__call__" in vars(obj) or any("__call__" in vars(b) for b in obj.__mro__[1:-1])
            proc = getattr(obj, "process", None)
            if has_call or callable(proc):
                try:
                    params = _numeric_kwargs(inspect.signature(obj.__init__))
                except (TypeError, ValueError):
                    params = {}
                add(name, obj, params, is_factory=True, doc=obj.__doc__)
        elif inspect.isfunction(obj) and obj.__module__ == mod_name:
            arity = _positional_arity(obj)
            if arity in (1, 2):
                add(name, obj, _numeric_kwargs(inspect.signature(obj)), is_factory=False, doc=obj.__doc__)
        elif (
            not inspect.ismodule(obj) and not inspect.isclass(obj) and callable(getattr(obj, "process", None))
        ):
            add(name, lambda o=obj: o, {}, is_factory=True, doc=getattr(type(obj), "__doc__", ""))
    return found


def _call(fn, pts: np.ndarray, labels: np.ndarray | None):
    """Invoke a user transform, tolerating the common signatures."""
    if callable(getattr(fn, "process", None)) and not inspect.isfunction(fn):
        fn = fn.process  # apairo-style preprocessor object
    arity = _positional_arity(fn)
    if arity == 2:
        return fn(pts, labels)
    return fn(pts)


def apply_transforms(frame: Frame, active: list[ActiveTransform]) -> Frame:
    """Run the active transforms on a frame, in order — all in memory.

    Integer `(N,)` output aligned to the cloud → a virtual labels channel named
    after the transform. `(M, 3+)` output → replaces the target cloud (labelings
    of a differently-sized result are the views' length-check problem, by design).
    A `(pts, labels)` tuple replaces the cloud AND adds the labels channel.
    """
    channels = dict(frame.channels)
    for t in active:
        if t.fn is None:
            continue
        pts = channels.get(t.cloud)
        if pts is None:
            continue
        try:
            out = _call(t.fn, np.asarray(pts), None)
        except Exception as e:  # noqa: BLE001 — user code must never kill the stream
            t.last_error = str(e)
            continue
        t.last_error = None
        name = t.spec.id.split("::", 1)[1]
        if isinstance(out, tuple) and len(out) == 2:
            new_pts, labels = out
            if new_pts is not None:
                channels[t.cloud] = np.asarray(new_pts)
            if labels is not None:
                channels[name] = np.asarray(labels)
        elif isinstance(out, np.ndarray) and out.ndim == 1 and np.issubdtype(out.dtype, np.integer):
            channels[name] = out
        elif isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] >= 3:
            channels[t.cloud] = out
        else:
            t.last_error = f"unsupported output: {type(out).__name__}" + (
                f" shape {out.shape}" if isinstance(out, np.ndarray) else ""
            )
    return Frame(channels=channels, timestamps=frame.timestamps, indices=frame.indices)


def virtual_specs(active: list[ActiveTransform]) -> list[ChannelSpec]:
    """Session channel specs for label-producing transforms (auto palette, ignore=-1
    so every produced id gets a color)."""
    from ..adapters.apairo_source import _PALETTE
    from ..core.labels import LabelClass

    out = []
    for t in active:
        name = t.spec.id.split("::", 1)[1]
        # ids are unknown until the first run; provision a generous 0..11 palette
        classes = LabelSet(
            [LabelClass(i, str(i), _PALETTE[i % len(_PALETTE)]) for i in range(12)],
            ignore_id=-1,
        )
        out.append(
            ChannelSpec(name, ChannelKind.LABELS, np.dtype("int32"), (None,), of=t.cloud, labelset=classes)
        )
    return out
