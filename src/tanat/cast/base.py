#!/usr/bin/env python3
"""Cast primitives, structural casts, and the recipe skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar

import polars as pl

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def apply_cast_exprs(
    lf: pl.LazyFrame,
    exprs: list[pl.Expr],
) -> pl.LazyFrame:
    """Apply pre-built cast expressions to *lf*.

    Returns *lf* unchanged when *exprs* is empty.
    """
    return lf.with_columns(exprs) if exprs else lf


def build_caster(recipe: list[pl.DataType]) -> Callable[[pl.Expr], pl.Expr]:
    """Build an expression caster from an ordered dtype recipe."""

    def caster(expr: pl.Expr) -> pl.Expr:
        for dtype in recipe:
            expr = expr.cast(dtype)
        return expr

    return caster


def probe_cast_recipe(
    lf: pl.LazyFrame,
    recipes: dict[str, list[pl.DataType]],
    n_rows: int = 10,
) -> None:
    """Validate multi-step cast recipes on a small sample of *lf*.

    Each column is cast through its recipe in order (``T0 → T1 → … → Tn``).
    Absent columns are silently skipped.

    Raises:
        TypeError: If any step fails, with column name and full chain in the message.
    """
    existing = set(lf.collect_schema().names())
    to_check = {c: dtypes for c, dtypes in recipes.items() if c in existing and dtypes}
    if not to_check:
        return
    non_null_filter = pl.any_horizontal(pl.col(c).is_not_null() for c in to_check)
    sample = lf.filter(non_null_filter).limit(n_rows)
    exprs = []
    for col, dtypes in to_check.items():
        expr = pl.col(col)
        for dtype in dtypes:
            expr = expr.cast(dtype)
        exprs.append(expr)
    cols_desc = ", ".join(
        f"'{c}' → {' → '.join(str(d) for d in dtypes)}"
        for c, dtypes in to_check.items()
    )
    try:
        sample.with_columns(exprs).collect()
    except Exception as exc:
        raise TypeError(
            f"Cast recipe validation failed on {n_rows}-row sample "
            f"({cols_desc}): {exc}"
        ) from exc


@dataclass(frozen=True)
class ScalarCast:
    """Ordered cast recipe for a single, well-known column."""

    recipe: list[pl.DataType] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return ``True`` when no cast step is registered."""
        return not self.recipe

    def final_dtype(self) -> pl.DataType | None:
        """Return the dtype the recipe ultimately casts to, ``None`` if empty."""
        return self.recipe[-1] if self.recipe else None

    def append(self, dtype: pl.DataType) -> ScalarCast:
        """Return a new recipe with *dtype* appended as a final step."""
        return replace(self, recipe=[*self.recipe, dtype])

    def copy(self) -> ScalarCast:
        """Return a deep copy of the recipe."""
        return replace(self, recipe=list(self.recipe))

    def caster(self) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster, or ``None`` when the recipe is empty."""
        if not self.recipe:
            return None
        return build_caster(self.recipe)


@dataclass(frozen=True)
class ColumnMapCast:
    """Ordered cast recipes for a named set of columns."""

    recipes: dict[str, list[pl.DataType]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return ``True`` when no column has a registered cast recipe."""
        return not self.recipes

    def append(self, schema: dict[str, pl.DataType]) -> ColumnMapCast:
        """Return a new map with each ``col → dtype`` of *schema* appended."""
        recipes = {col: list(recipe) for col, recipe in self.recipes.items()}
        for col, dtype in schema.items():
            recipes[col] = [*recipes.get(col, []), dtype]
        return replace(self, recipes=recipes)

    def copy(self) -> ColumnMapCast:
        """Return a deep copy of the map."""
        return replace(
            self, recipes={col: list(recipe) for col, recipe in self.recipes.items()}
        )

    def caster(self, col: str) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster for *col*, or ``None`` when unset."""
        recipe = self.recipes.get(col)
        if not recipe:
            return None
        return build_caster(recipe)

    def exprs(self) -> list[pl.Expr]:
        """Return one ``with_columns`` expression per registered column."""
        return [
            build_caster(recipe)(pl.col(col)) for col, recipe in self.recipes.items()
        ]

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply all registered column casts to *lf* (no-op when empty)."""
        exprs = self.exprs()
        return lf.with_columns(exprs) if exprs else lf

    def probe_lf(self, lf: pl.LazyFrame, n_rows: int = 10) -> None:
        """Eagerly evaluate an *n_rows* sample of *lf* to surface cast errors.

        The candidate recipe is applied only for the probe; *lf* itself remains
        lazy and unchanged for the caller.
        """
        probe_cast_recipe(lf, self.recipes, n_rows)


# ---------------------------------------------------------------------------
# Structural casts (shared between sequence and trajectory)
# ---------------------------------------------------------------------------

_CLOSED_DOMAIN_DTYPES = (pl.Enum, pl.Categorical)


def _is_closed_domain_dtype(dtype: pl.DataType) -> bool:
    """Return ``True`` for ``pl.Enum`` / ``pl.Categorical`` dtypes."""
    if dtype is pl.Categorical:
        return True
    return isinstance(dtype, _CLOSED_DOMAIN_DTYPES)


@dataclass(frozen=True)
class StructuralCasts:
    """Shared ordered casts consumed by stores: id and time index."""

    id: ScalarCast = field(default_factory=ScalarCast)
    time_index: ScalarCast = field(default_factory=ScalarCast)

    def is_empty(self) -> bool:
        """Return ``True`` when both id and time_index recipes are empty."""
        return self.id.is_empty() and self.time_index.is_empty()

    def copy(self) -> StructuralCasts:
        """Return a deep copy of the structural casts."""
        return replace(self, id=self.id.copy(), time_index=self.time_index.copy())

    def id_caster(self) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster for the id column, or ``None``."""
        return self.id.caster()

    def time_index_caster(self) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster for the time_index column, or ``None``."""
        return self.time_index.caster()

    def apply(
        self,
        lf: pl.LazyFrame,
        *,
        id_col: str | None = None,
        time_cols: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Apply id and (optionally) time-index structural casts to *lf*.

        Args:
            lf: Input LazyFrame.
            id_col: Name of the id column to cast.  ``None`` → skip.
            time_cols: Names of time columns to cast.  ``None`` / ``[]`` → skip.
        """
        exprs: list[pl.Expr] = []
        if id_col is not None:
            id_caster = self.id_caster()
            if id_caster is not None:
                exprs.append(id_caster(pl.col(id_col)))
        if time_cols:
            time_caster = self.time_index_caster()
            if time_caster is not None:
                exprs.extend(time_caster(pl.col(c)) for c in time_cols)
        return lf.with_columns(exprs) if exprs else lf

    def append_id(self, dtype: pl.DataType) -> StructuralCasts:
        """Append *dtype* to the id recipe; reject closed-domain dtypes.

        ``pl.Enum`` / ``pl.Categorical`` would conflict with any id outside
        the current view (the store consumes the cast on the full physical
        data). Materialise the view via ``save()`` first to narrow the
        domain.
        """
        if _is_closed_domain_dtype(dtype):
            raise TypeError(
                "cast_id() does not accept pl.Enum / pl.Categorical closed-domain "
                "dtypes: the ID column is consumed by the store on the full "
                "physical data (joins, grouping), not on the current view. "
                "Materialise the current view with save() first, then cast."
            )
        return replace(self, id=self.id.append(dtype))

    def append_time_index(self, dtype: pl.DataType) -> StructuralCasts:
        """Append *dtype* to the time_index recipe."""
        return replace(self, time_index=self.time_index.append(dtype))

    def probe(self, view) -> None:
        """Validate id and time_index recipes against full structural store data."""
        store = view._store  # pylint: disable=protected-access
        self._probe_id(store)
        self._probe_time_index(store)

    def _probe_id(self, store) -> None:
        """Validate the ID cast recipe on the store main index."""
        if self.id.is_empty():
            return
        id_col = store.main_id_col
        probe_cast_recipe(store.main_index.select(id_col), {id_col: self.id.recipe})

    def _probe_time_index(self, store) -> None:
        """Validate the time-index cast recipe on sequence time-index data."""
        if self.time_index.is_empty():
            return
        time_store = self._resolve_time_index_store(store)
        lf = time_store.time_index()
        columns = lf.collect_schema().names()
        probe_cast_recipe(lf, {col: self.time_index.recipe for col in columns})

    @staticmethod
    def _resolve_time_index_store(store):
        """Return the store that owns physical time-index rows."""
        # Sequence
        if hasattr(store, "time_index"):
            return store
        # Trajectory: delegate to any linked sequence store (they're homogeneous).
        stores = getattr(store, "sequence_stores", None)
        if stores:
            return next(iter(stores.values()))
        raise RuntimeError(
            "No sequence store linked - cannot validate structural cast recipe."
        )


# ---------------------------------------------------------------------------
# Recipe skeleton
# ---------------------------------------------------------------------------


class BaseCastRecipe:
    """Common methods shared by sequence- and trajectory-level recipes.

    Subclasses must:

    * be decorated with ``@dataclass(frozen=True)``,
    * declare ``structural: StructuralCasts`` and ``features: <FeatureCasts>``
      with default factories,
    * set ``_FEATURES_CLS`` to the dataclass used for ``features``.
    """

    # Subclass attributes declared here only for type-checker visibility.
    structural: StructuralCasts
    features: object

    # Subclass hook used by :meth:`coerce` to rebuild ``features`` from a dict.
    _FEATURES_CLS: ClassVar[type]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return ``True`` when no structural and no feature cast is registered."""
        return self.structural.is_empty() and self.features.is_empty()

    def copy(self):
        """Return a deep copy of the recipe."""
        return replace(
            self, structural=self.structural.copy(), features=self.features.copy()
        )

    # ------------------------------------------------------------------
    # Read-only shortcuts (structural side)
    # ------------------------------------------------------------------

    @property
    def id(self) -> list[pl.DataType]:
        """Ordered dtype steps of the id recipe."""
        return self.structural.id.recipe

    @property
    def time_index(self) -> list[pl.DataType]:
        """Ordered dtype steps of the time_index recipe."""
        return self.structural.time_index.recipe

    @property
    def static(self) -> dict[str, list[pl.DataType]]:
        """Per-column dtype steps of the static feature recipes."""
        return self.features.static.recipes

    @property
    def id_dtype(self) -> pl.DataType | None:
        """Final dtype of the id recipe, ``None`` if no cast is registered."""
        return self.structural.id.final_dtype()

    @property
    def time_index_dtype(self) -> pl.DataType | None:
        """Final dtype of the time_index recipe, ``None`` if no cast is registered."""
        return self.structural.time_index.final_dtype()

    def id_caster(self) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster for the id column, or ``None``."""
        return self.structural.id_caster()

    def time_index_caster(self) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster for the time_index column, or ``None``."""
        return self.structural.time_index_caster()

    def static_caster(self, col: str) -> Callable[[pl.Expr], pl.Expr] | None:
        """Return the compiled caster for static feature *col*, or ``None``."""
        return self.features.static.caster(col)

    # ------------------------------------------------------------------
    # Functional update
    # ------------------------------------------------------------------

    def replace(self, **kwargs):
        """Functional update.

        Recognised keys:

        * ``structural=`` / ``features=``: canonical nested form.
        * ``id=``        : shortcut to reset the structural ``id`` recipe.
        * ``time_index=``: shortcut to reset the structural ``time_index`` recipe.
        """
        structural = kwargs.pop("structural", self.structural)
        features = kwargs.pop("features", self.features)
        if "id" in kwargs:
            structural = replace(structural, id=ScalarCast(kwargs.pop("id")))
        if "time_index" in kwargs:
            structural = replace(
                structural, time_index=ScalarCast(kwargs.pop("time_index"))
            )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown cast recipe field(s): {unknown}")
        return replace(self, structural=structural, features=features)

    # ------------------------------------------------------------------
    # Coercion
    # ------------------------------------------------------------------

    @classmethod
    def coerce(cls, value):
        """Coerce *value* into a recipe instance.

        Accepted:

        * ``None`` → default recipe.
        * a recipe instance → returned unchanged.
        * a dict in the canonical nested form
          ``{"structural": ..., "features": ...}`` (each side may itself be a
          dict, which is then unpacked into the matching dataclass).
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            kwargs = dict(value)
            if isinstance(kwargs.get("structural"), dict):
                kwargs["structural"] = StructuralCasts(**kwargs["structural"])
            if isinstance(kwargs.get("features"), dict):
                kwargs["features"] = cls._FEATURES_CLS(**kwargs["features"])
            return cls(**kwargs)
        raise TypeError(
            f"'cast_recipe' must be a {cls.__name__} instance or dict, "
            f"got {type(value).__name__}"
        )
