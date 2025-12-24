from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    AsyncIterator,
    AsyncContextManager,
    Any,
    Callable,
    Iterable,
    Literal,
    Self,
    Sequence,
    Protocol,
    cast,
)
import inspect
import re
import sqlite3
import types
import typing

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.fields import PydanticUndefined  # type: ignore
import aiosqlite


def get_pydantic_model(v: Any | None) -> type[BaseModel] | None:
    if inspect.isclass(v) and issubclass(v, BaseModel):
        return v
    if isinstance(v, types.UnionType):
        for variant in typing.get_args(v):
            if (model := get_pydantic_model(variant)) is not None:
                return model
    return None


class TypedCursor[T: BaseModel](Protocol):
    @classmethod
    def wrap_cursor(cls, row_cls: type[T], cursor: aiosqlite.Cursor) -> Self:
        def row_factory(cursor: aiosqlite.Cursor, row: aiosqlite.Row) -> T:
            fields = row_cls.model_fields
            field_names = fields.keys()
            kwargs = dict(zip(field_names, row))
            for key, value in kwargs.items():
                field = fields[key]
                annotation = field.annotation
                if (
                    model := get_pydantic_model(annotation)
                ) is not None and value is not None:
                    kwargs[key] = model.model_validate_json(value)
            return row_cls(**kwargs)

        cursor.row_factory = row_factory  # type: ignore
        return cast(Self, cursor)

    async def __aiter__(self) -> AsyncIterator[T]: ...

    async def execute(self, sql: str, parameters: Iterable[Any] | None) -> Self: ...

    async def executemany(
        self, sql: str, parameters: Iterable[Iterable[Any]]
    ) -> Self: ...

    async def executescript(self, sql_script: str) -> Self: ...

    async def fetchone(self) -> T | None: ...

    async def fetchmany(self, size: int | None = None) -> Iterable[T]: ...

    async def fetchall(self) -> list[T]: ...

    async def close(self) -> None: ...

    @property
    def rowcount(self) -> int: ...

    @property
    def lastrowid(self) -> int | None: ...

    @property
    def arraysize(self) -> int: ...

    @arraysize.setter
    def arraysize(self, value: int) -> None: ...

    @property
    def connection(self) -> sqlite3.Connection: ...

    async def __aenter__(self): ...

    async def __aexit__(self, exc_type, exc_val, exc_tb): ...


@asynccontextmanager
async def select[T: TableModel](
    cls: type[T], db: aiosqlite.Connection, query: str = "", params: Sequence[Any] = ()
) -> AsyncIterator[TypedCursor[T]]:
    field_names = cls.model_fields.keys()
    query = "SELECT {} FROM {} {}".format(
        ", ".join(field_names), cls.__resolved_table_name__, query
    )
    async with db.execute(query, params) as cursor:
        yield TypedCursor.wrap_cursor(cls, cursor)


class TableModel(BaseModel):
    id: int = 0

    __indices__: list[tuple[str, ...]] = []
    __unique__: list[tuple[str, ...]] = []
    __resolved_table_name__: str = ""
    __table_name__: str | None = None

    def __init_subclass__(cls, **kwargs):
        if (table_name := cls.__table_name__) is None:
            table_name = re.sub(r"([a-zA-Z])([A-Z])", r"\1_\2", cls.__name__).lower()
        cls.__resolved_table_name__ = table_name
        super().__init_subclass__(**kwargs)

    @classmethod
    async def sqlite_init(cls, db: aiosqlite.Connection) -> None:
        for subcls in cls.__subclasses__():
            table_name = re.sub(r"([a-zA-Z])([A-Z])", r"\1_\2", subcls.__name__).lower()

            column_names = [name for name in subcls.model_fields.keys()]
            for name in column_names:
                if not re.match(r"^\w+$", name):
                    raise ValueError(f"invalid field name {name!r} on {subcls}")
            column_fields = {
                name: db_field(name, field)
                for name, field in subcls.model_fields.items()
            }

            async with db.execute(f"PRAGMA table_info({table_name})") as cursor:
                table_rows = await cursor.fetchall()

            if table_rows:
                # validate the existing columns
                db_columns: set[str] = set()
                for (
                    cid,
                    name,
                    field_type,
                    notnull,
                    default_value,
                    primary_key,
                    *_,
                ) in table_rows:
                    spec_parts = [
                        field_type.upper(),
                        ("PRIMARY KEY" if primary_key else ""),
                        ("NOT NULL" if notnull else ""),
                        (f"DEFAULT {default_value}" if default_value else ""),
                    ]
                    column_spec = " ".join([x for x in spec_parts if x])
                    db_columns.add(name)
                    if (model_spec := column_fields.get(name)) is not None:
                        if column_spec != model_spec:
                            raise TypeError(
                                f"db column spec does not match model {table_name=} {name=} {column_spec=} != {model_spec=}"
                            )
                    else:
                        raise TypeError(
                            f"db column missing from model {table_name=} {name=}"
                        )

                # create missing columns if possible
                missing_columns = set(column_names) - db_columns
                for name in missing_columns:
                    field = subcls.model_fields[name]
                    if field.default is PydanticUndefined:
                        raise TypeError(
                            f"db column missing and no default value present to create it {table_name=} column={name=}"
                        )
                    alter_query = "ALTER TABLE {} ADD COLUMN {} {}".format(
                        table_name, name, column_fields[name]
                    )
                    _ = await db.execute(alter_query)

            else:
                # create the table
                column_queries = [
                    "{} {}".format(name, field) for name, field in column_fields.items()
                ]

                # unique constraints
                # NOTE: no support for adding new unique constraints?
                for columns in subcls.__unique__:
                    for name in columns:
                        if not re.match(r"^\w+$", name):
                            raise ValueError(
                                f"invalid unique column {name!r} on {subcls}"
                            )
                        if name not in subcls.model_fields:
                            raise ValueError(
                                f"unique on missing column {name!r} on {subcls}"
                            )
                    column_list = ", ".join(columns)
                    column_queries.append(f"UNIQUE({column_list})")
                column_query = ",\n    ".join(column_queries)
                create_query = "CREATE TABLE IF NOT EXISTS {} (\n    {})".format(
                    table_name, column_query
                )
                _ = await db.execute(create_query)

            for columns in subcls.__indices__:
                index_name = "idx_" + ("__".join(columns))
                for name in columns:
                    if not re.match(r"^\w+$", name):
                        raise ValueError(f"invalid index column {name!r} on {subcls}")
                    if name not in subcls.model_fields:
                        raise ValueError(
                            f"index on missing column {name!r} on {subcls}"
                        )
                column_list = ", ".join(columns)
                _ = await db.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_list})"
                )

            await db.commit()

    @classmethod
    async def _create_typed(cls, db: aiosqlite.Connection, obj: Self) -> Self:
        names = [name for name in cls.model_fields.keys() if name != "id"]
        param_str = ", ".join("?" for _ in names)
        name_str = ", ".join(names)
        value_params = [getattr(obj, name) for name in names]
        value_params = [
            v.model_dump_json() if isinstance(v, BaseModel) else v for v in value_params
        ]

        query = f"INSERT INTO {cls.__resolved_table_name__} ({name_str}) VALUES ({param_str})"
        async with db.execute_insert(query, value_params) as rowid_tuple:
            assert rowid_tuple is not None
            obj.id = rowid_tuple[0]
        await db.commit()

        return obj

    @classmethod
    async def create[**P, S: Self](
        cls: Callable[P, S],
        db: aiosqlite.Connection,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> S:
        obj = cls(*args, **kwargs)
        return await obj._create_typed(db, obj)

    @classmethod
    def select(
        cls, db: aiosqlite.Connection, query: str = "", params: Sequence[Any] = ()
    ) -> AsyncContextManager[TypedCursor[Self]]:
        cursor_gen = select(cls, db, query=query, params=params) # type: ignore
        return cast(AsyncContextManager[TypedCursor[Self]], cursor_gen)

    @classmethod
    async def select_count(
        cls, db: aiosqlite.Connection, query: str = "", params: Sequence[Any] = ()
    ) -> int:
        query = "SELECT count(*) FROM {} {}".format(cls.__resolved_table_name__, query)
        async with db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row is not None else 0

    @classmethod
    async def remove(
        cls, db: aiosqlite.Connection, query: str, params: Sequence[Any] = ()
    ) -> None:
        if "where" not in query.lower():
            raise ValueError("remove() query must contain WHERE clause")
        query = f"DELETE FROM {cls.__resolved_table_name__} {query}"
        await db.execute(query, params)
        await db.commit()

    async def remove_one(self, db: aiosqlite.Connection) -> None:
        query = f"DELETE FROM {self.__class__.__resolved_table_name__} WHERE id=?"
        await db.execute(query, (self.id,))
        await db.commit()

    @classmethod
    async def _update(
        cls,
        db: aiosqlite.Connection,
        query: str,
        params: Sequence[Any] = (),
        /,
        **kwargs,
    ) -> None:
        if "where" not in query.lower():
            raise ValueError("update() query must contain WHERE clause")
        if not kwargs:
            return
        updates = [f"{name}=?" for name in kwargs]
        params = tuple(
            v.model_dump_json() if isinstance(v, BaseModel) else v
            for v in kwargs.values()
        ) + tuple(params)
        query = f"UPDATE {cls.__resolved_table_name__} SET {', '.join(updates)} {query}"
        await db.execute(query, params)
        await db.commit()

    @classmethod
    async def update(
        cls,
        db: aiosqlite.Connection,
        query: str,
        params: Sequence[Any] = (),
        /,
        **kwargs,
    ) -> None:
        cons = cls.model_construct()
        for k, v in kwargs.items():
            cls.__pydantic_validator__.validate_assignment(cons, k, v)
        await cls._update(db, query, params, **kwargs)

    async def update_one(self, db: aiosqlite.Connection, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.__pydantic_validator__.validate_assignment(self, k, v)
            setattr(self, k, v)
        await self._update(db, "WHERE id=?", (self.id,), **kwargs)


def db_field(name: str, field: FieldInfo) -> str:
    extra: list[str] = []
    if name == "id":
        extra.append("PRIMARY KEY")

    pytype = field.annotation
    origin = typing.get_origin(pytype)
    optional = False
    if origin is types.UnionType:
        args = typing.get_args(pytype)
        if len(args) == 2 and type(None) in args:
            optional = True
            pytype = [x for x in args if x is not type(None)][0]
    if name != "id" and not optional:
        extra.append("NOT NULL")

    if (default := field.default) is not PydanticUndefined:
        if default is None:
            extra.append("DEFAULT NULL")
        elif isinstance(default, (int, float)):
            extra.append(f"DEFAULT {default}")
        elif isinstance(default, str):
            extra.append(f"DEFAULT {default!r}")
        else:
            raise TypeError(f"unsupported default field {name=} {default=}")

    is_model = inspect.isclass(pytype) and issubclass(pytype, BaseModel)
    if is_model:
        dbtype = "JSONB"
    elif origin is Literal:
        dbtype = "TEXT"
    elif pytype is int:
        dbtype = "INTEGER"
    elif pytype is float:
        dbtype = "REAL"
    elif pytype is bool:
        dbtype = "BOOLEAN"
    elif pytype is str:
        dbtype = "TEXT"
    elif pytype is datetime:
        dbtype = "DATETIME"
    else:
        raise TypeError(f"unknown field type: {pytype}")

    return " ".join([dbtype] + extra)
