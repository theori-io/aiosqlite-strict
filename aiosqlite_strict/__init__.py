from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    AsyncIterator,
    AsyncContextManager,
    Any,
    Callable,
    ClassVar,
    Iterable,
    Literal,
    Optional,
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

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo
from pydantic.fields import PydanticUndefined # pyright: ignore
import aiosqlite
import orjson


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
    def wrap_cursor[T1: TableModel](cls, row_cls: type[T1], cursor: aiosqlite.Cursor) -> Self:
        def row_factory(cursor: aiosqlite.Cursor, row: aiosqlite.Row) -> T1:
            fields = row_cls.__sql_fields__
            field_names = fields.keys()
            kwargs = dict(zip(field_names, row))
            kwargs = {
                k: orjson.loads(v) if fields[k].dbtype == "JSONB" else v
                for k, v in kwargs.items()
            }
            return row_cls.model_validate(kwargs)

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
        ", ".join(field_names), cls.__table__, query
    )
    async with db.execute(query, params) as cursor:
        yield TypedCursor.wrap_cursor(cls, cursor)


@dataclass(slots=True)
class SqlField:
    name: str
    dbtype: str
    schema: str
    adapter: TypeAdapter = field(compare=False)
    optional: bool = False
    model: bool = False


class TableModel(BaseModel):
    id: int = 0

    __indices__: ClassVar[list[tuple[str, ...]]] = []
    __unique__: ClassVar[list[tuple[str, ...]]] = []
    __table__: ClassVar[str] = ""
    __sql_fields__: ClassVar[dict[str, SqlField]]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if (table_name := kwargs.get("table")) is None:
            table_name = re.sub(r"([a-zA-Z])([A-Z])", r"\1_\2", cls.__name__).lower()
        cls.__table__ = table_name

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        cls.__sql_fields__ = {
            name: db_field(name, field)
            for name, field in cls.model_fields.items()
        }

    @classmethod
    async def sqlite_init(cls, db: aiosqlite.Connection) -> None:
        for subcls in cls.__subclasses__():
            table_name = subcls.__table__

            column_names = [name for name in subcls.model_fields.keys()]
            for name in column_names:
                if not re.match(r"^\w+$", name):
                    raise ValueError(f"invalid field name {name!r} on {subcls}")
            column_fields = subcls.__sql_fields__

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
                        if column_spec != model_spec.schema:
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
                        table_name, name, column_fields[name].schema
                    )
                    _ = await db.execute(alter_query)

            else:
                # create the table
                column_queries = [
                    "{} {}".format(name, field.schema) for name, field in column_fields.items()
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

    def model_dump_sql(self) -> dict[str, Any]:
        sql_fields = self.__sql_fields__
        obj = self.model_dump(mode="json")
        return {
            k: orjson.dumps(v) if sql_fields[k].dbtype == "JSONB" else v
            for k, v in obj.items()
        }

    @classmethod
    def model_validate_sql(cls, obj: dict[str, Any]) -> Self:
        sql_fields = self.__sql_fields__
        obj = {
            k: orjson.loads(v) if sql_fields[k].dbtype == "JSONB" else v
            for k, (v, f) in obj.items()
        }
        return cls.model_validate(obj, mode="json")

    @classmethod
    async def _create_typed(cls, db: aiosqlite.Connection, obj: Self) -> Self:
        model = obj.model_dump_sql()
        model.pop("id", None)

        names = list(model.keys())
        param_str = ", ".join("?" for _ in names)
        name_str = ", ".join(names)
        value_params = list(model.values())

        query = f"INSERT INTO {cls.__table__} ({name_str}) VALUES ({param_str})"
        async with db.execute_insert(query, value_params) as rowid_tuple:
            assert rowid_tuple is not None
            obj.id = rowid_tuple[0]
        await db.commit()

        return obj

    @classmethod
    async def create[**P, S: Self]( # ty: ignore[invalid-type-variable-bound] # pyright: ignore[reportGeneralTypeIssues]
        cls: Callable[P, S], # pyright: ignore
        db: aiosqlite.Connection,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> S: # pyright: ignore[reportInvalidTypeVarUse]
        obj = cls(*args, **kwargs)
        return await obj._create_typed(db, obj) # ty: ignore[unresolved-attribute]

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
        query = "SELECT count(*) FROM {} {}".format(cls.__table__, query)
        async with db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row is not None else 0

    @classmethod
    async def remove(
        cls, db: aiosqlite.Connection, query: str, params: Sequence[Any] = ()
    ) -> None:
        if "where" not in query.lower():
            raise ValueError("remove() query must contain WHERE clause")
        query = f"DELETE FROM {cls.__table__} {query}"
        await db.execute(query, params)
        await db.commit()

    async def remove_one(self, db: aiosqlite.Connection) -> None:
        query = f"DELETE FROM {self.__class__.__table__} WHERE id=?"
        await db.execute(query, (self.id,))
        await db.commit()

    @classmethod
    async def _update(
        cls,
        db: aiosqlite.Connection,
        query: str,
        params: Sequence[Any] = (),
        /,
        **kwargs: Any,
    ) -> None:
        if "where" not in query.lower():
            raise ValueError("update() query must contain WHERE clause")
        if not kwargs:
            return
        sql_fields = cls.__sql_fields__
        updates = [f"{name}=?" for name in kwargs]

        kwargs_fields = {k: (v, sql_fields[k]) for k, v in kwargs.items()}
        params = tuple(
            f.adapter.dump_json(v) if f.dbtype == "JSONB" else f.adapter.dump_python(v, mode="json")
            for k, (v, f) in kwargs_fields.items()
        ) + tuple(params)

        query = f"UPDATE {cls.__table__} SET {', '.join(updates)} {query}"
        await db.execute(query, params)
        await db.commit()

    @classmethod
    async def update(
        cls,
        db: aiosqlite.Connection,
        query: str,
        params: Sequence[Any] = (),
        /,
        **kwargs: Any,
    ) -> None:
        cons = cls.model_construct()
        for k, v in kwargs.items():
            v2 = cls.__pydantic_validator__.validate_assignment(cons, k, v)
        await cls._update(db, query, params, **kwargs)

    async def update_one(self, db: aiosqlite.Connection, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.__pydantic_validator__.validate_assignment(self, k, v)
            setattr(self, k, v)
        await self._update(db, "WHERE id=?", (self.id,), **kwargs)


def db_field(name: str, field: FieldInfo) -> SqlField:
    extra: list[str] = []
    if name == "id":
        extra.append("PRIMARY KEY")

    pytype = field.annotation
    origin = typing.get_origin(pytype)
    optional = False
    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(pytype)
        if len(args) == 2 and type(None) in args:
            optional = True
            pytype = [x for x in args if x is not type(None)][0]
            origin = typing.get_origin(pytype)

    if name != "id" and not optional:
        extra.append("NOT NULL")

    if (default := field.default) is not PydanticUndefined:
        match default:
            case None:
                extra.append("DEFAULT NULL")
            case int(n):
                extra.append("DEFAULT {}".format(int(n)))
            case float(n):
                extra.append("DEFAULT {}".format(float(n)))
            case str(s):
                extra.append("DEFAULT {!r}".format(str(s)))
            case _:
                raise TypeError(f"unsupported default field {name=} {default=}")

    is_model = inspect.isclass(pytype) and issubclass(pytype, BaseModel)
    if origin in (types.UnionType, typing.Union):
        if all(inspect.isclass(arg) and issubclass(arg, BaseModel) for arg in typing.get_args(pytype)):
            is_model = True

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
    elif origin in (list, tuple, set):
        dbtype = "JSONB"
    else:
        raise TypeError(f"unsupported type {pytype}")

    schema = " ".join([dbtype] + extra)
    return SqlField(
        name=name,
        dbtype=dbtype,
        schema=schema,
        adapter=TypeAdapter(field.annotation),
        optional=optional,
        model=is_model,
    )
