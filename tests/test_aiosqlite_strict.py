from datetime import datetime
from typing import Literal
import sqlite3

from pydantic import BaseModel, ValidationError
import aiosqlite
import pytest

from aiosqlite_strict import TableModel, db_field, get_pydantic_model


def test_get_pydantic_model() -> None:
    class Child(BaseModel):
        value: int

    assert get_pydantic_model(Child) is Child
    assert get_pydantic_model(Child | None) is Child
    assert get_pydantic_model(int | None) is None
    assert get_pydantic_model(str) is None


def test_db_field_types_and_defaults() -> None:
    class Nested(BaseModel):
        label: str

    class Fields(BaseModel):
        id: int = 0
        count: int
        ratio: float
        active: bool
        name: str
        created_at: datetime
        kind: Literal["a", "b"]
        note: str | None = None
        nickname: str = "nick"
        meta: Nested

    fields = Fields.model_fields
    assert db_field("id", fields["id"]) == "INTEGER PRIMARY KEY DEFAULT 0"
    assert db_field("count", fields["count"]) == "INTEGER NOT NULL"
    assert db_field("ratio", fields["ratio"]) == "REAL NOT NULL"
    assert db_field("active", fields["active"]) == "BOOLEAN NOT NULL"
    assert db_field("name", fields["name"]) == "TEXT NOT NULL"
    assert db_field("created_at", fields["created_at"]) == "DATETIME NOT NULL"
    assert db_field("kind", fields["kind"]) == "TEXT NOT NULL"
    assert db_field("note", fields["note"]) == "TEXT DEFAULT NULL"
    assert db_field("nickname", fields["nickname"]) == "TEXT NOT NULL DEFAULT 'nick'"
    assert db_field("meta", fields["meta"]) == "JSONB NOT NULL"


def test_db_field_invalid_types() -> None:
    class InvalidDefault(BaseModel):
        bad: int = 1

    class BadDefaults(BaseModel):
        items: int = [1, 2, 3]  # type: ignore[assignment]

    class BadTypes(BaseModel):
        blob: bytes

    with pytest.raises(TypeError):
        db_field("items", BadDefaults.model_fields["items"])

    with pytest.raises(TypeError):
        db_field("blob", BadTypes.model_fields["blob"])

    assert (
        db_field("bad", InvalidDefault.model_fields["bad"])
        == "INTEGER NOT NULL DEFAULT 1"
    )


def test_table_name_resolution() -> None:
    class CamelCase(TableModel):
        name: str

    class CustomName(TableModel):
        __table_name__ = "custom_table"
        name: str

    assert CamelCase.__resolved_table_name__ == "camel_case"
    assert CustomName.__resolved_table_name__ == "custom_table"


@pytest.mark.asyncio
async def test_sqlite_init_and_crud() -> None:
    class Meta(BaseModel):
        tag: str

    class Base(TableModel):
        pass

    class UserProfile(Base):
        __indices__ = [("name",)]
        __unique__ = [("email",)]

        name: str
        email: str
        meta: Meta

    async with aiosqlite.connect(":memory:") as db:
        await Base.sqlite_init(db)

        assert UserProfile.__resolved_table_name__ == "user_profile"

        user1 = await UserProfile.create(
            db, name="name1", email="e1", meta=Meta(tag="t1")
        )
        user2 = await UserProfile.create(
            db, name="name2", email="e2", meta=Meta(tag="t2")
        )

        assert user1.id == 1
        assert user2.id == 2

        async with UserProfile.select(db) as cursor:
            rows = await cursor.fetchall()

        assert len(rows) == 2
        assert isinstance(rows[0], UserProfile)
        assert isinstance(rows[0].meta, Meta)
        assert rows[0].meta.tag == "t1"

        async with UserProfile.select(db, "WHERE name=?", ("name2",)) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row.email == "e2"

        count = await UserProfile.select_count(db)
        assert count == 2

        await user1.update_one(db, name="name1.1")
        assert user1.name == "name1.1"

        await user1.update_one(db)

        with pytest.raises(ValidationError):
            await user1.update_one(db, name=123)  # type: ignore[arg-type]

        async with UserProfile.select(db, "WHERE name=?", ("name1.1",)) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row.email == "e1"

        async with db.execute("PRAGMA index_list(user_profile)") as cursor:
            index_rows = await cursor.fetchall()
        index_names = {row[1] for row in index_rows}
        assert "idx_name" in index_names

        with pytest.raises(sqlite3.IntegrityError):
            await UserProfile.create(db, name="name3", email="e2", meta=Meta(tag="t3"))


@pytest.mark.asyncio
async def test_update_and_remove() -> None:
    class Base(TableModel):
        pass

    class Item(Base):
        name: str
        quantity: int

    async with aiosqlite.connect(":memory:") as db:
        await Base.sqlite_init(db)

        item1 = await Item.create(db, name="first", quantity=1)
        item2 = await Item.create(db, name="second", quantity=2)

        await Item.update(db, "WHERE id=?", [item1.id], name="updated", quantity=3)

        async with Item.select(db, "WHERE id=?", (item1.id,)) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row.name == "updated"
        assert row.quantity == 3

        await Item.remove(db, "WHERE id=?", (item2.id,))
        assert await Item.select_count(db) == 1

        with pytest.raises(ValueError):
            await Item.update(db, "id=?", (item1.id,), name="bad")

        with pytest.raises(ValueError):
            await Item.remove(db, "id=?", (item1.id,))


@pytest.mark.asyncio
async def test_remove_one() -> None:
    class Base(TableModel):
        pass

    class Item(Base):
        name: str
        quantity: int

    async with aiosqlite.connect(":memory:") as db:
        await Base.sqlite_init(db)

        item1 = await Item.create(db, name="first", quantity=1)
        item2 = await Item.create(db, name="second", quantity=2)

        await item1.remove_one(db)

        async with Item.select(db, "WHERE id=?", (item1.id,)) as cursor:
            row = await cursor.fetchone()

        assert row is None
        assert await Item.select_count(db) == 1

        async with Item.select(db, "WHERE id=?", (item2.id,)) as cursor:
            row = await cursor.fetchone()

        assert row is not None

@pytest.mark.asyncio
async def test_sqlite_init_existing_table_mismatch() -> None:
    class Base(TableModel):
        pass

    class Widget(Base):
        name: int

    async with aiosqlite.connect(":memory:") as db:
        await db.execute(
            "CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
        )
        await db.commit()

        with pytest.raises(TypeError):
            await Base.sqlite_init(db)


@pytest.mark.asyncio
async def test_schema_update_add_field_with_default() -> None:
    class Base(TableModel):
        pass

    class Record(Base):
        name: str
        note: str = "n/a"

    async with aiosqlite.connect(":memory:") as db:
        await db.execute(
            "CREATE TABLE record (id INTEGER PRIMARY KEY DEFAULT 0, name TEXT NOT NULL)"
        )
        await db.commit()

        await Base.sqlite_init(db)

        async with db.execute("PRAGMA table_info(record)") as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        assert "note" in columns


@pytest.mark.asyncio
async def test_schema_update_add_field_without_default_fails() -> None:
    class Base(TableModel):
        pass

    class Record(Base):
        name: str
        note: str

    async with aiosqlite.connect(":memory:") as db:
        await db.execute(
            "CREATE TABLE record (id INTEGER PRIMARY KEY DEFAULT 0, name TEXT NOT NULL)"
        )
        await db.commit()

        with pytest.raises(TypeError):
            await Base.sqlite_init(db)


@pytest.mark.asyncio
async def test_schema_update_rejects_extra_db_column() -> None:
    class Base(TableModel):
        pass

    class Record(Base):
        name: str

    async with aiosqlite.connect(":memory:") as db:
        await db.execute(
            "CREATE TABLE record (id INTEGER PRIMARY KEY DEFAULT 0, name TEXT NOT NULL, extra TEXT NOT NULL)"
        )
        await db.commit()

        with pytest.raises(TypeError):
            await Base.sqlite_init(db)


@pytest.mark.asyncio
async def test_invalid_field_name_is_rejected() -> None:
    class Base(TableModel):
        pass

    _ = type(
        "BadFieldName",
        (Base,),
        {"__annotations__": {"bad-name": str}},
    )

    async with aiosqlite.connect(":memory:") as db:
        with pytest.raises(ValueError):
            await Base.sqlite_init(db)


@pytest.mark.asyncio
async def test_invalid_unique_column_name_is_rejected() -> None:
    class Base(TableModel):
        pass

    class BadUniqueName(Base):
        __unique__ = [("bad-name",)]
        name: str

    async with aiosqlite.connect(":memory:") as db:
        with pytest.raises(ValueError):
            await Base.sqlite_init(db)


@pytest.mark.asyncio
async def test_missing_unique_column_is_rejected() -> None:
    class Base(TableModel):
        pass

    class BadUnique(Base):
        __unique__ = [("missing",)]
        name: str

    async with aiosqlite.connect(":memory:") as db:
        with pytest.raises(ValueError):
            await Base.sqlite_init(db)


@pytest.mark.asyncio
async def test_invalid_index_column_name_is_rejected() -> None:
    class Base(TableModel):
        pass

    class BadIndexName(Base):
        __indices__ = [("bad-name",)]
        name: str

    async with aiosqlite.connect(":memory:") as db:
        with pytest.raises(ValueError):
            await Base.sqlite_init(db)


@pytest.mark.asyncio
async def test_missing_index_column_is_rejected() -> None:
    class Base(TableModel):
        pass

    class BadIndex(Base):
        __indices__ = [("missing",)]
        name: str

    async with aiosqlite.connect(":memory:") as db:
        with pytest.raises(ValueError):
            await Base.sqlite_init(db)
