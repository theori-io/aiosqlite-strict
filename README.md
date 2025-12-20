# aiosqlite-strict

Strict Pydantic models on top of aiosqlite.

## Install

```bash
pip install aiosqlite-strict
```

## Usage

```python
import asyncio

import aiosqlite
from aiosqlite_strict import TableModel

class AppTable(TableModel):
    ...

class User(AppTable):
    __indices__ = [("name",)]

    name: str
    email: str

async def main() -> None:
    async with aiosqlite.connect(":memory:") as db:
        await AppTable.sqlite_init(db)

        user1 = await User.create(db, name="name1", email="email1")
        _ = await User.create(db, name="name2", email="email2")

        async with User.select(db) as cursor:
            print(await cursor.fetchall())

        await user1.update_one(db, name="name1.1")

        async with User.select(db, "WHERE name=?", ("name1.1",)) as cursor:
            print(await cursor.fetchall())

asyncio.run(main())
```

## Development

```bash
uv run pytest
```
