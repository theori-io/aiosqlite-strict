import asyncio

from aiosqlite_strict import TableModel
import aiosqlite

class AppTable(TableModel):
    ...

class User(AppTable):
    __indices__ = [("name",)]

    name: str
    email: str

async def main():
    async with aiosqlite.connect(":memory:") as db:
        await AppTable.sqlite_init(db)

        user1 = await User.create(db, name="name1", email="email1")
        user2 = await User.create(db, name="name2", email="email2")

        async with User.select(db) as cursor:
            print(await cursor.fetchall())

        await user1.update_one(db, name="name1.1")

        async with User.select(db, "WHERE name=?", ("name1.1",)) as cursor:
            print(await cursor.fetchall())

asyncio.run(main())
