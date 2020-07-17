"""Flask CLI commands related to databases.

This requires that when this file is imported, the app_context is pushed.
"""
import typing as T

import click
import peewee as pw
from flask import Flask
from playhouse.migrate import migrate

import flask_app
from flask_app.database.migrations import BaseMigration
from flask_app.database.models import database_proxy


@click.argument(
    "migration_name",
    type=click.Choice(
        [migration.__name__ for migration in BaseMigration.__subclasses__()]
    ),
)
@click.command("run_migration")
def run_migration(migration_name: str) -> None:
    """

    Args:
        migration_name: Must be the name of a migration class in
        flask_app.database.migrations
    """
    migration_class: T.Type[BaseMigration[pw.SqliteDatabase]] = getattr(
        flask_app.database.migrations, migration_name
    )
    migration = migration_class()
    if not migration.database_needs_migrations(database_proxy.obj):  # type: ignore[arg-type]
        print("Database doesn't need the migrations.")
        return

    models = migration.get_models_to_create()
    operations = migration.make_migrate_operations(database_proxy.obj)  # type: ignore[arg-type]
    with database_proxy.atomic():
        database_proxy.create_tables(models)
        migrate(*operations)
        print("Migration done")


ALL_COMMANDS = [run_migration]


def add_commands_to_app(app: Flask) -> None:
    for cmd in ALL_COMMANDS:
        app.cli.add_command(cmd)
