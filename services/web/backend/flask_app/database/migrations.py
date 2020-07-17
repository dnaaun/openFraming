import abc
import typing as T

import peewee as pw
from playhouse.migrate import Operation
from playhouse.migrate import SqliteMigrator
from playhouse.reflection import Introspector

from flask_app.database import models


_DatabaseSub = T.TypeVar("_DatabaseSub", bound=T.Union[pw.DatabaseProxy, pw.Database])


class BaseMigration(abc.ABC, T.Generic[_DatabaseSub]):
    """Make it easier to run database migrations."""

    @abc.abstractmethod
    def database_needs_migrations(self, db: _DatabaseSub) -> bool:
        """Check if the migration has already been run on the datbase.

        This is a very rough function since we don't thoroughly check that the expected
        models correspond exactly to what is in the db. Just for the columns / tables 
        we are interested in.

        Raises:
            RuntimeError: If we find that the database is in neither of the two states
                we expect to be in(ie, before migration, or after migration).

                For example, if the migration is to rename  a table, but no table with 
                the name before migration, or after migration, exists, then we raise
                runtime error.
        """
        pass

    @abc.abstractmethod
    def get_models_to_create(self) -> T.List[T.Type[pw.Model]]:
        pass

    @abc.abstractmethod
    def make_migrate_operations(self, db: _DatabaseSub) -> T.List[Operation]:
        pass


class AddTopicModelMetricsMigration(BaseMigration[_DatabaseSub]):
    def database_needs_migrations(self, db: pw.Database) -> bool:
        models_in_db = Introspector.from_database(db).generate_models()
        if "ldaset" not in models_in_db:
            raise RuntimeError(
                "Did not find models named  'ldaset' in models_in_db, but found"
                + str(models_in_db)
            )
        print("Found models in db:", models_in_db)
        topicmodelmetrics_found = "topicmodelmetrics" in models_in_db
        metrics_id_found = "metrics_id" in models_in_db["ldaset"]._meta.columns

        if topicmodelmetrics_found != metrics_id_found:
            raise RuntimeError(
                f"Inconsistent status:"
                f" topicmodelmetrics_found={topicmodelmetrics_found} and metrics_col_found={metrics_id_found}"
            )
        return not topicmodelmetrics_found

    def get_models_to_create(self) -> T.List[T.Type[pw.Model]]:
        return [models.TopicModelMetrics]

    def make_migrate_operations(self, db: _DatabaseSub) -> T.List[Operation]:
        migrator = SqliteMigrator(T.cast(pw.SqliteDatabase, db))

        ops = [
            migrator.add_column(
                "ldaset", "metrics_id", models.LDASet._meta.columns["metrics_id"]
            )
        ]
        return ops


class RenameClassifierMetricsTableMigration(BaseMigration[_DatabaseSub]):
    """This should actually have been part of AddTopicModelMetricsMigration, but I 
       forgot to do this.

       It's a lot of work to test migrations properly when there's no easy way to
       setup specific versions of the ORM, I think(since I haven't done it, I can only
       speculate).
    """

    def database_needs_migrations(self, db: _DatabaseSub) -> bool:
        models_in_db = Introspector.from_database(db).generate_models()
        metrics_found = "metrics" in models_in_db
        classifiermetrics_not_found = "classifiermetrics" not in models_in_db

        if metrics_found != classifiermetrics_not_found:
            raise RuntimeError(
                f"Inconsistent status:"
                f" classifiermetrics_not_found={classifiermetrics_not_found} and metrics_found={metrics_found}"
            )
        return metrics_found

    def get_models_to_create(self) -> T.List[T.Type[pw.Model]]:
        return []

    def make_migrate_operations(self, db: _DatabaseSub) -> T.List[Operation]:
        migrator = SqliteMigrator(db)
        ops = [migrator.rename_table("metrics", "classifiermetrics")]
        return ops
