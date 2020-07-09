from flask_app.settings import Settings
from pathlib import Path
import typing as T


class Version:
    # This is version of the code base(more accurately, the JSON API).
    # This is Semantically Versioned(look up semantic versioning!).
    # Ie, it's in MAJOR.MINOR format, changes in MAJOR version indicate backward
    # incompatible changes in the JSON API.
    VERSION = (0, 1)  # Ie, 0.1.

    # Used to check if the PROJECT_DATA_DIRECTORY contains a file system
    # that is incompatible with with the current version of the application.
    # (ie, the file / directory structure the code assumes is present changed.

    # This is also semanticallyversioned. Changes in MAJOR mean the project
    # data directory structure is incompatible with previous version.
    PROJECT_DATA_DIR_VERSION = (0, 1)

    @classmethod
    def project_data_dir_indicator_file(cls) -> Path:
        return Settings.PROJECT_DATA_DIRECTORY / "project_data_dir_version.txt"

    @classmethod
    def get_project_data_dir_version_on_disk(cls) -> T.Optional[T.Tuple[int, int]]:
        """Will be NONE if there's no indicator file on disk."""

        if not cls.project_data_dir_indicator_file().exists():
            return None
        else:
            with cls.project_data_dir_indicator_file().open() as f:
                cntents = f.read()
            major, minor = map(int, cntents.split("."))
            return (major, minor)

    @classmethod
    def ensure_project_data_dir_version_safe(cls) -> None:
        """Checks if the projec_data_dir version on disk is compatible with the code.

        Raises:
            RuntimeError: If project_data_dir_version on file is incompatible.

        Writes to:
            cls.project_data_dir_indicator_file(): The project data dir version of the 
            code.
        """
        on_disk = cls.get_project_data_dir_version_on_disk()
        if on_disk is not None and cls.versions_incompatible(
            on_disk, cls.PROJECT_DATA_DIR_VERSION
        ):
            raise RuntimeError(
                "The version of the PROJECT_DATA_DIRECTORY structure "
                f"on file(stored at {str(cls.project_data_dir_indicator_file())}) "
                "which is {on_disk}, "
                "is incompatible with the version that the version we expected: "
                "{cls.PROJECT_DATA_DIR_VERSION}."
            )

        with cls.project_data_dir_indicator_file().open("w") as f:
            f.write(".".join(map(str, cls.PROJECT_DATA_DIR_VERSION)))

    # Note it's a static method, so it's independent of the rest of the class
    @staticmethod
    def versions_incompatible(ver1: T.Tuple[int, int], ver2: T.Tuple[int, int]) -> bool:
        return ver1[0] != ver2[0]
