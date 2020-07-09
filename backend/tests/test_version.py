from flask_app.app import create_app
import typing as T
from flask_app.settings import Settings
from flask_app.version import Version
import os
from unittest import mock
import unittest
import tempfile
import logging

logger = logging.getLogger(__name__)


class TestProjectDataDirChecking(unittest.TestCase):
    def setUp(self) -> None:
        project_data_dir = tempfile.mkdtemp(prefix="project_data_")
        self._patch_cm = mock.patch.dict(
            os.environ, {"PROJECT_DATA_DIRECTORY": project_data_dir},
        )
        self._patch_cm.__enter__()
        Settings.initialize_from_env()

    def tearDown(self) -> None:
        Settings.deinitialize()
        self._patch_cm.__exit__()

    def _mock_project_data_dir_version(self, version: T.Tuple[int, int]) -> None:
        # Write incompatible_version to PROJECT_DATA_DIRECTORY
        with Version.project_data_dir_indicator_file().open("w") as f:
            f.write(".".join(map(str, version)))

    def test_incompatible_out_of_date_version(self) -> None:
        project_data_dir = tempfile.mkdtemp(prefix="project_data_")
        with mock.patch.dict(
            os.environ, {"PROJECT_DATA_DIRECTORY": project_data_dir},
        ):
            version_now = Version.PROJECT_DATA_DIR_VERSION
            incompatible_version = (version_now[0] + 1, 0)
            self._mock_project_data_dir_version(incompatible_version)

            # Check create_app fails with error
            with self.assertRaises(RuntimeError) as cm:
                create_app(logging_level=logging.DEBUG)

                self.assertIn(
                    "The version of the PROJECT_DATA_DIRECTORY structure on file",
                    str(cm.exception),
                )

    def test_compatible_out_of_date_version(self) -> None:

        version_now = Version.PROJECT_DATA_DIR_VERSION
        compatible_version = (version_now[0], version_now[1] + 1)
        self._mock_project_data_dir_version(compatible_version)

        # Doens't raise any errors
        create_app(logging_level=logging.DEBUG)
