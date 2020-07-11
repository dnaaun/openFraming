import logging
import mimetypes
import unittest

from flask import current_app
from flask import request
from tests.common import AppMixin
from tests.common import TESTING_FILES_DIR
from werkzeug.exceptions import BadRequest

from flask_app import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestValidate(AppMixin):
    def test_table_has_no_empty_cells(self) -> None:
        with self.assertRaises(BadRequest):
            utils.Validate.table_has_no_empty_cells([["not empty"], [""]])
        utils.Validate.table_has_no_empty_cells([["not empty"], ["also not empty"]])

    def test_validate_spreadsheet_and_get_table(self) -> None:
        for file_path in (TESTING_FILES_DIR / "file_formats").iterdir():
            for mimetype in [mimetypes.guess_type(file_path)[0], None]:
                with self.subTest(
                    f"validate {file_path.stem} {file_path.suffix} file"
                    " with " + (" no " if mimetype is None else "") + "mimetype"
                ):
                    with current_app.test_request_context(
                        "/url123",
                        data={"file": (file_path.open("rb"), file_path.name, mimetype)},
                    ):
                        uploaded_file = request.files["file"]
                        if file_path.stem == "valid":
                            logger.info(f"about to guess for {file_path}")
                            table = utils.Validate.spreadsheet_and_get_table(
                                uploaded_file
                            )
                            self.assertListEqual(
                                table,
                                [
                                    ["Header_1", "Header_2"],
                                    ["Data_row_col_1", "Data_row_col_2"],
                                ],
                            )
                        elif file_path.stem == "invalid":
                            with self.assertRaises(BadRequest):
                                table = utils.Validate.spreadsheet_and_get_table(
                                    uploaded_file
                                )

    def tearDown(self) -> None:
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
