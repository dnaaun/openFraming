import unittest

from werkzeug.exceptions import BadRequest

from flask_app import utils


class TestValidate(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_table_has_no_empty_cells(self) -> None:
        with self.assertRaises(BadRequest):
            utils.Validate.table_has_no_empty_cells([["not empty"], [""]])
        utils.Validate.table_has_no_empty_cells([["not empty"], ["also not empty"]])

    def tearDown(self) -> None:
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
