from flask import Flask


class TestClassifiers:
    def test_post(self, app: Flask) -> None:

        with app.test_client() as c:
            pass
