import unittest

from flask_app.app import API_URL_PREFIX
from flask_app.utils import Json
from tests.common import AppMixin


class TestTopicModels(AppMixin, unittest.TestCase):
    def test_it(self) -> None:
        url = API_URL_PREFIX + "/topic_models/"

        with self._app.test_client() as client, self._app.app_context():
            with self.subTest("creating a topic model"):
                resp = client.post(
                    url, json={"topic_model_name": "test_topic_model", "num_topics": 2}
                )
                self._assert_response_success(resp, url)
                resp_json: Json = resp.get_json()

                assert isinstance(resp_json, dict)
                topic_model_id = resp_json.pop("topic_model_id")
                expected_topic_model_json = {
                    "topic_model_name": "test_topic_model",
                    "num_topics": 2,
                    "topic_names": None,
                    "status": "not_begun",
                }

                self.assertDictEqual(
                    resp_json, expected_topic_model_json,
                )
            with self.subTest("getting all topic models"):
                resp = client.get(url)
                self._assert_response_success(resp, url)
                resp_json = resp.get_json()
                assert isinstance(resp_json, list)
                expected_topic_model_json.update({"topic_model_id": topic_model_id})
                expected_topic_model_list_json = [expected_topic_model_json]
                self.assertListEqual(resp_json, expected_topic_model_list_json)


if __name__ == "__main__":
    unittest.main()
