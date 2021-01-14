import typing as T
import unittest

from flask import url_for
from tests.common import AppSetup

import flask_app
from flask_app import emails


class TestEmails(AppSetup, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._emailer = emails.Emailer()

    def test_sending_email(self) -> None:
        try:
            self._emailer.send_email(
                email_template_name="classifier_inference_finished",
                to_email="davidat@bu.edu",
                classifier_name="test_email.py_Classifier",
                predictions_url=url_for(
                    flask_app.app.ClassifiersTestSetsPredictions.__name__,
                    classifier_id=0,
                    test_set_id=0,
                ),
            )
            self._emailer.send_email(
                email_template_name="classifier_training_finished",
                to_email="davidat@bu.edu",
                classifier_name="test_email.py_Classifier",
                metrics={"classifier_metric_1": 0.4},
            )
            self._emailer.send_email(
                email_template_name="topic_model_training_finished",
                to_email="davidat@bu.edu",
                topic_model_name="test_email.py_TopicModel",
                topic_model_preview_url="http://www.openframing.org/DOESNTEXISTYET",
                metrics={"topic_model_metric_1": 0.9},
            )
        except Exception as e:
            print(vars(e))
            raise (e)

    def tearDown(self) -> None:
        super().tearDown()
