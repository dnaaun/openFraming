import typing as T
import unittest

from flask import url_for
from tests.common import AppMixin

from flask_app import emails

if T.TYPE_CHECKING:
    _MixinBase = unittest.TestCase
else:
    _MixinBase = object


class TestEmails(AppMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._emailer = emails.Emailer()

    def test_sending_email(self) -> None:
        self._emailer.send_email(
            email_template_name="classifier_inference_finished",
            to_email="davidat@bu.edu",
            classifier_name="test_email.py_Classifier",
            predictions_url=url_for(
                "classifierstestsetspredictions", classifier_id=0, test_set_id=0
            ),
        )
        self._emailer.send_email(
            email_template_name="classifier_training_finished",
            to_email="davidat@bu.edu",
            classifier_name="test_email.py_Classifier",
        )
        self._emailer.send_email(
            email_template_name="topic_model_training_finished",
            to_email="davidat@bu.edu",
            topic_model_name="test_email.py_TopicModel",
            topic_model_preview_url="http://www.openframing.org/DOESNTEXISTYET",
        )

    def tearDown(self) -> None:
        super().tearDown()
