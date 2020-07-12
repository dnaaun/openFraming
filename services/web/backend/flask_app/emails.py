import logging
import typing as T

import typing_extensions as TT
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from flask_app.settings import needs_settings_init
from flask_app.settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ResponseProtocol(T.Protocol):
    @property
    def status_code(self) -> int:
        ...

    @property
    def body(self) -> str:
        ...

    @property
    def headers(self) -> T.Dict[str, str]:
        ...

    @property
    def to_dict(self) -> T.Dict[str, T.Any]:
        ...


class SengGridAPIClientProtocol(T.Protocol):
    def send(self, mail: Mail) -> ResponseProtocol:
        ...


class FakeResponse:
    @property
    def status_code(self) -> int:
        return 0

    @property
    def body(self) -> str:
        return ""

    @property
    def headers(self) -> T.Dict[str, str]:
        return {}

    @property
    def to_dict(self) -> T.Dict[str, T.Any]:
        return {}


class LogSendGridAPIClient:
    """A fake API client that just prints to console."""

    logging_string_fmt = """System generated the following email:
====
FROM: {from_}
TO: {to}
SUBJECT: {subject}
BODY: {body}
=====
"""

    def send(self, mail: Mail) -> FakeResponse:
        """Log email sent."""
        logging_string_fmt = self.logging_string_fmt
        if Settings.SENDGRID_API_KEY:
            logging_string_fmt += "SENDGRID_API_KEY was set, so this email was (most likely) actually sent."
        else:
            logging_string_fmt += (
                "SENDGRID_API_KEY was not set, so this email was not actually sent."
            )
        log_string = logging_string_fmt.format(
            from_=mail.from_email,
            to=[p.get() for p in mail.personalizations],
            subject=mail.subject.get(),
            body="==\n".join([m.get()["value"] for m in mail.contents]),
        )

        # Mock response object
        logger.info(log_string)
        return FakeResponse()


class EmailTemplate(TT.TypedDict):
    subject: str
    html_content: str


_email_templates: T.Final[T.Dict[str, EmailTemplate]] = {
    "classifier_training_finished": EmailTemplate(
        subject="[openFraming] Policy issue classifier training completed.",
        html_content=(
            """<h2>OpenFraming</h2>
Hi there!

The policy issue classifier you started training on openFraming.org has completed
training. The name you gave to this policy issue classifier was: {classifier_name}.

You will get another email when we complete <i>inference/prediction</i> on the
unlabelled dataset you uploaded.

Cheers!
"""
        ),
    ),
    "classifier_inference_finished": EmailTemplate(
        subject="[openFraming] Inference on unlabelled dataset was completed.",
        html_content=(
            """<h2>OpenFraming</h2>
Hi there!<br>

You requested to run inference on an unlabelled dataset with the following policy issue
classifier: {classifier_name}.<br>

Inference has completed! Please <a href={predictions_url}>click here</a> to download
your results.<br>

Have a great rest of your day!
"""
        ),
    ),
    "topic_model_training_finished": EmailTemplate(
        subject="[openFraming] Topic modeling completed.",
        html_content=(
            """<h2>OpenFraming</h2>
Hello!<br>

You requested to run topic modeling with your chosen topic model name of:
{topic_model_name}.<br>

Topic modeling has completed! Please <a href={topic_model_preview_url}>click here</a> to
view your topic modeling results.  On that page, you'll be able to preview the topics
discovered, and give the topic models names. You'll of course, be able to download the
results of the topic modeling.<br>

Have a great rest of your day!
"""
        ),
    ),
    "classifier_training_error": EmailTemplate(
        subject="[openFraming] Error encountered in policy issue classifier training.",
        html_content=(
            """<h2>OpenFraming</h2>
Hello,<br>

The policy issue classifier you started training on openFraming.org has encountered
an error. The name you gave to this policy issue classifier was: {classifier_name}.<br>

Unfortunately, you'll have to begin training again. If the problem persists, please
contact us by replying to this email.<br>

Cheers!
"""
        ),
    ),
    "classifier_inference_errror": EmailTemplate(
        subject="[openFraming] Error encountered while doing inference on unlabelled dataset.",
        html_content=(
            """<h2>OpenFraming</h2>
Hi there,<br>

You requested to run inference on an unlabelled dataset with the following policy issue
classifier: {classifier_name}. We ran into an error in processing your submission.<br>

Unfortunately, you'll have to begin this process again again. If the problem persists,
please contact us by replying to this email.<br>

Have a great rest of your day!
"""
        ),
    ),
    "topic_model_training_errror": EmailTemplate(
        subject="[openFraming] Error encountered in topic modeling.",
        html_content=(
            """<h2>OpenFraming</h2>
Hello,<br>

You requested to run topic modeling with your chosen topic model name of:
{topic_model_name}.<br>

We encountered an internal error in processing your submission. Please try again. If the 
problem persists, contact us by replying to this email.<br>

Have a great rest of your day!
"""
        ),
    ),
}


class Emailer:
    """Handle all email sending."""

    @needs_settings_init()
    def __init__(self) -> None:
        self._sg_clients: T.List[SengGridAPIClientProtocol] = []

        self._sg_clients.append(
            LogSendGridAPIClient()
        )  # Print to console no matter what

        if Settings.SENDGRID_API_KEY:
            self._sg_clients.append(
                SendGridAPIClient(api_key=Settings.SENDGRID_API_KEY)
            )

    @T.overload
    def send_email(
        self,
        email_template_name: T.Literal["classifier_training_finished"],
        to_email: str,
        *,
        classifier_name: str
    ) -> None:
        ...

    @T.overload
    def send_email(
        self,
        email_template_name: T.Literal["classifier_inference_finished"],
        to_email: str,
        *,
        classifier_name: str,
        predictions_url: str
    ) -> None:
        ...

    @T.overload
    def send_email(
        self,
        email_template_name: T.Literal["topic_model_training_finished"],
        to_email: str,
        *,
        topic_model_name: str,
        topic_model_preview_url: str
    ) -> None:
        ...

    def send_email(
        self,
        email_template_name: T.Literal[
            "classifier_inference_finished",
            "topic_model_training_finished",
            "classifier_training_finished",
        ],
        to_email: str,
        **template_values: str
    ) -> None:
        if email_template_name == "classifier_training_finished":
            assert "classifier_name" in template_values
        elif email_template_name == "classifier_inference_finished":
            assert "classifier_name" in template_values
            assert "predictions_url" in template_values
        elif email_template_name == "topic_model_training_finished":
            assert "topic_model_name" in template_values
            assert "topic_model_preview_url" in template_values

        template = _email_templates[email_template_name]
        html_content = template["html_content"].format(**template_values)
        message = Mail(
            from_email=Settings.SENDGRID_FROM_EMAIL or "NOSENDERSET",
            to_emails=to_email,
            subject=template["subject"],
            html_content=html_content,
        )

        for sg_client in self._sg_clients:
            try:
                sg_client.send(message)
            except Exception as e:
                logger.critical("Coudn't send email: " + str(vars(e)))
