"""A helper script to start RQ workers.

Has an option for allowing debugging (going to a pdb prompt on an Exception).

This is basically a stripped down (restricted) version of running

    rq worker

On the commandline.
"""
import typing as T
from argparse import ArgumentParser
from types import TracebackType

import ipdb  # type: ignore
from redis import Redis
from rq import Queue  # type: ignore
from rq import Worker


def ipdb_handler(job, exc_type, exc_value, traceback: TracebackType) -> None:  # type: ignore
    """Jump to an ipdb prompt for debugging workers."""
    ipdb.post_mortem(traceback)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "queue_name",
        type=str,
        choices=["classifiers", "topic_models"],
        help="Which queue the worker should address.",
    )
    parser.add_argument(
        "--debug", "-d", help="Drop to a pdb prompt on error.", action="store_true"
    )
    args = parser.parse_args()

    exception_handlers: T.Optional[
        T.List[T.Callable[[T.Any, T.Any, T.Any, TracebackType], None]]
    ] = None
    if args.debug:
        exception_handlers = [ipdb_handler]

    redis_conn = Redis()
    queue = Queue(args.queue_name, connection=redis_conn)
    worker = Worker(
        [queue], connection=redis_conn, exception_handlers=exception_handlers
    )
    worker.work()


if __name__ == "__main__":
    main()
