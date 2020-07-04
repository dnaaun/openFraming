# Contributing to the backend

## Make sure you don't break the existing tests
Currently there are unit tests using `unittest` from the Python Standard Library in the
[backend/tests/](backend/tests/) directory. In addition, we have continuous integration
setup on Github. Everytime someone makes a pull request to `master`, Github
automatically starts testing the code and shows the result in the pull request. Have a
look at [this pull request](https://github.com/davidatbu/openFraming/pull/144) where
this continuous integration was setup for what that looks like.

Therefore, before making any pull requests that affect the backend, it is wise to make
sure that all the tests are passing locally on your machine. To run all the tests, you
can type the following command assuming your current working directory is `backend/`

	$ python -m unittest

If you've never done unit testing before, have a look at
[backend/tests/test_utils.py](backend/tests/test_utils.py), which has a testing for one
method from the `Validate` class in
[backend/flask_app/utils.py](backend/flask_app/utils.py). Additionally, the
[official documentation on the `unittest`
library](https://docs.python.org/3/library/unittest.html) is pretty readable. If you
will be writing code dealing with Flask, I would suggest you try to understand the
`TestClassifiers.test_get` method in
[test_classifier.py](backend/tests/test_classifier.py), and you'll have to read parts of
[Flask's documentation on testing](https://flask.palletsprojects.com/en/1.1.x/testing/).

## You're encouraged to write your own tests
If you contribute new features, you'll have to make sure that they don't break existing
functionality. Feel free to make a pull request even if the tests are not passing
locally, and we can debug together.

You're not required to write new tests for new features, but you are welcome to!
