## Python version
This was made and tested on Python 3.7.3 and Anaconda Python 3.6.5. It should work on any Python3. Please create a GitHub issue if otherwise.

## Quick setup to get the server going

Clone this repo and `cd` into this directory.

First, setup a virtual environment and install the required packages.

	$ python -m venv openFraming
	$ pip install -r requirements.txt
    $ cd api
	$ source openFraming/bin/activate

You should be able to run the server using.
	
	$ flask run --host=0.0.0.0 --port=5000 --debugger --reload 

If you go to your browser and try the following URLS, you should get a simple JSON 
response back.

 1. http://localhost:5000/classifiers/0/progress
 2. http://localhost:5000/classifiers/1/progress
 3. http://localhost:5000/classifiers/2/progress
