## Required Python version
Python 3.5 or higher is required (due to use of PEP 526 style annotations).

## Installation

### Clone this repo and prepare a virtual environment.
	$ git clone https://github.com/davidatbu/openFraming.git
	$ cd openFraming/
	$ python -m venv venv/
	$ source venv/bin/activate

### Install necessary Python packages
	$ pip install -r backend/requirements_no_gpu.txt

### Download additional requirements

### NLTK Corpora
	$ python -m nltk.downloader stopwords wordnet

### Mallet
The installation of the Mallet library will depend on your platform. Have a look 
at the [installation instructions on their website.](http://mallet.cs.umass.edu/download.php). 
On Unix, it would look something like:

	$ wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz -O $HOME/mallet-2.0.8.tar.gz
	# mkdir $HOME/mallet
	$ tar -xvf ~/mallet-2.0.8.tar.gz --one-top-level=$HOME/mallet

One then has to export the directory where the mallet executable is found as an environment variable.
(This also depends on your platform).

	$ export MALLET_BIN_DIRECTORY=$HOME/mallet/mallet-2.0.8/bin

## Command to run

Running the development server should then be possible with:

	$ cd backend/flask_app/
	$ flask run --host=0.0.0.0 --port=5000 --debugger --reload 

You should be able to interact with the API endpoints documented at [the README in the `/backend` directory](backend/README.md).
