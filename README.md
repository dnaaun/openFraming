## Requirements
### Docker
You need [Docker](https://docs.docker.com/get-docker/). Feel free to read up on Docker if you wish.
My best short explanation for Docker is that, Docker is for deploying applications with complicated
dependencies, what the printing press was to publishing books(It allows you to do it in a much quicker,
and much more reproducible way).

The link above has guides on how to install Docker on the most popular platforms.

### Administrator (sudo) rights
Interacting with Docker is much easier if you have administrator rights. 

Have no fear though, the application itself doesn't run as root, so there's much less surface area
for a security breach.

## How to install

 1. Clone this repo.
 2. `cd` into the directory where this README file is found.
 3. Type:

```
docker-compose build
```

 You might have to add `sudo` at the beginning of that command.

 4. Then type:

```
docker-compose up
```

 You might have to type `sudo` here as well. 

## Emails
If you want to send actual emails through Sendgrid with this system(as opposed to just
printing the emails that would be sent to the console),  please set the environment
variables:

```bash
export SENDGRID_API_KEY=     # An API key from Sendgrid
export SENGRID_FROM_EMAIL=   # An email address to put in the "from" field. Note that
			     # you'll have to verify this email in Sendgrid as a 
			     # "Sender". 
```

If you happen to need `sudo` in the section above, please pass the `-E` flag to make
sure these environment variables are picked up. Ie,

```bash
sudo -E docker-compose up
```
