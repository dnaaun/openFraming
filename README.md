## Requirements
## Docker
You need [Docker](https://docs.docker.com/get-docker/). Feel free to read up on Docker if you wish.
My best short explanation for Docker is that, Docker is for deploying applications with complicated
dependencies, what the printing press was to publishing books(It allows you to do it in a much quicker,
and much more reproducible way).

The link above has guides on how to install Docker on the most popular platforms.

## Administrator (sudo) rights
Interacting with Docker is much easier if you have administrator rights. 

Have no fear though, the application itself doesn't run as root, so there's much less surface area
for a security breach.

# How to install

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
