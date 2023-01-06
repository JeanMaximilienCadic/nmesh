# PHONY are targets with no files to check, all in our case
.DEFAULT_GOAL := build

# Call with "make something conf_file=file.env" to overwrite
conf_file ?= .env
-include $(conf_file)

# Ensure that we have a configuration file
$(conf_file):
	$(error Please create a '$(conf_file)' file first, for example by copying example_conf.env. No '$(conf_file)' found)

# Makefile for launching common tasks
DOCKER_OPTS ?= \
    -e DISPLAY=${DISPLAY} \
	-v /dev/shm:/dev/shm \
	-v $(HOME)/.ssh:/home/foo/.ssh \
	-v $(HOME)/.config:/home/foo/.config \
	-v $(PWD):/workspace \
	-v $(SRV):/srv \
	-v $(FILESTORE):/FileStore \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /var/run/docker.sock:/var/run/docker.sock \
	--network=host \
	--privileged

VERSION=$(shell python -c 'from $(PACKAGE) import __version__;print(__version__)')

help:
	@echo "Usage: make {build,  bash, ...}"
	@echo "Please check README.md for instructions"
	@echo ""


# BUILD:
build: build_wheels build_dockers

# BUILD DOCKER
build_dockers: build_docker_vanilla build_docker_sandbox 

build_docker_vanilla:
	docker build . -t  $(IMAGE_VANILLA) --network host -f docker/vanilla/Dockerfile

build_docker_sandbox:
	docker build . -t  $(IMAGE_SANDBOX) --network host -f docker/sandbox/Dockerfile

# BUILD WHEEL
build_wheels: build_wheel

build_wheel:
	# Build the wheels
	@mv dist/*.whl dist/legacy/ || true
	@pip wheel . -w ./dist
	@mv dist/nmesh*.whl ./ && rm dist/*.whl && mv nmesh*.whl dist/
	@clean || true

# PUSH
push_dockers: push_docker_vanilla push_docker_sandbox

push_docker_sandbox:
	@docker tag $(IMAGE_SANDBOX) $(IMAGE_SANDBOX)-$(PACKAGE)_$(VERSION)
	docker push $(IMAGE_SANDBOX)
	docker push $(IMAGE_SANDBOX)-$(PACKAGE)_$(VERSION)

push_docker_vanilla:
	@docker tag $(IMAGE_VANILLA) $(IMAGE_VANILLA)-$(PACKAGE)_$(VERSION)
	docker push $(IMAGE_VANILLA)
	docker push $(IMAGE_VANILLA)-$(PACKAGE)_$(VERSION)

# PULL
pull_dockers: pull_docker_vanilla pull_docker_sandbox

pull_docker_vanilla:
	docker pull $(IMAGE_VANILLA)

pull_docker_sandbox:
	docker pull $(IMAGE_SANDBOX)

# DOCKER RUNs
sandbox:
	@docker stop dev_$(PACKAGE) || true
	@docker rm dev_$(PACKAGE) || true
	docker run --name dev_$(PACKAGE) ${DOCKER_OPTS} -dt $(IMAGE_SANDBOX)
	docker exec -it dev_$(PACKAGE) bash

# COMMON
install_wheels:
	pip install dist/*.whl

tests:
	python -W ignore -m nmesh.tests

# ALL
all: build checkout push_dockers
all_branch: build_wheels checkout
