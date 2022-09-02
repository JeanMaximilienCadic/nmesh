# PHONY are targets with no files to check, all in our case
.PHONY: build bash example_target check-env need_a_config jupyter tensorboard generate_requirement_file rm prune
.DEFAULT_GOAL := help

# Call with "make something conf_file=file.env" to overwrite
conf_file ?= .env
-include $(conf_file)

VERSION=$(shell python -c 'from $(CONTAINER) import __version__;print(__version__)')

# Ensure that we have a configuration file
$(conf_file):
	$(error Please create a '$(conf_file)' file first, for example by copying example_conf.env. No '$(conf_file)' found)

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
	docker tag $(IMAGE_VANILLA) $(IMAGE_VANILLA)_$(VERSION)

build_docker_sandbox:
	docker build . -t  $(IMAGE_SANDBOX) --network host -f docker/sandbox/Dockerfile
	docker tag $(IMAGE_SANDBOX) $(IMAGE_SANDBOX)_$(VERSION)

# BUILD WHEEL
build_wheels: build_wheel

build_wheel: clean
	# Build the wheels
	@mv dist/$(CONTAINER)*.whl dist/legacy/ || true; \
		python setup.py bdist_wheel && rm -r build *.egg-info; \

# PUSH
push_dockers: push_docker_vanilla push_docker_sandbox

push_docker_sandbox:
	@docker tag $(IMAGE_SANDBOX) $(IMAGE_SANDBOX)-$(CONTAINER)_$(VERSION)
	docker push $(IMAGE_SANDBOX)
	docker push $(IMAGE_SANDBOX)-$(CONTAINER)_$(VERSION)

push_docker_vanilla:
	@docker tag $(IMAGE_VANILLA) $(IMAGE_VANILLA)-$(CONTAINER)_$(VERSION)
	docker push $(IMAGE_VANILLA)
	docker push $(IMAGE_VANILLA)-$(CONTAINER)_$(VERSION)

# PULL
pull_dockers: pull_docker_vanilla pull_docker_sandbox

pull_docker_vanilla:
	docker pull $(IMAGE_VANILLA)

pull_docker_sandbox:
	docker pull $(IMAGE_SANDBOX)

# DOCKER RUNs
docker_run_sandbox_cpu:
	@docker stop dev_$(CONTAINER)_sandbox || true
	@docker rm dev_$(CONTAINER)_sandbox || true
	@docker run --restart always --name dev_$(CONTAINER)_sandbox --memory="16g" --memory-swap="32g" --shm-size 16G \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v $(HOME)/.config:/home/foo/.config \
			-v $(PWD):/workspace \
			-v $(SRV):/srv \
			-v $(FILESTORE):/FileStore \
			-dt $(IMAGE_SANDBOX)
	docker exec -it dev_$(CONTAINER)_sandbox bash

docker_run_sandbox_gpu:
	@docker stop dev_$(CONTAINER)_sandbox || true
	@docker rm dev_$(CONTAINER)_sandbox || true
	@docker run --restart always --name dev_$(CONTAINER)_sandbox --gpus all --memory="16g" --memory-swap="32g" --shm-size 16G \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v $(HOME)/.config:/home/foo/.config \
			-v $(PWD):/workspace \
			-v $(SRV):/srv \
			-v $(FILESTORE):/FileStore \
			-dt $(IMAGE_SANDBOX)
	docker exec -it dev_$(CONTAINER)_sandbox bash

# COMMON
clean:
	@rm -rf $(find . -type d -iname *__pycache__*)

checkout:
	# Update git
	@git checkout -b $(VERSION)_auto || true; \
		git add .; git commit -m $(VERSION)_auto; \
		git push origin $(VERSION)_auto

install_wheels:
	pip install dist/deps/*.whl; pip install dist/*.whl

tests:
	docker run -i  $(IMAGE_SANDBOX) python -m ai3ddst.tests

# ALL
all: build checkout push_dockers
all_branch: build_wheels checkout
