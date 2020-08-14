
build:
		docker build -t SSAD .

run:
		docker run --runtime nvidia -it --rm \
			--workdir /app/github \
			--volume ~/github_actions:/app/github_actions \
			--name SSAD \
			--hostname SSAD \
			SSAD
