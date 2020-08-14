
build:
		docker build -t ssad:latest .

run:
		docker run --runtime nvidia --rm --privileged \
			--network host \
			--workdir /app/github_actions \
			--volume ~/github_actions:/app/github_actions \
			--name ssad \
			--hostname ssad \
			ssad:latest
