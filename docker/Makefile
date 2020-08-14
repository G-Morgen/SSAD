
build:
		docker build -t ssad:latest .

run:
		docker run --runtime nvidia -it --rm --privileged \
			--network host \
			--workdir /app/github_actions \
			--volume /home/inoue/.ssh:/tmp/.ssh:ro \
			--volume ~/github_actions:/app/github_actions \
			--name ssad \
			--hostname ssad \
			ssad:latest
