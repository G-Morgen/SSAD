mkdir -p /dgx/github
mkdir -p /dgx/data
mkdir -p /dgx/shared/momo
sshfs inoue.momo:/dgx/inoue/github/ /dgx/github
sshfs inoue.momo:/dgx/inoue/data/ /dgx/data
sshfs inoue.momo:/dgx/shared/ /dgx/shared/momo
/usr/bin/zsh
