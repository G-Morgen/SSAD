cp /tmp/.ssh /root/.ssh -r
mkdir -p /dgx/github
mkdir -p /dgx/data
mkdir -p /dgx/shared/momo
sshfs inoue.momo:/dgx/inoue/github/ /dgx/github
sshfs inoue.momo:/dgx/inoue/data/ /dgx/data
sshfs inoue.momo:/dgx/shared/ /dgx/shared/momo
cd /app/github_actions/DVC
dvc pull data/202008131442_H.zip.dvc
dvc repro
cd /app/github_actions/SSAD/ssad
python run.py conf/config.yaml
