python /dgx/github/SSAD/ssad/run.py \
    /dgx/github/SSAD/ssad/conf/config.yaml \
    run_train.epoch=50 \
    augs.height=384 \
    augs.width=384 \

python /dgx/github/SSAD/ssad/run.py \
    /dgx/github/SSAD/ssad/conf/config.yaml \
    run_train.epoch=100 \
    augs.height=384 \
    augs.width=384 \

python /dgx/github/SSAD/ssad/run.py \
    /dgx/github/SSAD/ssad/conf/config.yaml \
    run_train.epoch=200 \
    augs.height=384 \
    augs.width=384 \

python /dgx/github/SSAD/ssad/run.py \
    /dgx/github/SSAD/ssad/conf/config.yaml \
    run_train.epoch=400 \
    augs.height=384 \
    augs.width=384 \
