python main_train.py \
        --dataset cifar100 \
        --model vgg11 \
        --epoch 200 \
        --tau 1.0 \
        --optim sgd \
        --lr 0.1 \
        --loss TGloss \
        -b 64 \
        -T 8 
