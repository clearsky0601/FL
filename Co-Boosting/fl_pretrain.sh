python fl_pretrain.py \
--dataset="cifar10" --partition="dir"  --beta=0.6 --seed=42 --num_users=10 \
--model="cnn" \
--local_lr=0.01 --local_ep=100 \
--sigma 0.0 \
--batch_size 128



# 解释一下这个脚本
# 这个脚本用于启动联邦学习的预训练过程。它使用了以下参数：
# --dataset="cifar10"：指定使用的训练数据集为CIFAR-10。
# --partition="dir"：数据分区方式为目录。
# --beta=0.6：设置beta参数，可能用于控制某种算法的权重。
# --seed=42：设置随机种子，以确保实验的可重复性。
# --num_users=10：指定参与训练的用户数量为10。
# --model="cnn"：选择使用的模型为卷积神经网络（CNN）。
# --local_lr=0.01：设置本地学习率为0.01。
# --local_ep=100：设置本地训练的轮数为100。
# --sigma 0.0：设置sigma参数，可能用于控制噪声或其他算法特性。
# --batch_size 128：设置每个训练批次的样本数量为128。
