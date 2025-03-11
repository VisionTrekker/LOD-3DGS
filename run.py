import os

CUDA_ID = 3

cmd = f'CUDA_VISIBLE_DEVICES={CUDA_ID} \
    python train.py \
    -s ../../Dataset/3DGS_Dataset/SmallCampus \
    -m output/small_test_trash \
    --sh_degree 2 \
    --depths depths \
    --split_mode default \
    --iterations 250000 \
    --scaling_lr 0.0015 \
    --position_lr_init 0.000016 \
    --opacity_reset_interval 250000 \
    --densify_until_iter 150000 \
    --densification_interval 10000 \
    --data_device cpu \
    --port 6060 \
    --test_iterations 5000 15000 30000 100000 150000 200000 250000 \
    --save_iterations 30000 150000 250000 \
    -r 1 \
    '
print(cmd)
os.system(cmd)

# 沿最长轴分裂
# cmd = f'CUDA_VISIBLE_DEVICES={CUDA_ID} \
#     python train.py \
#     -s ../../Dataset/3DGS_Dataset/SmallCampus \
#     -m output/small_test_maxscale \
#     --sh_degree 2 \
#     --depths depths \
#     --split_mode max_scale \
#     --iterations 250000 \
#     --scaling_lr 0.0015 \
#     --position_lr_init 0.000016 \
#     --opacity_reset_interval 250000 \
#     --densify_until_iter 150000 \
#     --densification_interval 10000 \
#     --data_device cpu \
#     --test_iterations 5000 15000 30000 100000 150000 200000 250000 \
#     --save_iterations 30000 150000 250000 \
#     -r 1 \
#     '
# print(cmd)
# os.system(cmd)