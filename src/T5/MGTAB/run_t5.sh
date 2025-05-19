#!/bin/bash

# 设置三个不同的种子值
SEED1=97
SEED2=815
SEED3=1149
SEED4=945371
SEED5=123456

# 确保输出目录存在
mkdir -p model
mkdir -p res
mkdir -p logs

# 运行第一个实验
echo "开始运行实验1，种子值: $SEED1"
nohup python train.py --seed $SEED1 > logs/t5_seed_${SEED1}.log 2>&1 &
echo "实验1完成"

# 运行第二个实验
echo "开始运行实验2，种子值: $SEED2"
nohup python train.py --seed $SEED2 > logs/t5_seed_${SEED2}.log 2>&1 &
echo "实验2完成"

# 运行第三个实验
echo "开始运行实验3，种子值: $SEED3"
nohup python train.py --seed $SEED3 > logs/t5_seed_${SEED3}.log 2>&1 &
echo "实验3完成"

# 运行第四个实验
echo "开始运行实验4，种子值: $SEED4"
nohup python train.py --seed $SEED4 > logs/t5_seed_${SEED4}.log 2>&1 &
echo "实验4完成"

# 运行第三个实验
echo "开始运行实验5，种子值: $SEED5"
nohup python train.py --seed $SEED5 > logs/t5_seed_${SEED5}.log 2>&1 &
echo "实验5完成"

echo "所有实验已完成，结果已保存到res目录"
echo "日志文件已保存到logs目录"
