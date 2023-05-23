# eventchains 事件抽取
## 项目描述
简要描述这个项目的目的和功能。
## 环境依赖
- Python 2.7
- Jdk 1.8
- Anaconda
## 安装流程
1. 克隆本仓库

`git clone https://github.com/kevinJhz/eventchains-aaai.git`

2.创建环境

`conda create -n py27 python=2.7`

`conda activate py27`

3.安装依赖

```
cd eventchains-aaai/main/

chmod +x * -R

cd ./lib

# eventchains-aaai/main/lib/python/
make

cd python

# ./eventchains-aaai/main/lib/python/
make
```

4. 编辑配置文件

```
vim main/bin/event_pipeline/config/gigaword-nyt
```

```
# Config file for event chains pipeline

# 自定义任务名, 此处为 gigaword-nyt
PIPELINE_NAME=gigaword-nyt
HUMAN_READABLE_NAME="Gigaword, NYT"

# Point to a script that will extract the input text for us
INPUT_EXTRACTOR=1-parse/preprocess/extract_gigaword.sh

# This is specific to Gigaword, used by the input extractor
# 指定原始数据存放位置，使用绝对路径。本例中数据集位于 /root/eventchains-aaai/main/data/LDC-samples/LDC2003T05.gz
GIGAWORD_FILES=(/root/eventchains-aaai/main/data/LDC-samples/*.gz)

# Number of processes to use in any stage
PROCESSES=12

```

```
vim main/bin/event_pipeline/config/local
```

```
# Change these values to local settings

# This should point to a directory on a large disk used for temporary storage
# Best if this can be read/written fast -- i.e. local disk (not networked)
# 临时目录
LOCAL_WORKING_DIR=/root/eventchains-aaai/main/chains

# This points to the directory where the final output of the process is stored
# It also needs to have plenty of space, but won't be accessed many times, so needn't be super-fast
# 最终 output 存放目录
LOCAL_FINAL_DIR=/root/eventchains-aaai/main/final-eventchains
```


pip install -r requirements.txt
3. 执行以下命令进行训练/测试/使用
bash 
# 训练模型
python train.py

# 测试模型 
python test.py

# 使用模型进行预测 
python use.py
添加开发环境的安装与配置步骤,让其他开发者可以轻松运行您的项目。
## 项目结构
- data/ - 存放数据
- models/ - 存放模型结构与权重
- train.py - 训练脚本
- test.py - 测试脚本
- use.py - 使用脚本
- requirements.txt - 环境依赖列表
简要描述项目的文件结构与每个文件的用途。
## 补充信息
- 添加项目开发计划、路线图等额外信息
- 遇到的坑与解决方案
- 参考资料与致谢
- 等等