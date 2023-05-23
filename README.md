# eventchains 事件抽取
## 项目描述
简要描述这个项目的目的和功能。
## 环境依赖
- Python 2.7
- Jdk 1.8
- Anaconda
## 安装流程
###  1. 克隆本仓库

`git clone https://github.com/kevinJhz/eventchains-aaai.git`


### 2. 创建环境

`conda create -n py27 python=2.7`

`conda activate py27`

### 3. 安装依赖包、下载模型

```
cd eventchains-aaai/main/

chmod +x * -R

cd ./lib

# main/lib/
make

cd python

# main/lib/python/
make

cd ../../models

# main/models/
make
```
### 4. 上传数据
#### 4.1 原始数据
原始数据一般是若干 .txt 结尾的文本语料，打包成一个 .gz 压缩文件。

一个语料库示例是
 [LDC2003T05.gz](https://catalog.ldc.upenn.edu/desc/addenda/LDC2003T05.gz)
，由 5973 篇新闻组成。

我们通常在`main/data/`下创建一个子目录，用于放置原始数据(如 `main/data/LDC-samples/LDC2003T05.gz`，`LDC-samples`是我们创建的目录)。

在配置文件中指定该子目录路径，见 [5.1](#5.1)

#### 4.2 黑名单
在若干篇新闻组成的语料库中，难免有一些主题无关的篇章，手动删除它们很费力，更方便的做法是将要去除的文档名写入一个黑名单，脚本根据该黑名单过滤掉对应的文档。

反之，如果没有要过滤的文档，只需放一个名为 `gagaword_duplicates.gz` 的空文件。

**黑名单的默认路径**是 `main/bin/event_pipeline/1-parse/preprocess/gigaword/gigaword_duplicates.gz`，在启动 stage 1 确保该路径下存在对应的`gigaword_duplicates.gz`文件。

也可以在[extract_gigaword.sh](https://github.com/kevinJhz/eventchains-aaai/blob/main/main/bin/event_pipeline/1-parse/preprocess/extract_gigaword.sh)
的第 12 行修改黑名单的路径。

### 5. 一些需要手动配置的地方

#### 5.1 gigaword-nyt 指定原始数据路径
<a name="5.1"></a>
```
# eventchains-aaai/main/
vim main/bin/event_pipeline/config/gigaword-nyt
```
[main/bin/event_pipeline/config/gigaword-nyt](https://github.com/kevinJhz/eventchains-aaai/blob/main/main/bin/event_pipeline/config/gigaword-nyt) 内容如下:
```gigaword-nyt
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

#### 5.2 local 指定 临时文件存放路径 和 最终输出路径

```
# eventchains-aaai/main/
# 编辑 local 文件 
vim main/bin/event_pipeline/config/local
```
[main/bin/event_pipeline/config/local](https://github.com/kevinJhz/eventchains-aaai/blob/main/main/bin/event_pipeline/config/local) 内容如下:

```local
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
#### 5.3 修改 StreamEntitiesExtractor.java 中的路径


该文件在 stage 5 中被执行 , 将前几个阶段的输出作为输入，因此需要指定对应的路径。

这些路径原本是通过脚本的参数传入，但测试时存在一些问题。**因此我们选择手动配置 projectPath 和 baseFileName 两个路径。**

```
# eventchains-aaai/main/
# 编辑 StreamEntitiesExtractor.java 
vim ./src/main/java/cam/whim/coreference/StreamEntitiesExtractor.java
```
[StreamEntitiesExtractor.java](https://github.com/kevinJhz/eventchains-aaai/blob/main/main/src/main/java/cam/whim/coreference/StreamEntitiesExtractor.java) 内容如下:

```
// StreamEntitiesExtractor.java

[1]     package cam.whim.coreference;
        ......
[82]    String projectPath = "/root/eventchains-aaai";
[83]    String baseFileName="LDC2003T05";
        ......
```

**projectPath** 项目主路径， `/root/eventchains-aaai`是实验时的配置。

**baseFileName**  原始数据的文件名。如 `data/LDC-samples/LDC2003T05.gz` 对应的 `baseFileName` 是 `LDC2003T05`

#### 5.4 修改 Tokenize.java 中的路径

该文件在 stage 2 中被执行，**需要手动配置 projectPath 项目主路径。**

```
# eventchains-aaai/main/
# 编辑 StreamEntitiesExtractor.java 
vim ./src/main/java/cam/whim/opennlp/Tokenize.java
```

[Tokenize.java](https://github.com/kevinJhz/eventchains-aaai/blob/main/main/src/main/java/cam/whim/opennlp/Tokenize.java) 内容如下:

```
// Tokenize.java

[1]     package cam.whim.opennlp;
        ......
[44]    String projectPath = "/root/eventchains-aaai";
        ......
```

**projectPath** 项目主路径， `/root/eventchains-aaai`是实验时的配置。



### 3. 执行以下命令进行训练/测试/使用

脚本的启动命令如下：
```
# eventchains-aaai/bin/
./event_pipeline/pipeline.sh [PIPELINE_NAME] [number]
```
其中 [ PIPELINE_NAME ] 是任务名称，对应 config 文件名

[number] 取值为 1 ~ 5，对应 5 个子任务。

```angular2html
cd ./bin

# 切换到 python 2.7
conda activate py27

# eventchains-aaai/bin/
# 阶段1: 文档提取
./event_pipeline/pipeline.sh gigaword-nyt 1

# 阶段2: 分词
./event_pipeline/pipeline.sh gigaword-nyt 2

# 阶段3: 语法分析
./event_pipeline/pipeline.sh gigaword-nyt 3

# 阶段4: 依存关系
./event_pipeline/pipeline.sh gigaword-nyt 4

# 阶段5: 共指消解
./event_pipeline/pipeline.sh gigaword-nyt 5
```

## 项目结构
```angular2html
.
├── main
│   ├── bin                 - 脚本文件
│   ├── chains              - 临时存放中间产物
│   ├── data                - 原始数据存放处
│   ├── lib                 - 第三方依赖文件
│   ├── models              - 模型文件
│   ├── src                 - java 和 python源代码
│   ├── out                 - src/main/java 编译后的字节码
│   ├── final-eventchains   - 存放各阶段的输出
│   └── README.md
├── LICENSE
└── README.md
```

## 补充信息
- 添加项目开发计划、路线图等额外信息
- 遇到的坑与解决方案
- 参考资料与致谢
- 等等