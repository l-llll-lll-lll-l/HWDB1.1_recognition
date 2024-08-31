### 运行程序指南

1. **下载数据集**  
   首先，前往官网下载 `HWDB1.1tst_gnt` 和 `HWDB1.1trn_gnt` 数据集，参考网上教程解压成.gnt文件，运行`code/gnt2png.py`（需要先编辑gnt2png脚本，设置数据集路径）得到 `.data/test/` 和 `.data/train/` 两个文件夹，将它们从`data/`移动到项目根目录，最终文件结构如下。
   ```bash
   目录结构：
    ├─code
    ├─report
    ├─src
    ├─test
    │  ├─一
    │  ├─丁
    │  ├─七
    │  ├─万
    │  ├─丈
    │  ├─三
    ...
    ├─train
    │  ├─一
    │  ├─丁
    │  ├─七
    │  ├─万
    │  ├─丈
    │  ├─三
    ...
   ```
  

2. **生成 TFRecords 训练数据文件**  
   首先使用`cd`命令,切到项目根目录下，然后使用 `dataset_gnt_tools.py` 生成 TFRecords 训练数据文件。注意下面给出的设备要求，建议先考虑仅生成部分数据。调用格式如下：
   - 生成全部数据：  
     ```bash
     python .\code\dataset_gnt_tools.py test --all
     python .\code\dataset_gnt_tools.py train --all
     ```
   - 部分生成（只拿取每类前num个样本）：  
     ```bash
     python .\code\dataset_gnt_tools.py test --num 2
     python .\code\dataset_gnt_tools.py train --num 8
     ```
   > 注意：使用全部数据生成数据集需要磁盘有 **60G** 剩余空间，且内存可用空间多于 **50G**。进度条一开始卡在0的时间可能1~10分钟不等，请耐心等待或选择较小的num数先进行实验。

3. **训练模型**  
   在生成两个数据集后，可以运行 `train.py` 进行训练。最后，将权重文件移到 `src` 目录下，使用 `varify.py` 进行验证。注意，验证需要使用前面已经生成的`test.tfrecords`测试数据集。`report/`中已存在的验证结果使用的是全部测试数据。
   - 进行训练：  
     ```bash
     python .\code\train.py
     ```
   - 验证结果（filename需要换成想要验证的weight文件名）：  
     ```bash
     python .\code\varify.py --model filename
     ```
### 运行结果
  - 本模型基于在HWDB1.0上已训练完成的模型进行迁移学习获得，原模型参见https://github.com/angzhou/anchor
  - 目前在HWDB1.1测试集数据上已经达到96.60%的准确率，但这应该不是这个模型的极限
  - 详细训练报告参见report/（重新运行report.ipynb需要依靠code里面的the_model和utils支持）