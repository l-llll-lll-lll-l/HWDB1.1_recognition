### 运行程序指南

1. **下载数据集**  
   首先，前往官网下载 `HWDB1.1tst_gnt` 和 `HWDB1.1trn_gnt` 数据集，参考网上教程解压并移动（使用网上的gnt2png脚本），使项目根目录中拥有 `test/` 和 `train/` 两个文件夹。
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
   切到项目根目录下，使用 `dataset_gnt_tools.py` 生成 TFRecords 训练数据文件。调用格式如下：
   
   - 生成全部数据：  
     ```bash
     python .\code\dataset_gnt_tools.py {test,train} --all
     ```
   - 部分生成：  
     ```bash
     python .\code\dataset_gnt_tools.py {test,train} --num 5
     ```
   > 注意：使用全部数据生成数据集需要磁盘有 **60G** 剩余空间，且内存可用空间多于 **50G**。进度条一开始卡在0的时间可能1~10分钟不等，请耐心等待或选择较小的num数先进行实验。

3. **训练模型**  
   在生成两个数据集后，可以运行 `train.py` 进行训练。最后，将权重文件移到 `src` 目录下，使用 `varify.py` 进行验证。
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