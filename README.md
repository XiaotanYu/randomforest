# Randomforest for Drug Design
<img src="https://img.shields.io/badge/license-GNU-blue.svg"/><img src="https://img.shields.io/badge/python-3.7-green"/><img src="https://img.shields.io/badge/pandas-1.3.5-green"/><img src="https://img.shields.io/badge/scikit--learn-1.0.2-green"/><img src="https://img.shields.io/badge/scikit--learn-1.0.2-green"/><img src="https://img.shields.io/badge/deepchem-2.6.1-green"/><img src="https://img.shields.io/badge/rdkit-2020.09-green"/>

## 1.开源定位
- 实现经典的randomforest模型在药物活性预测中的应用，可以输出模型评估、决策树、特征重要性等可视化的结果，用于论文发表。
- 提供示例文件，傻瓜操作，会看代码就会用。降低药学/化学工作者将机器学习用于自己生产/科研环节所需的时间成本、精力成本。
- 目前GitHub暂无相关开源项目，鲜有可借鉴的经验，如有疏漏不妥之处，还请不吝赐教！

## 2.用前注意事项

### 环境配置
- 使用前必须按照上述Tag将环境配置齐全。
- **如果先安装rdkit，再安装deepchem，会发生冲突，环境修复非常麻烦！强烈建议单独构建python 3.7虚拟环境，然后先安装deepchem，再安装rdkit。**

### 参数设置
- 所有您需要改的地方，都已经加上了注释。请务必注意原始数据存放的位置和程序中的路径是对应的。

## 3.输出结果示例
![cluster](https://user-images.githubusercontent.com/112002049/191893954-93c1f89d-d430-4d9b-9b5a-6241ab5306b3.jpg)
![boxplot](https://user-images.githubusercontent.com/112002049/191893968-3c808693-5cf5-45da-9d06-0d0a7c7ddae1.jpg)
![var_importance2](https://user-images.githubusercontent.com/112002049/191893982-5aa0e3ae-bbe4-4d86-80b0-724090a0c76a.jpg)
