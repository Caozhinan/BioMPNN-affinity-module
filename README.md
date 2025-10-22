# BioMPNN Pipeline 使用指南

## 概述

`run_pipeline.py` 是一个端到端的整合脚本，用于自动执行 **蛋白质-配体结合亲和力预测** 的完整流程。该脚本依次调用三个核心模块：

1. **预处理 (`batch_custom_preprocess.py`)** - 提取结合口袋并转换分子格式  
   关键函数：`custom_preprocess.py:14-32`
2. **图构建 (`graph_constructor.py`)** - 生成异构图表示  
   关键函数：`graph_constructor.py:355-377`
3. **训练/测试 (`train.py`)** - 训练或评估 EHIGN 模型  
   关键函数：`train.py:66-83`

---

## 输入格式

脚本接受包含以下列的 CSV 文件：

| 列名 | 说明 | 示例 |
|------|------|------|
| name | 复合物唯一标识符 | 2tpi |
| receptor | 蛋白质 PDB 文件路径 | /path/to/2tpi_protein.pdb |
| ligand | 配体文件路径 (PDB/MOL2/SDF) | /path/to/2tpi_ligand.sdf |
| pk | 结合亲和力值 (-log10(Kd)) | 4.31 |

### CSV 示例

```csv
name,receptor,ligand,pk
2tpi,/xcfhome/zncao02/dataset_bap/PDBBind/pro-lig_all/2tpi/2tpi_protein.pdb,/xcfhome/zncao02/dataset_bap/PDBBind/pro-lig_all/2tpi/2tpi_ligand.sdf,4.31


# 基本用法

## 1. 完整训练流程

从原始数据到训练完成的一键式执行：

```bash
python run_pipeline.py --data_csv data.csv --mode train
```

这将依次执行：

- 提取 5Å 结合口袋  
- 生成异构图 (.dgl 文件)  
- 自动 9:1 分割训练/验证集  
- 训练 EHIGN 模型  

### 2. 测试模式

使用全部数据进行评估：

```bash
python run_pipeline.py --data_csv test_data.csv --mode test
```

### 3. 仅数据准备

只执行预处理和图构建，不训练：

```bash
python run_pipeline.py --data_csv data.csv --skip_train
```

---

## 常用命令示例

### 自定义超参数训练

```bash
python run_pipeline.py \
    --data_csv data.csv \
    --mode train \
    --epochs 300 \
    --batch_size 128 \
    --learning_rate 5e-5 \
    --hidden_feat_size 512
```

### 调整口袋提取距离

```bash
python run_pipeline.py \
    --data_csv data.csv \
    --distance 6.0 \
    --mode train
```

### 多 GPU 训练

```bash
python run_pipeline.py \
    --data_csv data.csv \
    --mode train \
    --cuda_device "0,1"
```

### 加速并行处理

```bash
python run_pipeline.py \
    --data_csv data.csv \
    --preprocess_n_jobs 16 \
    --graph_num_process 32 \
    --mode train
```

### 增量执行

如果某些步骤已完成，可以跳过：

#### 跳过预处理（已有 .rdkit 文件）

```bash
python run_pipeline.py \
    --data_csv data.csv \
    --skip_preprocess \
    --mode train
```

#### 跳过图构建（已有 .dgl 文件）

```bash
python run_pipeline.py \
    --data_csv data.csv \
    --skip_preprocess \
    --skip_graph \
    --mode train
```

#### 只做预处理

```bash
python run_pipeline.py \
    --data_csv data.csv \
    --skip_graph \
    --skip_train
```

---

## 主要参数说明

### 流程控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_csv` | 必需 | 输入 CSV 文件路径 |
| `--mode` | train | 运行模式：train 或 test |
| `--skip_preprocess` | False | 跳过预处理步骤 |
| `--skip_graph` | False | 跳过图构建步骤 |
| `--skip_train` | False | 跳过训练/测试步骤 |

### 预处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--distance` | 5.0 | 口袋提取距离 (Å) |
| `--preprocess_n_jobs` | 4 | 并行进程数 |
| `--preprocess_timeout` | 300 | 单个复合物超时时间 (秒) |

### 图构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--graph_type` | Graph_EHIGN_5edges | 图类型前缀 |
| `--dis_threshold` | 5.0 | 相互作用距离阈值 (Å) |
| `--graph_num_process` | 28 | 图构建并行进程数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 96 | 训练批次大小 |
| `--epochs` | 200 | 最大训练轮数 |
| `--early_stop_epoch` | 30 | 早停耐心值 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--train_ratio` | 0.9 | 训练集比例 (仅训练模式) |
| `--random_seed` | 42 | 随机种子 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--node_feat_size` | 35 | 节点特征维度 |
| `--edge_feat_size` | 17 | 边特征维度 |
| `--hidden_feat_size` | 256 | 隐藏层维度 |
| `--layer_num` | 3 | GNN 层数 |

---

## 输出文件

执行完成后，将在相应目录生成：

- **预处理输出：** `{complex_dir}/{name}.rdkit` → RDKit 分子对象  
- **图构建输出：** `{complex_dir}/{graph_type}-{name}.dgl` → DGL 异构图  
- **训练输出：** `best_model.pt` → 最佳模型权重文件  

---

## 查看完整参数列表

```bash
python run_pipeline.py --help
```

---

## 工作流程

1. 输入 CSV  
2. 预处理  
3. 图构建  
4. 训练/测试  
5. 输出模型  

### 流程说明

- 验证 CSV 格式：检查必需列是否存在  
- 预处理：提取口袋、转换格式、生成 `.rdkit` 文件  
- 图构建：构建 5 种边类型的异构图，保存为 `.dgl` 文件  
- 训练/测试：训练模式自动 9:1 分割数据集，测试模式使用全部数据  

---

## 注意事项

- 确保输入文件路径正确且文件存在  
- 预处理需要 **PyMOL** 和 **OpenBabel** 环境  
- 训练需要 GPU 支持（通过 `--cuda_device` 指定）  
- 中间文件（`.rdkit`、`.dgl`）会保存在原始数据目录中  

---

## Notes

该 README 基于完整的 `run_pipeline.py` 脚本编写，涵盖了所有主要功能和参数。  
脚本支持灵活的流程控制，可以根据需要跳过已完成的步骤，适合批量数据处理和模型训练场景。