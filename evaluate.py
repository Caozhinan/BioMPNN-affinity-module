#!/usr/bin/env python  
import os  
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
  
import torch  
import pandas as pd  
import numpy as np  
from torch.utils.data import DataLoader  
from sklearn.metrics import mean_squared_error, mean_absolute_error  
from scipy.stats import pearsonr  
  
from EHIGN import DTIPredictor  
from graph_constructor import collate_fn  
from train import CustomGraphDataset  
  
# 读取测试集CSV  
test_csv = '/xcfhome/zncao02/dataset_bap/test_set/pdbbind/csv/processed_valid.csv'  
test_df = pd.read_csv(test_csv)  
  
# 调试信息  
print(f"CSV列名: {test_df.columns.tolist()}")  
print(f"测试集样本数: {len(test_df)}")  
  
# 列名标准化  
if 'pK' in test_df.columns:  
    test_df = test_df.rename(columns={'pK': 'pk'})  
  
# 验证必需的列  
required_cols = ['name', 'complex_dir']  
missing_cols = [col for col in required_cols if col not in test_df.columns]  
if missing_cols:  
    raise ValueError(f"CSV缺少必需的列: {missing_cols}。当前列: {test_df.columns.tolist()}")  
  
# 配置参数  
batch_size = 96  
graph_type = 'Graph_EHIGN_5edges'  
model_path = 'best_model.pt'  
  
# 创建数据集  
test_set = CustomGraphDataset(test_df, graph_type=graph_type)  
print(f"成功加载测试集: {len(test_set)} 个样本")  
  
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  
                        collate_fn=collate_fn, num_workers=8)  
  
# 加载模型  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
model = DTIPredictor(node_feat_size=35, edge_feat_size=17,  
                    hidden_feat_size=256, layer_num=3).to(device)  
model.load_state_dict(torch.load(model_path, map_location=device))  
model.eval()  
  
# 测试  
pred_list = []  
label_list = []  
name_list = []  
  
with torch.no_grad():  
    for data in test_loader:  
        bg, label = data  
        bg, label = bg.to(device), label.to(device)  
          
        pred_lp, pred_pl = model(bg)  
        pred = (pred_lp + pred_pl) / 2  
          
        pred_list.append(pred.detach().cpu().numpy())  
        label_list.append(label.detach().cpu().numpy())  
  
# 计算指标  
pred = np.concatenate(pred_list, axis=0)  
label = np.concatenate(label_list, axis=0)  
  
rmse = np.sqrt(mean_squared_error(label, pred))  
mae = mean_absolute_error(label, pred)  
pearson_r, _ = pearsonr(pred, label)  
  
print(f"\n测试集结果:")  
print(f"RMSE: {rmse:.4f}")  
print(f"MAE: {mae:.4f}")  
print(f"Pearson相关系数: {pearson_r:.4f}")  
  
# 保存预测结果  
results_df = pd.DataFrame({  
    'name': test_df['name'][:len(pred)],  
    'true_pK': label,  
    'pred_pK': pred,  
    'error': np.abs(label - pred)  
})  
results_df.to_csv('test_predictions.csv', index=False)  
print(f"\n预测结果已保存到 test_predictions.csv")