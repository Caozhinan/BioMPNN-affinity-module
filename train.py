import os  
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
import pandas as pd  
import numpy as np  
from sklearn.metrics import mean_squared_error  
from scipy.stats import pearsonr  
  
from EHIGN import DTIPredictor  
from graph_constructor import collate_fn  
import dgl  
  
# 自定义 GraphDataset 类以适配你的 CSV 格式  
class CustomGraphDataset(object):  
    def __init__(self, data_df, graph_type='Graph_EHIGN_5edges'):  
        self.data_df = data_df  
        self.graph_type = graph_type  
        self.graph_paths = []  
        self._pre_process()  
      
    def _pre_process(self):  
        for i, row in self.data_df.iterrows():  
            name = row['name']  
            complex_dir = row['complex_dir']  
            graph_path = os.path.join(complex_dir, f"{self.graph_type}-{name}.dgl")  
              
            if os.path.exists(graph_path):  
                self.graph_paths.append(graph_path)  
            else:  
                print(f"Warning: {graph_path} not found, skipping...")  
      
    def __getitem__(self, idx):  
        return torch.load(self.graph_paths[idx])  
      
    def __len__(self):  
        return len(self.graph_paths)  
  
# 验证函数  
def val(model, dataloader, device):  
    model.eval()  
    pred_list = []  
    label_list = []  
      
    for data in dataloader:  
        bg, label = data  
        bg, label = bg.to(device), label.to(device)  
          
        with torch.no_grad():  
            pred_lp, pred_pl = model(bg)  
            pred = (pred_lp + pred_pl) / 2  
            pred_list.append(pred.detach().cpu().numpy())  
            label_list.append(label.detach().cpu().numpy())  
      
    pred = np.concatenate(pred_list, axis=0)  
    label = np.concatenate(label_list, axis=0)  
    pr = pearsonr(pred, label)[0]  
    rmse = np.sqrt(mean_squared_error(label, pred))  
      
    model.train()  
    return rmse, pr  
  
# 主训练流程  
if __name__ == '__main__':  
    # 配置参数  
    batch_size = 96  
    epochs = 200  
    early_stop_epoch = 30  
    learning_rate = 1e-4  
    weight_decay = 1e-6  
    graph_type = 'Graph_EHIGN_5edges'  
      
    # 读取 CSV 文件  
    train_csv = '/xcfhome/zncao02/dataset_bap/Bindingnetv2/csv/processed_train.csv'  
    valid_csv = '/xcfhome/zncao02/dataset_bap/Bindingnetv2/csv/processed_valid.csv'  
      
    train_df = pd.read_csv(train_csv)  
    valid_df = pd.read_csv(valid_csv)  
      
    print(f"训练集样本数: {len(train_df)}")  
    print(f"验证集样本数: {len(valid_df)}")  
      
    # 创建数据集  
    train_set = CustomGraphDataset(train_df, graph_type=graph_type)  
    valid_set = CustomGraphDataset(valid_df, graph_type=graph_type)  
      
    print(f"成功加载训练集: {len(train_set)} 个样本")  
    print(f"成功加载验证集: {len(valid_set)} 个样本")  
      
    # 创建 DataLoader  
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,   
                             collate_fn=collate_fn, num_workers=8)  
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,   
                             collate_fn=collate_fn, num_workers=8)  
      
    # 初始化模型  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    model = DTIPredictor(node_feat_size=35, edge_feat_size=17,   
                        hidden_feat_size=256, layer_num=3).to(device)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    criterion = nn.MSELoss()  
      
    # 训练循环  
    best_valid_rmse = float('inf')  
    patience_counter = 0  
      
    model.train()  
    for epoch in range(epochs):  
        # 训练阶段  
        epoch_loss = 0.0  
        for data in train_loader:  
            bg, label = data  
            bg, label = bg.to(device), label.to(device)  
              
            pred_lp, pred_pl = model(bg)  
            loss = (criterion(pred_lp, label) + criterion(pred_pl, label) +   
                   criterion(pred_lp, pred_pl)) / 3  
              
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
              
            epoch_loss += loss.item() * label.size(0)  
          
        epoch_loss = epoch_loss / len(train_set)  
        epoch_rmse = np.sqrt(epoch_loss)  
          
        # 验证阶段  
        valid_rmse, valid_pr = val(model, valid_loader, device)  
          
        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, train_rmse={epoch_rmse:.4f}, "  
              f"valid_rmse={valid_rmse:.4f}, valid_pr={valid_pr:.4f}")  
          
        # 保存最佳模型  
        if valid_rmse < best_valid_rmse:  
            best_valid_rmse = valid_rmse  
            patience_counter = 0  
            torch.save(model.state_dict(), 'best_model.pt')  
            print(f"保存最佳模型 (valid_rmse={valid_rmse:.4f})")  
        else:  
            patience_counter += 1  
            if patience_counter >= early_stop_epoch:  
                print(f"Early stopping at epoch {epoch}")  
                break  
      
    # 加载最佳模型并最终验证  
    model.load_state_dict(torch.load('best_model.pt'))  
    final_valid_rmse, final_valid_pr = val(model, valid_loader, device)  
    print(f"\n最终验证结果: RMSE={final_valid_rmse:.4f}, Pearson={final_valid_pr:.4f}")