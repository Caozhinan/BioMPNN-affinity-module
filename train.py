import os  
import argparse  
import logging  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
import pandas as pd  
import numpy as np  
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split  
from scipy.stats import pearsonr  
  
from EHIGN import DTIPredictor  
from graph_constructor import collate_fn  
import dgl  
  
class CustomGraphDataset(object):  
    def __init__(self, data_df, graph_type='Graph_EHIGN_5edges'):  
        self.data_df = data_df  
        self.graph_type = graph_type  
        self.graph_paths = []  
        self.valid_indices = []  
        self._pre_process()  
      
    def _pre_process(self):  
        """预处理：检查 DGL 文件是否存在，跳过缺失的样本"""  
        total_samples = len(self.data_df)  
        skipped_samples = []  
          
        for i, row in self.data_df.iterrows():  
            name = row['name']  
            receptor_path = row['receptor']  
            complex_dir = os.path.dirname(receptor_path)  
            graph_path = os.path.join(complex_dir, f"{self.graph_type}-{name}.dgl")  
              
            if os.path.exists(graph_path):  
                self.graph_paths.append(graph_path)  
                self.valid_indices.append(i)  
            else:  
                skipped_samples.append({  
                    'name': name,  
                    'directory': complex_dir,  
                    'expected_file': f"{self.graph_type}-{name}.dgl"  
                })  
                logging.warning(f"⚠️  DGL 文件缺失 - 目录: {complex_dir}, 样本: {name}")  
          
        logging.info(f"✅ 数据集加载完成:")  
        logging.info(f"   - 总样本数: {total_samples}")  
        logging.info(f"   - 成功加载: {len(self.graph_paths)}")  
        logging.info(f"   - 跳过样本: {len(skipped_samples)}")  
          
        if skipped_samples:  
            logging.warning(f"\n⚠️  以下 {len(skipped_samples)} 个样本因 DGL 文件缺失被跳过:")  
            for sample in skipped_samples[:5]:  
                logging.warning(f"   - {sample['name']}: {sample['directory']}/{sample['expected_file']}")  
            if len(skipped_samples) > 5:  
                logging.warning(f"   ... 还有 {len(skipped_samples) - 5} 个样本被跳过")  
          
        if len(self.graph_paths) == 0:  
            raise ValueError("❌ 错误: 没有找到任何有效的 DGL 文件！请检查图构建步骤是否完成。")  
      
    def __getitem__(self, idx):  
        return torch.load(self.graph_paths[idx])  
      
    def __len__(self):  
        return len(self.graph_paths)  
  
def val(model, dataloader, device, return_predictions=False):  
    """  
    验证函数  
      
    Args:  
        model: 模型  
        dataloader: 数据加载器  
        device: 设备  
        return_predictions: 是否返回详细预测结果  
      
    Returns:  
        如果 return_predictions=False: (rmse, pr)  
        如果 return_predictions=True: (rmse, pr, predictions, labels)  
    """  
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
      
    if return_predictions:  
        return rmse, pr, pred, label  
    else:  
        return rmse, pr  
  
def main():  
    parser = argparse.ArgumentParser(description="Train/Test DTI binding affinity prediction model")  
      
    # 数据相关参数  
    parser.add_argument('--data_csv', type=str, required=True,  
                       help="Path to input CSV file with columns: name,receptor,ligand,pk")  
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',  
                       help="Mode: train (auto-split 9:1) or test (load model and evaluate)")  
    parser.add_argument('--train_ratio', type=float, default=0.9,  
                       help="Training set ratio for auto-split (default: 0.9)")  
    parser.add_argument('--random_seed', type=int, default=42,  
                       help="Random seed for train/valid split")  
    parser.add_argument('--graph_type', type=str, default='Graph_EHIGN_5edges',  
                       help="Graph type for loading DGL files")  
      
    # 训练超参数  
    parser.add_argument('--batch_size', type=int, default=96,  
                       help="Batch size for training")  
    parser.add_argument('--epochs', type=int, default=200,  
                       help="Maximum number of training epochs")  
    parser.add_argument('--early_stop_epoch', type=int, default=30,  
                       help="Number of epochs for early stopping patience")  
    parser.add_argument('--learning_rate', type=float, default=1e-4,  
                       help="Learning rate for optimizer")  
    parser.add_argument('--weight_decay', type=float, default=1e-6,  
                       help="Weight decay for optimizer")  
      
    # 模型参数  
    parser.add_argument('--node_feat_size', type=int, default=35,  
                       help="Node feature size")  
    parser.add_argument('--edge_feat_size', type=int, default=17,  
                       help="Edge feature size")  
    parser.add_argument('--hidden_feat_size', type=int, default=256,  
                       help="Hidden feature size")  
    parser.add_argument('--layer_num', type=int, default=3,  
                       help="Number of layers in the model")  
      
    # 其他参数  
    parser.add_argument('--num_workers', type=int, default=8,  
                       help="Number of workers for DataLoader")  
    parser.add_argument('--cuda_device', type=str, default="0",  
                       help="CUDA device ID (e.g., '0' or '0,1')")  
    parser.add_argument('--save_path', type=str, default='best_model.pt',  
                       help="Path to save/load the model")  
    parser.add_argument('--log_file', type=str, default=None,  
                       help="Path to save training log file")  
    parser.add_argument('--output_csv', type=str, default=None,  
                       help="Path to save prediction results CSV (test mode only)")  
      
    args = parser.parse_args()  
      
    # 配置日志系统  
    if args.log_file:  
        logging.basicConfig(  
            level=logging.INFO,  
            format='%(asctime)s - %(levelname)s - %(message)s',  
            datefmt='%Y-%m-%d %H:%M:%S',  
            handlers=[  
                logging.FileHandler(args.log_file, mode='w'),  
                logging.StreamHandler()  
            ]  
        )  
        logging.info(f"📝 日志将保存到: {args.log_file}")  
    else:  
        logging.basicConfig(  
            level=logging.INFO,  
            format='%(asctime)s - %(levelname)s - %(message)s',  
            datefmt='%Y-%m-%d %H:%M:%S',  
            handlers=[logging.StreamHandler()]  
        )  
      
    # 设置 CUDA 设备  
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device  
      
    # 读取 CSV 文件  
    df = pd.read_csv(args.data_csv)  
    required_columns = ['name', 'receptor', 'ligand', 'pk']  
    if not all(col in df.columns for col in required_columns):  
        # 尝试 pK (大写K)  
        if 'pK' in df.columns:  
            df = df.rename(columns={'pK': 'pk'})  
        else:  
            raise ValueError(f"CSV must contain columns: {required_columns} (or 'pK' instead of 'pk')")  
      
    logging.info(f"📂 读取 CSV 文件: {args.data_csv}")  
    logging.info(f"   - CSV 中总样本数: {len(df)}")  
      
    if args.mode == 'train':  
        # 训练模式：随机分割为训练集和验证集  
        train_df, valid_df = train_test_split(  
            df,  
            train_size=args.train_ratio,  
            random_state=args.random_seed,  
            shuffle=True  
        )  
        logging.info(f"\n🔄 训练模式: 自动分割数据集 ({args.train_ratio*100:.0f}% 训练 / {(1-args.train_ratio)*100:.0f}% 验证)")  
        logging.info(f"   - 训练集 CSV 样本数: {len(train_df)}")  
        logging.info(f"   - 验证集 CSV 样本数: {len(valid_df)}")  
    else:  
        # 测试模式：使用全部数据  
        train_df = df  
        valid_df = df  
        logging.info(f"\n🧪 测试模式: 使用全部数据进行评估")  
        logging.info(f"   - 数据集 CSV 样本数: {len(df)}")  
      
    # 创建数据集  
    logging.info(f"\n🔍 检查训练集 DGL 文件...")  
    train_set = CustomGraphDataset(train_df, graph_type=args.graph_type)  
      
    logging.info(f"\n🔍 检查验证集 DGL 文件...")  
    valid_set = CustomGraphDataset(valid_df, graph_type=args.graph_type)  
      
    logging.info(f"\n✅ 最终数据集统计:")  
    logging.info(f"   - 训练集有效样本: {len(train_set)}")  
    logging.info(f"   - 验证集有效样本: {len(valid_set)}")  
      
    # 创建 DataLoader  
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  
                             collate_fn=collate_fn, num_workers=args.num_workers)  
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,  
                             collate_fn=collate_fn, num_workers=args.num_workers)  
      
    # 初始化模型  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    model = DTIPredictor(node_feat_size=args.node_feat_size,  
                        edge_feat_size=args.edge_feat_size,  
                        hidden_feat_size=args.hidden_feat_size,  
                        layer_num=args.layer_num).to(device)  
      
    # ========== 关键修改：区分训练和测试模式 ==========  
    if args.mode == 'test':  
        # 测试模式：加载模型并评估  
        logging.info(f"\n📥 测试模式：加载模型权重: {args.save_path}")  
        if not os.path.exists(args.save_path):  
            raise FileNotFoundError(f"❌ 模型文件不存在: {args.save_path}")  
          
        model.load_state_dict(torch.load(args.save_path))  
        logging.info("✅ 模型加载成功!")  
          
        # 进行评估并获取详细预测结果  
        logging.info(f"\n🧪 开始测试评估...")  
        test_rmse, test_pr, predictions, labels = val(model, valid_loader, device, return_predictions=True)  
        logging.info(f"\n🎉 测试结果: RMSE={test_rmse:.4f}, Pearson={test_pr:.4f}")  
          
        # 保存预测结果到 CSV  
        if args.output_csv is None:  
            # 默认输出文件名  
            csv_dir = os.path.dirname(args.data_csv)  
            csv_basename = os.path.splitext(os.path.basename(args.data_csv))[0]  
            args.output_csv = os.path.join(csv_dir, f"{csv_basename}_predictions.csv")  
          
        # 创建预测结果 DataFrame  
        # 获取有效样本的信息  
        valid_indices = valid_set.valid_indices  
        result_df = valid_df.iloc[valid_indices].copy()  
        result_df['predicted_pK'] = predictions.flatten()  
        result_df['true_pK'] = labels.flatten()  
        result_df['error'] = result_df['predicted_pK'] - result_df['true_pK']  
        result_df['absolute_error'] = np.abs(result_df['error'])  
          
        # 保存到 CSV  
        result_df.to_csv(args.output_csv, index=False)  
        logging.info(f"📊 预测结果已保存到: {args.output_csv}")  
        logging.info(f"   - 包含列: {list(result_df.columns)}")  
          
    else:  
        # 训练模式：执行训练循环  
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,  
                              weight_decay=args.weight_decay)  
        criterion = nn.MSELoss()  
          
        logging.info(f"\n🚀 开始训练...")  
        logging.info(f"   - 学习率: {args.learning_rate}")  
        logging.info(f"   - 批次大小: {args.batch_size}")  
        logging.info(f"   - 最大轮数: {args.epochs}")  
        logging.info(f"   - 早停耐心值: {args.early_stop_epoch}")  
          
        # 训练循环  
        best_valid_rmse = float('inf')  
        patience_counter = 0  
          
        model.train()  
        for epoch in range(args.epochs):  
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
              
            logging.info(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, train_rmse={epoch_rmse:.4f}, "  
                        f"valid_rmse={valid_rmse:.4f}, valid_pr={valid_pr:.4f}")  
              
            # 保存最佳模型  
            if valid_rmse < best_valid_rmse:  
                best_valid_rmse = valid_rmse  
                patience_counter = 0  
                torch.save(model.state_dict(), args.save_path)  
                logging.info(f"💾 保存最佳模型 (valid_rmse={valid_rmse:.4f})")  
            else:  
                patience_counter += 1  
                if patience_counter >= args.early_stop_epoch:  
                    logging.info(f"⏹️  Early stopping at epoch {epoch}")  
                    break  
          
        # 加载最佳模型并最终验证  
        logging.info(f"\n📊 加载最佳模型进行最终评估...")  
        model.load_state_dict(torch.load(args.save_path))  
        final_valid_rmse, final_valid_pr = val(model, valid_loader, device)  
        logging.info(f"\n🎉 最终验证结果: RMSE={final_valid_rmse:.4f}, Pearson={final_valid_pr:.4f}")  
  
if __name__ == '__main__':  
    main()

