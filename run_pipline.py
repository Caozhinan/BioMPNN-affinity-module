#!/usr/bin/env python  
"""  
整合脚本：自动执行预处理、图构建和训练/测试的完整流程  
"""  
  
import os  
import sys  
import argparse  
import subprocess  
import pandas as pd  
from pathlib import Path  
  
  
def run_command(cmd, description):  
    """执行命令并处理错误"""  
    print(f"\n{'='*80}")  
    print(f"Step: {description}")  
    print(f"Command: {' '.join(cmd)}")  
    print(f"{'='*80}\n")  
      
    result = subprocess.run(cmd, capture_output=False, text=True)  
      
    if result.returncode != 0:  
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")  
        sys.exit(1)  
      
    print(f"\n✅ {description} completed successfully\n")  
  
  
def main():  
    parser = argparse.ArgumentParser(  
        description="端到端流程：预处理 → 图构建 → 训练/测试",  
        formatter_class=argparse.RawDescriptionHelpFormatter,  
        epilog="""  
示例用法:  
  # 完整训练流程  
  python run_pipeline.py --data_csv data.csv --mode train  
    
  # 只做预处理和图构建  
  python run_pipeline.py --data_csv data.csv --skip_train  
    
  # 测试模式  
  python run_pipeline.py --data_csv data.csv --mode test  
    
  # 跳过预处理  
  python run_pipeline.py --data_csv data.csv --mode train --skip_preprocess  
    
  # 跳过图构建  
  python run_pipeline.py --data_csv data.csv --mode train --skip_graph  
        """  
    )  
      
    # === 必需参数 ===  
    parser.add_argument('--data_csv', type=str, required=True,  
                       help="输入 CSV 文件路径 (格式: name,receptor,ligand,pk)")  
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',  
                       help="运行模式: train (训练) 或 test (测试)")  
      
    # === 流程控制参数 ===  
    parser.add_argument('--skip_preprocess', action='store_true',  
                       help="跳过预处理步骤（假设 .rdkit 文件已存在）")  
    parser.add_argument('--skip_graph', action='store_true',  
                       help="跳过图构建步骤（假设 .dgl 文件已存在）")  
    parser.add_argument('--skip_train', action='store_true',  
                       help="跳过训练/测试步骤，只执行预处理和图构建")  
      
    # === 预处理参数 (batch_custom_preprocess.py) ===  
    parser.add_argument('--distance', type=float, default=5.0,  
                       help="口袋提取距离 (Å) [默认: 5.0]")  
    parser.add_argument('--preprocess_n_jobs', type=int, default=4,  
                       help="预处理并行进程数 [默认: 4]")  
    parser.add_argument('--preprocess_batch_size', type=int, default=1000,  
                       help="预处理批次大小 [默认: 1000]")  
    parser.add_argument('--preprocess_timeout', type=int, default=300,  
                       help="单个复合物处理超时时间(秒) [默认: 300]")  
      
    # === 图构建参数 (graph_constructor.py) ===  
    parser.add_argument('--graph_type', type=str, default='Graph_EHIGN_5edges',  
                       help="图类型前缀 [默认: Graph_EHIGN_5edges]")  
    parser.add_argument('--dis_threshold', type=float, default=5.0,  
                       help="相互作用距离阈值 (Å) [默认: 5.0]")  
    parser.add_argument('--graph_num_process', type=int, default=28,  
                       help="图构建并行进程数 [默认: 28]")  
      
    # === 训练参数 (train.py) ===  
    parser.add_argument('--batch_size', type=int, default=96,  
                       help="训练批次大小 [默认: 96]")  
    parser.add_argument('--epochs', type=int, default=200,  
                       help="最大训练轮数 [默认: 200]")  
    parser.add_argument('--early_stop_epoch', type=int, default=30,  
                       help="早停耐心值 [默认: 30]")  
    parser.add_argument('--learning_rate', type=float, default=1e-4,  
                       help="学习率 [默认: 1e-4]")  
    parser.add_argument('--weight_decay', type=float, default=1e-6,  
                       help="权重衰减 [默认: 1e-6]")  
    parser.add_argument('--train_ratio', type=float, default=0.9,  
                       help="训练集比例（仅训练模式）[默认: 0.9]")  
    parser.add_argument('--random_seed', type=int, default=42,  
                       help="随机种子 [默认: 42]")  
      
    # === 模型参数 ===  
    parser.add_argument('--node_feat_size', type=int, default=35,  
                       help="节点特征维度 [默认: 35]")  
    parser.add_argument('--edge_feat_size', type=int, default=17,  
                       help="边特征维度 [默认: 17]")  
    parser.add_argument('--hidden_feat_size', type=int, default=256,  
                       help="隐藏层维度 [默认: 256]")  
    parser.add_argument('--layer_num', type=int, default=3,  
                       help="GNN 层数 [默认: 3]")  
      
    # === 其他参数 ===  
    parser.add_argument('--num_workers', type=int, default=8,  
                       help="DataLoader 工作进程数 [默认: 8]")  
    parser.add_argument('--cuda_device', type=str, default="0",  
                       help="CUDA 设备 ID [默认: '0']")  
    parser.add_argument('--save_path', type=str, default='best_model.pt',  
                       help="模型保存路径 [默认: best_model.pt]")  
      
    args = parser.parse_args()  
      
    # 验证输入文件  
    if not os.path.exists(args.data_csv):  
        print(f"❌ Error: CSV 文件不存在: {args.data_csv}")  
        sys.exit(1)  
      
    # 验证 CSV 格式  
    try:  
        df = pd.read_csv(args.data_csv)  
        required_columns = ['name', 'receptor', 'ligand', 'pk']  
        if not all(col in df.columns for col in required_columns):  
            print(f"❌ Error: CSV 必须包含列: {required_columns}")  
            print(f"   实际列: {list(df.columns)}")  
            sys.exit(1)  
        print(f"✅ CSV 文件验证通过: {len(df)} 个复合物")  
    except Exception as e:  
        print(f"❌ Error: 无法读取 CSV 文件: {e}")  
        sys.exit(1)  
      
    csv_dir = os.path.dirname(os.path.abspath(args.data_csv))  
    processed_csv = os.path.join(csv_dir, "processed_valid.csv")  
      
    # ========== Step 1: 预处理 ==========  
    if not args.skip_preprocess:  
        preprocess_cmd = [  
            'python', 'batch_custom_preprocess.py',  
            '--csv', args.data_csv,  
            '--distance', str(args.distance),  
            '--n_jobs', str(args.preprocess_n_jobs),  
            '--batch_size', str(args.preprocess_batch_size),  
            '--timeout', str(args.preprocess_timeout)  
        ]  
        run_command(preprocess_cmd, "预处理 (batch_custom_preprocess.py)")  
          
        # 检查预处理输出  
        if not os.path.exists(processed_csv):  
            print(f"❌ Error: 预处理未生成输出文件: {processed_csv}")  
            sys.exit(1)  
    else:  
        print(f"\n⏭️  跳过预处理步骤")  
        if not os.path.exists(processed_csv):  
            print(f"⚠️  Warning: processed_csv 不存在，将使用原始 CSV: {args.data_csv}")  
            processed_csv = args.data_csv  
      
    # ========== Step 2: 图构建 ==========  
    if not args.skip_graph:  
        graph_cmd = [  
            'python', 'graph_constructor.py',  
            '--csv_file', processed_csv,  
            '--graph_type', args.graph_type,  
            '--dis_threshold', str(args.dis_threshold),  
            '--num_process', str(args.graph_num_process),  
            '--create'  
        ]  
        run_command(graph_cmd, "图构建 (graph_constructor.py)")  
    else:  
        print(f"\n⏭️  跳过图构建步骤")  
      
    # ========== Step 3: 训练/测试 ==========  
    if not args.skip_train:  
        train_cmd = [  
            'python', 'train.py',  
            '--data_csv', processed_csv,  
            '--mode', args.mode,  
            '--graph_type', args.graph_type,  
            '--batch_size', str(args.batch_size),  
            '--epochs', str(args.epochs),  
            '--early_stop_epoch', str(args.early_stop_epoch),  
            '--learning_rate', str(args.learning_rate),  
            '--weight_decay', str(args.weight_decay),  
            '--train_ratio', str(args.train_ratio),  
            '--random_seed', str(args.random_seed),  
            '--node_feat_size', str(args.node_feat_size),  
            '--edge_feat_size', str(args.edge_feat_size),  
            '--hidden_feat_size', str(args.hidden_feat_size),  
            '--layer_num', str(args.layer_num),  
            '--num_workers', str(args.num_workers),  
            '--cuda_device', args.cuda_device,  
            '--save_path', args.save_path  
        ]  
          
        mode_desc = "训练" if args.mode == 'train' else "测试"  
        run_command(train_cmd, f"{mode_desc} (train.py)")  
    else:  
        print(f"\n⏭️  跳过训练/测试步骤")  
      
    # ========== 完成 ==========  
    print(f"\n{'='*80}")  
    print(f"🎉 完整流程执行成功！")  
    print(f"{'='*80}")  
    if not args.skip_train:  
        mode_desc = "训练" if args.mode == 'train' else "测试"  
        print(f"模式: {mode_desc}")  
    print(f"输入 CSV: {args.data_csv}")  
    print(f"处理后 CSV: {processed_csv}")  
    if args.mode == 'train' and not args.skip_train:  
        print(f"最佳模型: {args.save_path}")  
    print(f"{'='*80}\n")  
  
  
if __name__ == '__main__':  
    main()