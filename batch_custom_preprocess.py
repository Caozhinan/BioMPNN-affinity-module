import os
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import Chem, RDLogger
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import logging
import multiprocessing
import shutil
import subprocess

# --- 全局设置 ---
RDLogger.DisableLog('rdApp.*')
log_file = "processing_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

# --- 核心功能函数 ---

def generate_pocket_from_paths(receptor_path, ligand_path, output_dir, name, distance=5):
    """
    生成口袋文件。使用唯一的对象名称避免PyMOL警告，并在保存前标准化PDB格式。
    """
    pocket_output_path = os.path.join(output_dir, f'Pocket_{distance}A.pdb')
    
    if os.path.exists(pocket_output_path) and os.path.getsize(pocket_output_path) > 0:
        # 快速检查，如果已存在的PDB文件能被我们新的、更宽容的加载方式读取，就跳过
        try:
            temp_mol = Chem.MolFromPDBFile(pocket_output_path, removeHs=True, sanitize=False)
            if temp_mol is not None:
                return pocket_output_path
        except Exception:
            logging.warning(f"[{name}] Found existing but invalid pocket file. Regenerating...")

    try:
        pymol.cmd.reinitialize()
        
        # 使用唯一的对象名称（例如，基于进程ID和名称），避免命名冲突和警告
        receptor_obj = f"receptor_{name}"
        ligand_obj = f"ligand_{name}"

        pymol.cmd.load(receptor_path, receptor_obj)
        pymol.cmd.remove(f'resn HOH and {receptor_obj}')
        pymol.cmd.load(ligand_path, ligand_obj)
        pymol.cmd.remove(f'hydrogens and {ligand_obj}')
        
        selection_name = f'Pocket_{name}'
        pymol.cmd.select(selection_name, f'byres {ligand_obj} around {distance}')
        
        atom_count = pymol.cmd.count_atoms(selection_name)
        if atom_count == 0:
            logging.warning(f"[{name}] PyMOL selection for pocket was empty. No pocket generated.")
            return None
            
        # 强制标准化B-factor和Occupancy，解决PDB格式问题
        pymol.cmd.alter(selection_name, 'b=20.0')
        pymol.cmd.alter(selection_name, 'o=1.00')

        pymol.cmd.save(pocket_output_path, selection_name)

        if os.path.exists(pocket_output_path) and os.path.getsize(pocket_output_path) > 0:
            return pocket_output_path
        else:
            logging.error(f"[{name}] PyMOL saved an empty or no file for the pocket.")
            return None

    except Exception as e:
        logging.error(f"[{name}] PyMOL Error generating pocket: {e}", exc_info=True)
        return None
    finally:
        pymol.cmd.delete('all')

def process_ligand_file(ligand_path, output_dir, name):
    """
    处理配体文件，检查空文件。
    """
    ligand_output_path = os.path.join(output_dir, f'{name}_ligand.pdb')
    file_ext = os.path.splitext(ligand_path)[1].lower()
    
    try:
        if file_ext == '.pdb':
            shutil.copy2(ligand_path, ligand_output_path)
        else:
            result = subprocess.run(
                ['obabel', ligand_path, '-O', ligand_output_path, '-d'],
                capture_output=True, text=True, check=False, timeout=120
            )
            if result.returncode != 0:
                logging.error(f"[{name}] OpenBabel conversion failed. Stderr: {result.stderr}")
                return None
        
        if os.path.exists(ligand_output_path) and os.path.getsize(ligand_output_path) > 0:
            return ligand_output_path
        else:
            logging.warning(f"[{name}] Ligand file was generated but is empty. Original: {ligand_path}")
            return None

    except subprocess.TimeoutExpired:
        logging.error(f"[{name}] OpenBabel conversion timed out.")
        return None
    except Exception as e:
        logging.error(f"[{name}] Error processing ligand file: {e}")
        return None

def process_one_complex(row_dict, distance):
    """
    处理单个复合物的完整流程，包含针对非连续残基的健壮RDKit加载逻辑。
    """
    name = row_dict['name']
    try:
        receptor_path = row_dict['receptor']
        ligand_path = row_dict['ligand']
        pk_value = float(row_dict['pk'])
        complex_dir = os.path.dirname(receptor_path)

        if not all(os.path.exists(p) for p in [receptor_path, ligand_path]):
            logging.warning(f"[{name}] Receptor or ligand file not found. Skipping.")
            return None

        ligand_pdb_path = process_ligand_file(ligand_path, complex_dir, name)
        if not ligand_pdb_path: 
            return None
        
        pocket_path = generate_pocket_from_paths(receptor_path, ligand_pdb_path, complex_dir, name, distance)
        if not pocket_path: 
            return None

        # --- 关键修复：健壮地加载PDB文件 ---
        ligand = Chem.MolFromPDBFile(ligand_pdb_path, removeHs=True)
        if ligand is None:
            logging.error(f"[{name}] RDKit failed to load ligand from file: {ligand_pdb_path}")
            return None

        # 1. 以“宽容”模式加载，不进行化学检查 (sanitize=False)
        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True, sanitize=False)
        if pocket is None:
            # 如果这样都加载失败，说明文件本身可能有更严重的问题
            logging.error(f"[{name}] RDKit failed to load pocket even with sanitize=False: {pocket_path}")
            return None
        
        # 2. 尝试进行化学检查，如果失败，则记录警告但继续使用
        try:
            Chem.SanitizeMol(pocket)
        except Exception as e:
            logging.warning(f"[{name}] Could not sanitize pocket molecule (likely due to discontinuous residues). Proceeding with unsanitized molecule. RDKit error: {e}")
        # ------------------------------------

        save_path = os.path.join(complex_dir, f"{name}.rdkit")
        with open(save_path, 'wb') as f:
            pickle.dump((ligand, pocket), f)

        return {'name': name, 'pK': pk_value, 'complex_dir': complex_dir}

    except Exception as e:
        logging.error(f"[{name}] Unhandled exception in process_one_complex: {e}", exc_info=True)
        return None

# --- 分批处理主函数 (无变化) ---
def generate_complex_from_csv_batched(csv_file, distance=5, n_jobs=1, batch_size=1000, task_timeout=300):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.error(f"Input CSV file not found: {csv_file}")
        return None, None
        
    required_columns = ['receptor', 'ligand', 'name', 'pk']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    rows = df.to_dict(orient='records')
    total_tasks = len(rows)
    batches = [rows[i:i + batch_size] for i in range(0, total_tasks, batch_size)]
    
    processed_data = []
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    processed_csv_path = os.path.join(csv_dir, "processed_valid.csv")

    main_pbar = tqdm(total=total_tasks, desc="Overall Progress")
    for i, batch in enumerate(batches):
        logging.info(f"--- Starting Batch {i+1}/{len(batches)} (Size: {len(batch)}) ---")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(process_one_complex, row, distance): row['name'] for row in batch}
            
            batch_pbar = tqdm(total=len(futures), desc=f"Batch {i+1}", leave=False)
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=task_timeout)
                    if result:
                        processed_data.append(result)
                except TimeoutError:
                    logging.error(f"Task for '{name}' timed out after {task_timeout} seconds and was skipped.")
                except Exception as exc:
                    logging.error(f"Task for '{name}' generated an exception: {exc}")
                
                batch_pbar.update(1)
                main_pbar.update(1)
            batch_pbar.close()

        if processed_data:
            temp_df = pd.DataFrame(processed_data)
            header = not os.path.exists(processed_csv_path)
            temp_df.to_csv(processed_csv_path, mode='a', index=False, header=header)
            logging.info(f"Batch {i+1} results saved. Total processed so far: {len(pd.read_csv(processed_csv_path))}")
            processed_data = []

    main_pbar.close()
    
    if os.path.exists(processed_csv_path):
        final_df = pd.read_csv(processed_csv_path)
        logging.info(f"All batches complete. Final processed data saved to: {processed_csv_path}")
        return final_df, processed_csv_path
    else:
        logging.warning("No data was successfully processed.")
        return None, None

# --- 主程序入口 (无变化) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Robustly process protein-ligand complexes from a CSV file using batch processing.")
    parser.add_argument('--csv', type=str, required=True, help="Input CSV file path with complex information.")
    parser.add_argument('--distance', type=float, default=5.0, help="Pocket extraction distance in Angstroms (Å).")
    parser.add_argument('--n_jobs', type=int, default=4, help="Number of parallel processes to use.")
    parser.add_argument('--batch_size', type=int, default=1000, help="Number of complexes to process in each batch.")
    parser.add_argument('--timeout', type=int, default=300, help="Timeout in seconds for processing a single complex.")
    
    args = parser.parse_args()

    processed_csv = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "processed_valid.csv")
    if os.path.exists(processed_csv):
        user_input = input(f"Found existing results file '{processed_csv}'.\nDo you want to remove it and start fresh? (y/n): ")
        if user_input.lower() == 'y':
            try:
                os.remove(processed_csv)
                print("Removed old results file.")
            except OSError as e:
                print(f"Error removing file: {e}")

    logging.info(f"Starting processing with {args.n_jobs} workers, batch size of {args.batch_size}, and task timeout of {args.timeout}s.")
    logging.info(f"Logs will be saved to: {log_file}")
    
    processed_df, processed_csv_path = generate_complex_from_csv_batched(
        csv_file=args.csv, 
        distance=args.distance, 
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        task_timeout=args.timeout
    )

    if processed_df is not None:
        print("\n--- Processing Finished! ---")
        print(f"Successfully processed a total of {len(processed_df)} complexes.")
        print(f"Results have been saved to: {processed_csv_path}")
    else:
        print("\n--- Processing Finished ---")
        print("No complexes were successfully processed. Please check the log file for details.")
    
    print(f"Detailed logs are available in: {log_file}")