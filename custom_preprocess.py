import os  
import pickle  
import argparse  
from rdkit import Chem  
import pandas as pd  
from tqdm import tqdm  
import pymol  
from rdkit import RDLogger  
from concurrent.futures import ProcessPoolExecutor, as_completed  
  
RDLogger.DisableLog('rdApp.*')  
  
  
def generate_pocket_from_paths(receptor_path, ligand_path, output_dir, name, distance=5):  
    """从给定的蛋白质和配体文件路径生成口袋"""  
    pocket_output_path = os.path.join(output_dir, f'Pocket_{distance}A.pdb')  
  
    if os.path.exists(pocket_output_path):  
        return pocket_output_path  
  
    try:  
        pymol.cmd.load(receptor_path, 'protein')  
        pymol.cmd.remove('resn HOH')  
        pymol.cmd.load(ligand_path, 'ligand')  
        pymol.cmd.remove('hydrogens')  
        pymol.cmd.select('Pocket', f'byres ligand around {distance}')  
        pymol.cmd.save(pocket_output_path, 'Pocket')  
        pymol.cmd.delete('all')  
        return pocket_output_path  
    except Exception as e:  
        print(f"Error generating pocket for {name}: {e}")  
        return None  
  
  
def process_ligand_file(ligand_path, output_dir, name):  
    """处理配体文件，转换为PDB格式"""  
    ligand_output_path = os.path.join(output_dir, f'{name}_ligand.pdb')  
  
    file_ext = os.path.splitext(ligand_path)[1].lower()  
  
    if file_ext == '.pdb':  
        import shutil  
        shutil.copy2(ligand_path, ligand_output_path)  
    else:  
        os.system(f'obabel "{ligand_path}" -O "{ligand_output_path}" -d')  
  
    return ligand_output_path  
  
  
def process_one_complex(row_dict, distance):  
    """处理单个复合物，直接在原始文件目录下生成中间文件"""  
    receptor_path = row_dict['receptor']  
    ligand_path = row_dict['ligand']  
    name = row_dict['name']  
    pk_value = float(row_dict['pk'])  
  
    # 使用蛋白质文件的父目录作为输出目录  
    complex_dir = os.path.dirname(receptor_path)  
  
    if not os.path.exists(receptor_path):  
        print(f"Receptor file not found: {receptor_path}")  
        return None  
    if not os.path.exists(ligand_path):  
        print(f"Ligand file not found: {ligand_path}")  
        return None  
  
    try:  
        pocket_path = generate_pocket_from_paths(  
            receptor_path, ligand_path, complex_dir, name, distance  
        )  
        if pocket_path is None:  
            return None  
  
        ligand_pdb_path = process_ligand_file(ligand_path, complex_dir, name)  
  
        ligand = Chem.MolFromPDBFile(ligand_pdb_path, removeHs=True)  
        if ligand is None:  
            print(f"Unable to process ligand of {name}")  
            return None  
  
        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)  
        if pocket is None:  
            print(f"Unable to process protein of {name}")  
            return None  
  
        save_path = os.path.join(complex_dir, f"{name}.rdkit")  
        with open(save_path, 'wb') as f:  
            pickle.dump((ligand, pocket), f)  
  
        return {  
            'name': name,   
            'pK': pk_value,  
            'complex_dir': complex_dir  # 返回实际的复合物目录路径  
        }  
  
    except Exception as e:  
        print(f"Error processing {name}: {e}")  
        return None  
  
  
def generate_complex_from_csv(csv_file, distance=5, n_jobs=1):  
    """从CSV文件生成复合物数据（多进程版本）"""  
    df = pd.read_csv(csv_file)  
  
    required_columns = ['receptor', 'ligand', 'name', 'pk']  
    if not all(col in df.columns for col in required_columns):  
        raise ValueError(f"CSV must contain columns: {required_columns}")  
  
    rows = df.to_dict(orient='records')  
    processed_data = []  
  
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:  
        futures = {  
            executor.submit(process_one_complex, row, distance): row['name']  
            for row in rows  
        }  
  
        pbar = tqdm(total=len(futures), desc="Processing")  
        for future in as_completed(futures):  
            result = future.result()  
            if result:  
                processed_data.append(result)  
            pbar.update(1)  
        pbar.close()  
  
    if processed_data:  
        # 将处理结果保存到CSV文件的同一目录  
        csv_dir = os.path.dirname(os.path.abspath(csv_file))  
        processed_csv_path = os.path.join(csv_dir, "processed_data.csv")  
          
        processed_df = pd.DataFrame(processed_data)  
        processed_df.to_csv(processed_csv_path, index=False)  
        print(f"Processed data saved to: {processed_csv_path}")  
        return processed_df, processed_csv_path  
    else:  
        print("No data was successfully processed")  
        return None, None  
  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description="Process protein-ligand complexes from CSV (in-place processing).")  
    parser.add_argument('--csv', type=str, required=True, help="Input CSV file path")  
    parser.add_argument('--distance', type=float, default=5.0, help="Pocket extraction distance (Å)")  
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of processes")  
  
    args = parser.parse_args()  
  
    print("Processing complexes from CSV (in-place)...")  
    processed_df, processed_csv = generate_complex_from_csv(  
        args.csv, args.distance, args.n_jobs  
    )  
  
    if processed_df is not None:  
        print(f"Successfully processed {len(processed_df)} complexes")  
        print("Generated files in each complex directory:")  
        print("- Pocket_5A.pdb (protein pocket)")  
        print("- {name}_ligand.pdb (converted ligand)")  
        print("- {name}.rdkit (serialized complex)")  
    else:  
        print("No complexes were successfully processed")