import argparse
import pandas as pd
import torch
import numpy as np
import sys

from pathlib import Path
path = Path(__file__).parent
path = path  / "mostlyai"
sys.path.insert(0, str(path))
from mostlyai import engine




def parse_args():
    parser = argparse.ArgumentParser(description="Train synthetic data generation model")
    
    parser.add_argument("input_csv", type=str, help="Insert the path to the input CSV file")
    parser.add_argument("--folder_name", type=str, default="sequential_test", 
                       help="Insert the name of the folder where the model will be saved (default: sequential_test)")
    parser.add_argument("--choice", type=str, default="sequential", 
                       choices=["flat", "sequential"],
                       help="Choose the model type: 'flat' for flat tabular data, 'sequential' for sequential data (default: sequential)")        
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workspace_dir = f"./{args.folder_name}"
    ws = Path(workspace_dir)
    
    file_name = args.input_csv
    if ".csv" not in file_name:
        file_name = f"{file_name}.csv"
    df = pd.read_csv(file_name)

    engine.init_logging()

    choice = args.choice.lower()
    print(f"Choice: {choice}")
    if choice not in ["flat", "sequential"]:
        choice = "flat"

    if choice.lower() == "flat":
        engine.set_random_state(0)
    else:
        engine.set_random_state(46)

    if choice.lower() == "flat":
        engine.split(                   
            workspace_dir=ws,
            tgt_data=df,
            model_type="TABULAR",
            trn_val_split=0.975
        )
    else:
        engine.split(                   
            workspace_dir=ws,
            tgt_data=df,
            model_type="TABULAR",
            trn_val_split=0.975,
            tgt_context_key="group_id"
        )

    engine.analyze(workspace_dir=ws)
    engine.encode(workspace_dir=ws)

    if choice.lower() == "flat":
        engine.train(
            workspace_dir=ws,
            model="MOSTLY_AI/Large",
            enable_flexible_generation=False,
            max_epochs=1000, 
            device=device,  
        )   
    else:
        engine.train(
            workspace_dir=ws,
            model="MOSTLY_AI/Medium",
            batch_size=512,
            max_sequence_window=10, 
            enable_flexible_generation= False, 
            max_epochs=400, 
            device=device,
        )

    if choice.lower() == "flat":
        sample_size = 100000
    else:
        sample_size = 20000


    engine.generate(
        workspace_dir=ws, 
        sample_size=sample_size,
        device=device,
        sampling_temperature=1.0,
        sampling_top_p=1.0
    )

        
    generated_data = pd.read_parquet(ws / "SyntheticData")

    if choice.lower() == "flat":
        output_path = ws / "synthetic_flat.csv"
    else:
        output_path = ws / "synthetic_seq.csv"
    generated_data.to_csv(output_path, index=False)

    print(f"Synthetic data saved to: {output_path}")


if __name__ == "__main__":
    main()