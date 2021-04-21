import os
import sys
import yaml
import numpy as np
import pandas as pd

def save_data_arrays(output_dir, X_train, y_train, X_test, y_test):
    os.makedirs(output_dir, exist_ok=True)
        
    with open(os.path.join(output_dir, 'X_train.npy'), 'wb') as f:
        np.save(f, X_train)
        
    with open(os.path.join(output_dir, 'X_test.npy'), 'wb') as f:
        np.save(f, X_test)
        
    with open(os.path.join(output_dir, 'y_train.npy'), 'wb') as f:
        np.save(f, y_train)
    
    with open(os.path.join(output_dir, 'y_test.npy'), 'wb') as f:
        np.save(f, y_test)
        
        
def prepare_data(train_csv_file: str,
                 test_csv_file: str,
                 output_dir: str,
                 target="label"):
           
    # Train
    df_train = pd.read_csv(train_csv_file)
    y_train = np.array(df_train[target])
    X_train = np.array(df_train.drop([target], axis=1))

    # Test
    df_test = pd.read_csv(test_csv_file)
    y_test = np.array(df_test[target])
    X_test = np.array(df_test.drop([target], axis=1))
    
    with open("data_shapes.log", 'w') as outfile:
        outfile.write(f"Training: {X_train.shape}, test: {X_test.shape}\n")
        
    print(f"Training: {X_train.shape}, test: {X_test.shape}\n")
    save_data_arrays(output_dir, X_train, y_train, X_test, y_test)
   
   
if __name__ == "__main__":
    params = yaml.safe_load(open('params.yaml'))['prepare']

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py train-csv test-csv\n")
        sys.exit(1)

    train_csv_file = sys.argv[1]
    test_csv_file  = sys.argv[2]
    
    prepare_data(train_csv_file, test_csv_file, os.path.join('data', 'prepared'))