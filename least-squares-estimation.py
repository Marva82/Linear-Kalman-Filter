import numpy as np

def CalculateLeastSquaresSolution(Hmatrix, Ymatrix):
    # 1. Transpose the H matrix
    Ht = np.transpose(Hmatrix)
    
    # 2. Calculate the inverse of (H^T * H)
    Ht_H = np.matmul(Ht, Hmatrix)
    Ht_H_inv = np.linalg.inv(Ht_H)
    
    # 3. Calculate (H^T * Y)
    Ht_Y = np.matmul(Ht, Ymatrix)
    
    # 4. Multiply them together to get the final X matrix
    Xmatrix = np.matmul(Ht_H_inv, Ht_Y)
    
    return Xmatrix

def CalculateLineOfBestFitSolution(Dataset):
    H_list = []
    Y_list = []
    
    # Loop through each [y_i, r_i] pair in the dataset
    for data_point in Dataset:
        y_i = data_point[0]
        r_i = data_point[1]
        
        # Append as a row to the Y matrix
        Y_list.append([y_i])
        
        # Append as a row to the H matrix: [r_i, 1]
        H_list.append([r_i, 1])
        
    # Convert standard Python lists to NumPy arrays
    Hmatrix = np.array(H_list)
    Ymatrix = np.array(Y_list)
    
    # Call the solver function from Part 1
    LineParam = CalculateLeastSquaresSolution(Hmatrix, Ymatrix)
    
    return LineParam