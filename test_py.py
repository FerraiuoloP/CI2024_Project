import numpy as np


#PASTE HERE THE sXXXXX.py FILE SO WE CAN TEST IT BEFORE SUBMITTING IT


# f0 is provided by the professor
def f0(x: np.ndarray) -> np.ndarray:
    return 0




def f1(x: np.ndarray) -> np.ndarray: 
    return 0


def f2(x: np.ndarray) -> np.ndarray: 
    return 0


def f3(x: np.ndarray) -> np.ndarray: 
    return 0


def f4(x: np.ndarray) -> np.ndarray: 
    return 0


def f5(x: np.ndarray) -> np.ndarray: 
    return 0


def f6(x: np.ndarray) -> np.ndarray: 
    return 0


def f7(x: np.ndarray) -> np.ndarray: 
    return 0


def f8(x: np.ndarray) -> np.ndarray:
    return 0

functions = [f0, f1, f2, f3, f4, f5, f6, f7, f8]


for i in range(9):
    # Load the data
    with np.load(f'data/problem_{i}.npz') as problem:
        x = problem['x']
        y = problem['y']

    
    # Test the functions and compute the mse
    y_pred = functions[i](x)
    mse = np.mean(np.square(y - y_pred))
    print("Problem", i, "MSE:", mse)




