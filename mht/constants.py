import numpy as np

EPS = np.finfo('d').eps#极小的浮点正数
LARGE = np.finfo('d').max#极大的浮点正数
LOG_0 = -LARGE#极小的浮点负数
MISS = None#丢包否
if __name__ == '__main__':
    #help(np.finfo('d'))
    print(EPS)