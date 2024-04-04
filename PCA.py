import numpy as np
arr = [1,2,3,4,5]
a = [n*n if n%2 == 0 else'odd'for n in arr ]
print(a)