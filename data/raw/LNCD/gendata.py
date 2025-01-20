import numpy
import random

datalen = 8164

arr = list(range(0, 8164))
random.shuffle(arr)

split1 = int(datalen*0.7)
split2 = int(datalen*0.8)

arr0 = arr[:split1]
arr1 = arr[split1: split2]
arr2 = arr[split2:]

train_idx =  numpy.array(arr0)
val_idx   =  numpy.array(arr1)
test_idx  =  numpy.array(arr2)

numpy.savez('example.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)