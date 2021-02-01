#!/usr/bin/python3
from multiprocessing import Process, Queue, Value, Array
import time, os, random
import numpy as np

def qinit(q, num):
    for i in range(num):
        q.put(i)

def moving_average(arr, count):
    return np.sum(arr[:])/count

def f(q, count, arr):
    while not q.empty():
        ind = q.get()
        arr[ind] = ind
        count.value += 1
        num = count.value
        ma = moving_average(arr, num)
        print("pid({}):\t".format(os.getpid()),
              num,
              ma)

if __name__ == '__main__':
    num_proc = 4
    num_data = 100 
    process_list = []
    q = Queue()
    num = Value('i', 0)
    arr = Array('d', np.zeros(num_data))

    qinit(q, num_data)

    for i in range(num_proc):
        p = Process(target=f, args=(q,num,arr))
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()

    print(arr[:])
