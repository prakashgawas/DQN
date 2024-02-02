#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:50:46 2024

@author: prakashgawas
"""
import multiprocessing
# def add_to_value(addend, value, lock):
#     with lock:
#         value.value += addend

# if __name__ == '__main__':
#     with multiprocessing.Manager() as manager:
#         lock = manager.Lock()
#         value = manager.Value(float, 0.0)
#         with multiprocessing.Pool(2) as pool:
#             pool.starmap(add_to_value,
#                           [(float(i), value, lock) for i in range(100)])
#         print(value.value)
        
        
from multiprocessing.managers import BaseManager

class MathsClass:
    def __init__(self):
        self.z = 5
    def add(self, x, y):
        self.z =  x + y
    def return_z(self):
        return self.z

class MyManager(BaseManager):
    pass

MyManager.register('Maths', MathsClass)

if __name__ == '__main__':
    with MyManager() as manager:
        maths = manager.Maths()
        with multiprocessing.Pool(2) as pool:
            maths.add(2,4)
            print(maths.return_z())
        
        
            
