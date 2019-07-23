#!/usr/bin/python

# This file is just to test whether the SFOP library 
# generated works or not 

import ctypes
import numpy as np
# from numpy.ctypeslib import ndpointer

def fillprototype(f, restype, argtypes):
	f.restype = restype
	f.argtypes = argtypes


def main():
	TestLib = ctypes.cdll.LoadLibrary('build/src/libsfop.so')
	fillprototype(TestLib.mymain, ctypes.POINTER(ctypes.c_float), [ctypes.c_char_p])
	s1 = "/home/abhinav/Downloads/my_photo.JPG"
	str_file = ctypes.c_char_p(s1.encode('utf-8'))


	kp = TestLib.mymain(str_file) 	# Get the float pointer
	print("Print from Python :")	
	array_size = int(kp[0]) 
	print(array_size)  				# This number is equal to 1 + 2*no_of_kp.

	cc = np.array(np.fromiter(kp, dtype=np.float32, count=array_size))
	print (cc[1:array_size+1])
	fillprototype(TestLib.free_mem, None, [ctypes.POINTER(ctypes.c_float)])
	TestLib.free_mem(kp)

	no_of_kp = int((array_size-1)/2)
	final_kp = np.reshape(cc[1:array_size+1], (no_of_kp, 2))
	final_kp = np.around(final_kp, decimals = 3)
	print(final_kp)


if __name__ == '__main__':
        main()
