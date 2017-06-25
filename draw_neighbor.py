from __future__ import division # python 2
import numpy as np
import random
from matplotlib import pyplot as plt
import operator  
import csv
import glob

filename = "./recordToFindKValue.txt"
infile = open(filename)

plt.figure(1)



x = []
a = []
b = []

counter = 0
flag = 0

for line in infile:
	if(line == "\n"):
		continue
	words = line.split()
	if words[0] == "k_value:" :
		x.append(int(words[len(words)-1]))
		counter = 0
	elif words[0] == "Result" :
		if(flag == 0):
			flag = 1
			continue
		if counter == 0:
			a.append(float(words[len(words)-1]))
		elif counter == 1 :
			b.append(float(words[len(words)-1]))
		counter += 1

print len(x)
print len(a)
print len(b)

plt.plot(x,a,'r', label="movie_movie")
plt.plot(x,b, label = "user_user ")

plt.xlim(0.0, 50)
plt.ylim(0.93, 1.06)

plt.title("k_value vs RMSE")
plt.xlabel("k_value")
plt.ylabel("RMSE")
plt.legend()




plt.show()
