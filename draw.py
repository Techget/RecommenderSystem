from __future__ import division # python 2
import numpy as np
import random
from matplotlib import pyplot as plt
import operator  
import csv
import glob

filename = "./recordToFindDamp.txt"
infile = open(filename)

plt.figure(1)



x = []
a = []
b = []
c = []

counter = 0


for line in infile:
	if(line == "\n"):
		continue
	words = line.split()
	if words[0] == "damp_value:" :
		x.append(float(words[len(words)-1]))
		counter = 0
	elif words[0] == "Result":
		if counter == 0:
			a.append(float(words[len(words)-1]))
		elif counter == 1 :
			b.append(float(words[len(words)-1]))
		elif counter == 2 :
			c.append(float(words[len(words)-1]))
		counter += 1

print len(x)
print len(a)

# naive = plt.plot(x,a,'r', label="native")
# movie_movie = plt.plot(x,b, label = "movie_movie")
# user_user = plt.plot(x,c,"g", label = "user_user")

plt.plot(x,a,'r', label="native")
plt.plot(x,b, label = "movie_movie")
plt.plot(x,c,"g", label = "user_user")

plt.xlim(0.0, 5.1)
plt.ylim(0.852, 1.2)

plt.title("damp_value vs RMSE")
plt.xlabel("Damp paramter")
plt.ylabel("RMSE")
plt.legend()
# plt.legend([naive, movie_movie, user_user], ('naive', 'movie_movie', 'user_user'), 'best')





plt.show()
