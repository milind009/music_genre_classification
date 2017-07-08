import numpy as np
from scipy.io import wavfile 
import os
import nncd as fn
import librosa

genres = ['Blues', 'Classical']

np.random.seed(5)
Theta1 = np.random.rand(50, 25801) * (2*0.0025028) - 0.0025028

Theta2 = np.random.rand(2, 51) * (2*0.31623) - 0.31623
alpha = 0.03
I = np.eye(2);
# genres = ['Rock']
for i in range(10):

 Delta1 = np.zeros((50, 25801), 'float64')
 Delta2 = np.zeros((2, 51), 'float64')
 J=0
 for genre in genres:
  genresDir = os.path.join('C:\Genre', genre) 

  # input matrix and output matrix
  X = np.zeros((75, 25800))
  Y = I[genres.index(genre)];

  files = [f for f in os.listdir(genresDir) if f.endswith('.wav')]
  files.sort()

  print ("Reading " + genre + " Input")
  for trainingExample in range(75):
   signal,sr=librosa.load(genresDir+'/'+files[i])
   mfcc=librosa.feature.mfcc(signal[:660000],sr=sr,n_mfcc=20)
   mfcc=np.reshape(mfcc,(1,25800))
   X[trainingExample] = mfcc
  print ("Input Ready")

  j, del1, del2 = fn.nnCostFunction(X, Y, Theta1, Theta2)
  J += j
  Delta1 += del1
  Delta2 += del2

 Theta1 = Theta1 - alpha*Delta1
 Theta2 = Theta2 - alpha*Delta2
 print ("Cost after iteration",i+1,J)
np.savetxt('C:\Genre/theta1.xls',Theta1,delimiter=',')
np.savetxt('C:\Genre/theta2.xls',Theta2,delimiter=',')
Theta2.shape
Theta1.shape