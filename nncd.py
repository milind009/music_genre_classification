import numpy as np

def nnCostFunction(X, Y, Theta1, Theta2):
	a1 = np.hstack((np.ones((75,1)), X))  
	z2 = np.matmul(a1, Theta1.transpose())
	a2 = np.hstack((np.ones((75, 1)), sigmoid(z2)))
	z3 = np.matmul(a2, Theta2.transpose())
	h = sigmoid(z3)
	J = np.sum(np.sum((-Y)*np.log(h) - (1-Y)*np.log(1-h)))/150
	sigma3 = h - Y;
	# print "sigma3", str(sigma3.shape), str(sigma3.dtype) 
	sigma2 = np.matmul(sigma3, Theta2) * sigmoidGradient(np.hstack((np.ones((75, 1)), z2)))
	# print "sigma2", str(sigma2.shape), str(sigma2.dtype)
	sigma2 = sigma2[:, 1:]
	# print "sigma2", str(sigma2.shape), str(sigma2.dtype)

	# accumulate gradients
	delta_1 = np.matmul(sigma2.transpose(), a1)/150
	delta_2 = np.matmul(sigma3.transpose(), a2)/150

	return J, delta_1, delta_2 

def sigmoid(A):
	return 1/(1+np.exp(-A))

def sigmoidGradient(A):
	return sigmoid(A)*(1-sigmoid(A))
def predict(X, Theta1, Theta2):
 a1 = np.hstack((np.ones((75,1)), X))  
 z2 = np.matmul(a1, Theta1.transpose())
 a2 = np.hstack((np.ones((75, 1)), sigmoid(z2)))
 z3 = np.matmul(a2, Theta2.transpose())
 h = sigmoid(z3)
 predictions = np.argmax(h, 1)
 return predictions