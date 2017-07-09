Music Genre Classification for 2 genres

Neural network consists of three layers

Input layer:25800 nodes(excluding bias)
            Mel Cepstral coefficients(MFCC)  

Hidden layer:50 nodes(excluding bias)

Output layer(h):2 nodes
Y matrix=desired output matrix
Activation function in each layer:sigmoid function

Theta1 Matrix:Edges between input and hidden layer(dimensions[50,25801])

Theta2 Matrix:Edges between hidden and output layer(dimensions[2,51])

Cost function=((-Y)*log(h)-(1-Y)*log(1-h))/m
m=no. of training samples

sigma3=Error for the output layer

sigma2=Error for the hidden layer
