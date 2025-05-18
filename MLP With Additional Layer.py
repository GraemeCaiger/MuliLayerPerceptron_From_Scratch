import numpy as np
from sklearn import datasets
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg'
import matplotlib.pyplot as plt


# load iris dataset
iris = datasets.load_iris()
X = iris["data"]  # Retrives all 4 colomns of data
y = (iris["target"] == 2).astype(int)  # Gets the target array for when Iris-Virginica, uses as type int to convert boolian to 1 and 0.
print("X has shape:")
print(X.shape)
print("y= ")
print(y) # Just 150 values in a flat array
print(y.shape)
y = y.reshape([150,1]) # 2D array for compatibity with other functions. 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)



class MLP:
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, learning_rate: float = 0.01):  # note int statments etc 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  # List of sizes of hidden layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # List to store weights and biases for each layer 
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        prev_size = self.input_size  # Starting with the input size
        
        # Create weights and biases for hidden layers
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size))  # Random weights between previous layer and current hidden layer (For first itteration this is input to 1st hidden)
            self.biases.append(np.zeros((1, hidden_size)))  # Biases for the nth hidden layer (all at zero to start). 
            prev_size = hidden_size  # Update the previous size to current hidden layer size

        # Create weights and biases for the output layer
        self.weights.append(np.random.randn(prev_size, self.output_size))  # Weights between last hidden layer and output
        self.biases.append(np.zeros((1, self.output_size)))  # Biases for the output layer
        
        # Print the initial shapes of the weights and biases
        print("The Initial Shape of the MLP is:")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Layer {i+1} - Weights shape: {w.shape}, Biases shape: {b.shape}")
    
    def mean_squared_error(self,y_true,y_pred):
        return np.mean(0.5 * np.square(y_true - y_pred))
    
    ###* Training the Neural network ~~~~~~~~~~~~~~
    def fit_Batch(self, X, y, epochs=1000):
        costs = []
        for epoch in range(epochs):
            # *FEEDFORWARD ~~~~>
            z_list = []
            A = [X] # Initialises the first layer of activations to be the Input. First element same shape as X (150,4). Lists can contain items of varied shape So for example: 
            # So if number of nerons per layer Likeso : Input [4] , Hidden1 [5], Hidden2 [3], Output [1]
            # Then A = [ A[0]        # (150,4)
            #            A[1]        # (150,5)
            #            A[2]        # (150,3)
            #            A[3]        # (150,1)
            #           ] 
            # Note, this variable shape of elemnts feature is not shared by numpy arrays. This is still a fine use of a list though as the calculations are done on the numpy array elements within the list. 
            for w, b in zip(self.weights, self.biases):  # Zip combines itterables. Index is the same for both A and B each itteration. 
                z = A[-1].dot(w) + b  # A[-1] gives you the last item in the A array. If this were A[-2] it would be the second to last etc 
                z_list.append(z)
                activation = sigmoid(z)
                A.append(activation) # Shape depends on shape of the w matrix dotted with.  A[n](150,4) • W(Input_size,Output_size) => A[n+1](150,Output_size)      

            # *BACKPROPERGATION <~~~~
            #  [Images/d_weight_and_Biases equations.png] 
            # $$C = \frac{1}{2}(y-a)^2 $$

            delta = (A[-1] - y) * sigmoid_derivative(z[-1])  #! Sets the INITIAL value of $$\frac{\delta C}{\delta z_l} = \frac{\delta C}{\delta a_L}\frac{\delta a_L}{\delta z_L} = \delta = (a-y) \sigma^{\prime}(z) $$
            d_weights = []                                   # Same shape as z 
            d_biases =  []

            for l in reversed(range(len(self.weights))): # range(len(self.weights)) creates a sequence of layer indices. For example if there were 3 weight matracies this would be [0, 1, 2] 
                a_prev = A[l]                            # reversed reverses the order of this list giving last to first indexes of the array [2,1,0]. l takes on value of the index.
                dw = a_prev.T.dot(delta)  # $$dw = a \cdot \delta $$
                db = np.sum(delta, axis=0, keepdims=True) # $$db =  \Sigma \delta $$
                d_weights.insert(0, dw)
                d_biases.insert(0, db)

                # From Previous EQ $$\frac{\delta a_L}{\delta z_L} = \sigma^{\prime}  $$  Only depends on what layer your in remains the same regardless of considering additional layers but $$\frac{\delta C}{\delta a_l} $$ requires answering to the layer that preceded it 
                # 
                # recentering L at the layer beforw  gives  $$C(a_{L+1}(z_{L+1}(a_{L})))$$  $$\implies$$ $$\frac{\delta C}{\delta a_l} = (\frac{\delta C}{\delta z_{L+1}} \frac{\delta z_{L+1}}{\delta a_{L}}) $$  Hence for this layer before "L" $$\implies$$ $$ \frac{\delta C}{\delta z} = (\frac{\delta C}{\delta z_{L+1}} \frac{\delta z_{L+1}}{\delta a_{L}})\frac{\delta a_L}{z_L} $$   This expression simplifies to $$ \delta_L = (\delta_{L+1} \cdot W_{L+1}^T) \cdot \sigma^{\prime}(z_L) $$
                if l > 0:
                    delta = delta.dot(self.weights[l].T) * sigmoid_derivative(z[l - 1]) #! Recursively update delta: $$ \delta $$ for the next layer (i.e., one layer earlier)  l index esentially L+1
            
            #* Update weights and biases ---
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * d_weights[i]
                self.biases[i] -= self.learning_rate * d_biases[i]

            #* Compute cost and store it in costs list
            cost = self.mean_squared_error(y, A[-1])
            costs.append(cost)

            if epoch % 100 == 0:  # print the cost every 100 epochs
                print(f"Epoch {epoch}, Cost: {cost}")
        #* Plotting the cost function
        plt.plot(range(epochs), costs)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function Improvement Over Epochs')
        plt.show()

    def predict(self, X):
        layer1 = X.dot(self.weights1) + self.bias1
        activation1 = sigmoid(layer1)
        layer2 = activation1.dot(self.weights2) + self.bias2
        activation2 = sigmoid(layer2)
        return (activation2 > 0.5).astype(int)




mlp = MLP(input_size=4, hidden_sizes=[4,3,2], output_size=1)

# train the MLP on the training data
mlp.fit_Batch(X, y)

# make predictions on the test data
#y_pred = mlp.predict(X)

# evaluate the accuracy of the MLP
#accuracy = np.mean(y_pred == y)
#print(f"Accuracy: {accuracy:.2f}")