import numpy as np
import pickle

class Network:
    def __init__(self, layers, alpha=0.001, gamma=0.9):
        '''
            Arguments:
            layers: array of layer sizes (includes non-hidden layers)
            alpha: learning rate
            gamma: momentum constant (using nesterov's momentum)
        '''

        # constants
        self.layers = layers
        self.alpha = alpha
        self.gamma = gamma

        # paramaters to update
        self.W = [] # list of matrices of weights for each layer
        self.B = [] # list of arrays of biases for each layer

        # initialize layers and set up random weights/biases
        self.init_layers()

    def init_layers(self):

        # iterate through layers, create weights matrix and bias array
        # of random values
        for l in range(len(self.layers)-1):

            # initialize weight
            w = np.random.randn(self.layers[l+1], self.layers[l])
            self.W.append(w/np.sqrt(self.layers[l]))

            # initialize bias
            b = np.random.randn(self.layers[l+1])
            self.B.append(b)


    def feedforward(self, a):
        '''
            Fills self.preactivations and self.activations which represent node values
            before and after activation function respectively

            Arguments:
            - a: np.array (2D) of inputs to first layer
            Returns:
            - activations of final layer
        '''

        # initialize activations and preactivations with first layer
        self.activations = [a]    # post ReLU
        self.preactivations = [a] # pre ReLU


        # iterate through layers, update values
        for layer in range(len(self.W)):

            # get next layer values before activation, add to preactivations
            net = self.W[layer]@self.activations[layer] + self.B[layer]
            self.preactivations.append(net)

            # apply activation function, add to activations
            net = self.leaky_ReLU(net)  # apply activation func
            self.activations.append(net)

        return self.activations[-1]


    def backpropagate(self, X_sample, y_sample):
        '''
            Backpropogate through layers to collect gradients of parameters

            Arguments:
            - X_sample: np.array 2D of inputs, single sample
            - y_sample: np.array 2D of expected output, single sample
            Returns:
            - dW: list[np.array] represents gradients to apply to self.W weights
            - dB: list[np.array] represents gradients to apply to self.B weights
        '''

        # first feedforward to set activations and preactivations
        _ = self.feedforward(X_sample)

        # initialize lists that will be returned
        delta_weights = []
        delta_bias = []

        # get cost
        cost = self.activations[-1] - y_sample

        # initialize first value of delta
        # (represents vector dC/dZ where Z is preactivations of a given layer)
        delta = cost*self.leaky_ReLU_deriv(self.preactivations[-1])
        dW = np.outer(delta, self.activations[-2])

        # add dW and dB to arrays (dB = delta)
        delta_weights.append(dW)
        delta_bias.append(delta)

        # loop through rest of layers in reverse
        for layer in range(len(self.layers)-2, 0, -1):

            # update delta
            delta = self.leaky_ReLU_deriv(self.preactivations[layer])*(self.W[layer].T.dot(delta))

            # get dW
            dW = np.outer(delta, self.activations[layer-1])

            # add dW and dB to arrays
            delta_weights.append(dW)
            delta_bias.append(delta)

        # reverse since we added deltas in reverse order
        delta_weights.reverse()
        delta_bias.reverse()

        return delta_weights, delta_bias


    def fit(self, X, y, epochs=50, batch_size=10, verbose=False,
            max_grad=1000, loss_end=0.0001):
        '''
            Fit model to training examples
            Arguments:
            - X: np array of training inputs
            - Y: np array of training outputs
            - epochs: Number of times to iterate through all training examples
            - batch_size: Size of batches to average gradient over
            - verbose: Boolean for outputting loss each epoch
            - max_grad: upper and lower limit on gradient
            - loss_end: loss value that halts program when reached
            Returns:
            - loss_history: list of losses over each epoch
        '''

        # initialize loss history and number of samples
        loss_history = []
        S = len(X)

        # ensure enough dimensions to X
        if X.ndim==1:
            X = np.atleast_2d(X).T

        # initialize previous parameters (used in momentum calculation)
        self.W_last = self.W.copy()
        self.B_last = self.B.copy()

        # iterate through epochs
        for e in range(epochs):

            # shuffle data
            shuffled_indices = np.random.permutation(S)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            # iterate over batches, track loss of batch
            batch_losses = []
            for i in range(0, S, batch_size):

                # get current batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # calculate and store batch loss
                preds = [self.feedforward(x) for x in X_batch]
                batch_loss = np.mean([(pred - y_act) ** 2 for pred, y_act in zip(preds, y_batch)])
                batch_losses.append(batch_loss)

                # initialize aggregated gradients
                dW_aggr = [np.zeros_like(w) for w in self.W]
                dB_aggr = [np.zeros_like(b) for b in self.B]

                # store momentum
                W_mm = [self.gamma*(w_curr - w_last) for w_curr, w_last in zip(self.W, self.W_last)]
                B_mm = [self.gamma*(b_curr - b_last) for b_curr, b_last in zip(self.B, self.B_last)]

                # set previous parameters to current ones
                self.W_last = self.W.copy()
                self.B_last = self.B.copy()

                # update parameters with momentum
                self.W = [w+w_mm for w, w_mm in zip(self.W, W_mm)]
                self.B = [b+b_mm for b, b_mm in zip(self.B, B_mm)]

                # backpropogate over each sample in batch, store aggregatred grads
                for X_sample, y_sample in zip(X_batch, y_batch):
                    dW, dB = self.backpropagate(X_sample, y_sample)
                    dW_aggr = [aggr_dw + dw for aggr_dw, dw in zip(dW_aggr, dW)]
                    dB_aggr = [aggr_db + db for aggr_db, db in zip(dB_aggr, dB)]

                # average gradients and apply learning rate
                dW_aggr = [self.alpha*dw / len(X_batch) for dw in dW_aggr]
                dB_aggr = [self.alpha*db / len(X_batch) for db in dB_aggr]

                # clip gradients
                dW_final = [np.clip(dw, -max_grad, max_grad) for dw in dW_aggr]
                dB_final = [np.clip(db, -max_grad, max_grad) for db in dB_aggr]

                # Apply final gradient to
                self.W = [w - dW for w, dW in zip(self.W, dW_final)]
                self.B = [b - dB for b, dB in zip(self.B, dB_final)]

            # store loss history of the epoch
            loss_history.append(np.mean(batch_losses))
            if verbose:
                print(f"Epoch {e + 1}: Loss = {round(loss_history[-1], 5)}")

            # early return if loss is below loss_end threshold
            if loss_history[-1]<loss_end:
                return loss_history

        return loss_history


    def predict(self, X):
        if X.ndim==1:
            X = np.atleast_2d(X).T
        return [self.feedforward(x) for x in X]


    def test_accuracy(self, X_test, y_test, model_type='binary'):
        if X_test.ndim == 1:
            X_test = np.atleast_2d(X_test).T
        preds = [self.feedforward(x) for x in X_test]

        if model_type=='binary':
            preds = [1 if pred[0]>0.5 else 0 for pred in preds]

            corr = 0
            for pred, act in zip(preds, y_test):
                if pred==act:
                    corr += 1

            return corr/len(preds)

        elif model_type=='categorical':
            corr = 0
            for pred, act in zip(preds, y_test):
                num_pred = np.argmax(pred)
                if act[num_pred]>0:
                    corr += 1

            return corr/len(preds)

    def ReLU(self, array):
        return np.maximum(0, array)

    def leaky_ReLU(self, array):
        return np.where(array>0, array, 0.1*array)
    def leaky_ReLU_deriv(self, array):
        return np.where(array>0, 1, 0.1)


    def ReLU_deriv(self, array):
        return np.where(array>0, 1, 0)

    def save_model(self, path):
        ''' Save model to path using pickle '''

        model_params = {'weights': self.W, 
                        'biases': self.B}
        
        with open(path, 'wb') as file:
            pickle.dump(model_params, file)

    def load_model(self, path):
        ''' Load model from path using pickle '''

        with open(path, 'rb') as file:
            model_params = pickle.load(file)
        
        self.W = model_params['weights']
        self.B = model_params['biases']