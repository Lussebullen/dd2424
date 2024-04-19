import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

# Constants
N = 10000   # Number of data points
K = 10      # Number of classes
D = 3072    # Number of dimensions

# Utility functions

def loadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('Datasets/cifar-10-batches-py/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def onehotencoding(labels):
    """
    One-hot encodes the given labels.
    
    Args:
        labels (np.array): The labels to be one-hot encoded.
        
    Returns:
        np.array: The one-hot encoded labels.
    """
    
    hot = np.zeros((K, N))
    for i in range(N):
        hot[labels[i]][i] = 1
    return hot

def load(fname):
    """
    Load the data from the given filename.
    
    Args:
        fname (str): The path to the file containing the data.
        
    Returns:
        dict: A dictionary containing the loaded data with the following keys:
            - "labels": The labels of the data.
            - "hot": The one-hot encoded labels.
            - "data": The data itself.
    """

    batch = loadBatch(fname)
    out = {}
    out["labels"] = batch[b"labels"]
    out["data"] = batch[b"data"].T.astype(float)
    out["hot"] = onehotencoding(out["labels"]) 
    
    return out

def transform(dataset):
    """
    Transforms the given dataset by normalizing the data.
    """
    data = dataset["data"]
    meanX = np.mean(data, axis=0)
    stdX = np.std(data, axis=0)
    dataset["data"] = (data - meanX) / stdX

class LinearLayer:
    def __init__(self):
        """
        Initializes the weights and biases of the layer.
        """
        self.W = np.random.normal(0,0.01,size=(K,D))
        self.b = np.random.normal(0,0.01,size=(K,1))
    
    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def evaluateClassifier(self, X, W, b):
        """
        Evaluate the classifier for a given input.

        Args:
            X (numpy.ndarray): Input data of shape (d, N), where d is the number of features and N is the number of samples.
            
        Returns:
            numpy.ndarray: Softmax probabilities of shape (K, N), where K is the number of classes and N is the number of samples.
        """
        s = W @ X + b
        return self.softmax(s)
    
    def lcross(self, x, y, W, b):
        """
        Calculates the cross-entropy loss for a single example.

        Args:
            x (numpy.ndarray): Input data of shape (d, N).
            y (numpy.ndarray): One-hot encoded true label of shape (K, N).

        Returns:
            float: Cross-entropy loss.
        """
        P = self.evaluateClassifier(x, W, b)
        return - y * np.log(P)
    
    def computeCost(self, X, Y, W, b, lmda):
        """
        Compute the cost function for linear regression with regularization.

        Args:
            X (numpy.ndarray): Input data of shape (d, N).
            Y (numpy.ndarray): One-hot encoded true label of shape (K, N).
            lmda (float): Regularization parameter.

        Returns:
        float: The computed cost.

        """
        reg_term = lmda * np.sum(W ** 2)

        loss_cross = self.lcross(X, Y, W, b)

        return 1 / X.shape[1] * np.sum(loss_cross) + reg_term, np.sum(loss_cross)
    
    def computeAcc(self, X, Y, W, b):
        """
        Compute the accuracy of the classifier.

        Args:
            X (numpy.ndarray): Input data of shape (d, N).
            Y (numpy.ndarray): One-hot encoded True label of shape (N,).

        Returns:
            float: Accuracy of the classifier.

        """
        P = self.evaluateClassifier(X, W, b)
        pred = np.argmax(P, axis=0)
        return np.mean(pred == np.argmax(Y, axis=0))
    
    def computeGrads(self, X, Y, lmbd):
        """
        Compute the gradients of the cost function with respect to the parameters.

        Args:
            X (numpy.ndarray): Input data of shape (d, N).
            Y (numpy.ndarray): One-hot encoded true label of shape (K, N).
            lmbd (float): Regularization parameter.

        Returns:
            list: A list containing the gradients of the cost function with respect to the weight matrix W and the bias vector b.
        """
        
        P = self.evaluateClassifier(X, self.W, self.b)
        G = P - Y
        gradB = 1 / X.shape[1] * np.sum(G, axis = 1)
        
        gradW = 1 / X.shape[1] * G @ X.T + 2 * lmbd * self.W

        return [gradW, gradB.reshape(K,1)]
    
    def ComputeGradsNumSlow(self, X, Y, lmbd, h):
        """ Converted from matlab code """
        no 	= 	self.W.shape[0]

        grad_W = np.zeros(self.W.shape)
        grad_b = np.zeros((no, 1))
        
        for i in range(len(self.b)):
            b_try = np.array(self.b)
            b_try[i] -= h
            c1 = self.computeCost(X, Y, self.W, b_try, lmbd)[0]

            b_try = np.array(self.b)
            b_try[i] += h
            c2 = self.computeCost(X, Y, self.W, b_try, lmbd)[0]

            grad_b[i] = (c2-c1) / (2*h)

        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                W_try = np.array(self.W)
                W_try[i,j] -= h
                c1 = self.computeCost(X, Y, W_try, self.b, lmbd)[0]

                W_try = np.array(self.W)
                W_try[i,j] += h
                c2 = self.computeCost(X, Y, W_try, self.b, lmbd)[0]

                grad_W[i,j] = (c2-c1) / (2*h)

        return [grad_W, grad_b.reshape(K,1)]
    
    def relerr(self, ga, gn, eps=1e-6):
        """
        Calculates the relative error between two vectors.

        Args:
            ga (numpy.ndarray): Analytical gradient.
            gn (numpy.ndarray): Numerical gradient.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

        Returns:
            float: The relative error between ga and gn.
        """
        
        diff = np.linalg.norm(ga - gn)
        norma = np.linalg.norm(ga)
        normn = np.linalg.norm(gn)
        numer = max(eps, norma + normn)
        return diff / numer
    
    def miniBatchGD(self, train, lmbd=0.1, n_batch=100, eta=0.001, n_epochs=20, val=None):
        """
        Perform mini-batch gradient descent.

        Args:
            X (numpy.ndarray): Input data of shape (d, N).
            Y (numpy.ndarray): One-hot encoded true label of shape (K, N).
            W (numpy.ndarray): Weight matrix of shape (K, d).
            b (numpy.ndarray): Bias vector of shape (K, 1).
            lmbd (float, optional): Regularization parameter. Defaults to 0.1.
            n_batch (int, optional): Number of mini-batches. Defaults to 100.
            eta (float, optional): Learning rate. Defaults to 0.001.
            n_epochs (int, optional): Number of epochs. Defaults to 20.

        Returns:
            tuple: A tuple containing the weight matrix W and the bias vector b.
        """
        X = train["data"]
        Y = train["hot"]

        costs = {"train" : [], "val" : []}
        loss = {"train" : [], "val" : []}
        accs = {"train" : [], "val" : []}
        
        for epoch in tqdm(range(n_epochs)):
            permutation = np.random.permutation(X.shape[1])
            X = X[:, permutation]
            Y = Y[:, permutation]

            for j in range(int(X.shape[1] / n_batch)):
                j_start = j * n_batch
                j_end = (j + 1) * n_batch
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]
                gradW, gradB = self.computeGrads(X_batch, Y_batch, lmbd)
                self.W -= eta * gradW
                self.b -= eta * gradB
            c, l = self.computeCost(X, Y, self.W, self.b, lmbd)
            costs["train"].append(c)
            loss["train"].append(l)
            accs["train"].append(self.computeAcc(X, Y, self.W, self.b))
            if val:
                c, l = self.computeCost(val["data"], val["hot"], self.W, self.b, lmbd)
                costs["val"].append(c)
                loss["val"].append(l)
                accs["val"].append(self.computeAcc(val["data"], val["hot"], self.W, self.b))
        return self.W, self.b, costs, loss, accs

def plotResults(title, costs, loss, accs, test_acc = None):
    plt.figure(figsize=(16, 6))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.plot(costs["train"], label="Training")
    plt.plot(costs["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(loss["train"], label="Training")
    plt.plot(loss["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(accs["train"], label="Training")
    plt.plot(accs["val"], label="Validation")
    if test_acc:
        plt.axhline(test_acc, color="red", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()    
    plt.savefig(f'Images/{title}.png')

def genWeightImage(slice):
    """
    Generates an image from the given slice of the weight matrix.
    
    Args:
        slice (numpy.ndarray): The slice of the weight matrix.
        
    Returns:
        numpy.ndarray: The generated image.
    """
    
    img = slice.reshape(32, 32, 3, order="F")
    img = img - np.min(img)
    img = img / np.max(img)
    return img

def genMatrixVisualization(title, W):
    plt.figure(figsize=(12, 6))

    for i in range(K):
        plt.subplot(2, 5, i+1)
        plt.imshow(genWeightImage(W[i, :]))
        plt.title(f"Slice {i}")

    plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(f'Images/{title}.png')

def main():
    # Load and normalize datasets
    train = load("data_batch_1")
    val = load("data_batch_2")
    test = load("test_batch")
    transform(train)
    transform(val)
    transform(test)

    # Check analytical gradients
    lin = LinearLayer()
    gnw, bnb = lin.ComputeGradsNumSlow(train["data"][:,:20], train["hot"][:,:20], 0.3, 1e-6)
    gaw, gab = lin.computeGrads(train["data"][:,:20], train["hot"][:,:20], 0.3)
    print("Relative error between analytical and numerical gradients:")
    print(lin.relerr(gaw, gnw))
    print(lin.relerr(gab, bnb))

    # Hyperparameters [lambda, n_epochs, n_batch, eta]
    settings = [[0, 40, 100, 0.1],
                [0, 40, 100, 0.001],
                [0.1, 40, 100, 0.001],
                [1, 40, 100, 0.001]
                ]
    
    test_accs = []

    for i, params in enumerate(settings):
        lmbd, n_epochs, n_batch, eta = params
        lin = LinearLayer()
        W, b, costs, loss, accs = lin.miniBatchGD(train, lmbd=lmbd, n_batch=n_batch, eta=eta, n_epochs=n_epochs, val=val)
        test_acc = lin.computeAcc(test["data"], test["hot"], W, b)
        test_accs.append(test_acc)
        plotResults(f"paramset{i}results", costs, loss, accs, test_acc)
        genMatrixVisualization(f"paramset{i}visuals", W)
    
    print("Test accuracies:")
    print(test_accs)

if __name__ == "__main__":
    main()