from pypuf.learner.base import Learner          # Perceptron super class
from pypuf.simulation.base import Simulation    # Perceptron return type

class LinearizationModel():
    """
    Helper class to linearize challenges of a k-XOR PUF.
    Instantiate using 'monomials' - a list of lists containing indices,
    which defines how to compute linearized variables from a challenge.
    Example: [[1,2,4],[1,6]] => X1 = C1*C2*C4; X2 = C1*C6
    """

    def __init__(self, monomials):
        """
        :param monomials: list of lists containing indices to compute x's
                          from challenge
        """
        # Monomials, defining how to build Xs from a challenge
        self.monomials = monomials

    def lin_old(self, inputs):
        """
        Convert array of challenges to Xs accoring to self.monomials.
        Param inputs has shape N, n - meaning N challenges of n bits.
        """
        N, n = inputs.shape
        out = np.empty(shape=(N, len(self.monomials)), dtype=np.int8)
        for idx, m in enumerate(self.monomials):
            out[:, idx] = np.prod(inputs[:, list(m)], axis=1)
        return out

    def linearize(self, inputs, block_size=10**5):
        N, n = inputs.shape
        try:
            linearized_challenges = empty(shape=(N, len(self.monomials)), dtype=int8)
        except MemoryError as e:
            print(e)
        monomial_matrix = zeros(shape=(len(self.monomials), n), dtype=float32)
        for idx, m in enumerate(self.monomials):
            monomial_matrix[idx, list(m)] = 1
        for start in range(0, len(inputs), block_size):
            inputs01 = .5 - .5 * inputs[start:start+block_size]
            linearized_challenges[start:start+block_size] = 1 - 2 * ((inputs01 @ monomial_matrix.T).astype(int8) % 2)
        return linearized_challenges

class NeuralRegression(Learner):

    def __init__(self, train_set, valid_set, monomials=None,
                 batch_size=64, epochs=1000, learning_rate=0.001, gpu_id=None):
        # Training parameters
        self.train_set = train_set
        self.valid_set = valid_set
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gpu_id = gpu_id

        # If no monomials are provided, use identity
        if monomials is None:
            monomials = [[i] for i in range(train_set.instance.n)]
        self.monomials = monomials

        # Model parameters
        self.input_len = len(self.monomials)
        self.model = None

        # Build linearization model
        linearizer = LinearizationModel(self.monomials)
        # Apply linearization to each row in numpy array
        self.linearize = linearizer.linearize

        # Debugging data
        self.history = None

    def prepare(self):
        """
        Construct and compile Perceptron.
        Called in self.learn().
        """
        import torch
        import torch.nn as nn
        from torch.utils.data.dataloader import DataLoader
        from torch.utils.data import TensorDataset

        # Convert challenges to linearized subchallenge representation
        print("Tranforming CRPs according to Monomials.")
        x = torch.from_numpy(self.linearize(self.train_set.challenges))
        y = torch.from_numpy(self.train_set.responses)
        x_valid = torch.from_numpy(self.linearize(self.valid_set.challenges))
        y_valid = torch.from_numpy(self.valid_set.responses)
        print("Done Tranforming!")
        ts = TensorDataset(x, y)
        vs = TensorDataset(x_valid, y_valid)

        self.train_loader = DataLoader(ts, self.batch_size,
                                  shuffle=True, pin_memory=True)
        self.valid_loader = DataLoader(vs, self.batch_size,
                                  shuffle=True, pin_memory=True)
        input_len = self.input_len
        class SoelterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.tensor(input_len, requires_grad=True)
            def forward(self, x):
                out = torch.prod(torch.mul(x, self.weights))
                return out
        self.model = SoelterModel()
        if self.gpu_id is not None:
            self.model.cuda(torch.device("cuda"))


    def loss_func(self, outputs, labels):
        l = torch.log(1 - torch.sigmoid(torch.mul(outputs, labels)))
        torch.cumsum(l)
    self.loss_func = loss_func


    def loss_batch(self, x, y, opt=None, metric=None):
        # Calculate loss
        preds = self.model(x)
        loss = self.loss_func(preds, y)
        if opt is not None:
            # Compute gradients
            loss.backward()
            # Update parameters
            opt.step()
            # Reset gradients
            opt.zero_grad()
        metric_result = None
        if metric is not None:
            # Compute the metric
            metric_result = metric(preds, y)
        return loss.item(), len(x), metric_result


    def evaluate(self):
        def pypuf_accuracy(y_true, y_pred):
            accuracy = (1 + torch.mean(torch.sign(y_true * y_pred))) / 2
            return accuracy
        metric = pypuf_accuracy
        with torch.no_grad():
            # Pass each batch through the model
            results = [self.loss_batch(x, y, metric=metric)
                       for x, y in self.valid_loader]
            # Separate losees, counts and metrics
            losses, nums, metrics = zip(*results)
            # Total size of the dataset
            total = np.sum(nums)
            # Avg. loss across batches
            avg_loss = np.sum(np.multiply(losses, nums)) / total
            avg_metric = None
            if metric is not None:
                # Avg. of metric across batches
                avg_metric = np.sum(np.multiply(metrics, nums)) / total
            return avg_loss, total, avg_metric


    def fit(self, verbose=True):
	opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
	for epoch in range(self.epochs):
	    # Training
	    for x, y in self.train_loader:
		loss,_,_ = self.loss_batch(self.model, x, y, opt)
	    # Evaluation
	    result = self.evaluate()
	    val_loss, total, val_metric = result
	    # Print progress
	    if verbose:
		print('Epoch [{}/{}], Loss: {:.4f}'
		      .format(epoch+1, self.epochs, val_loss))
	    else:
		print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'
		      .format(epoch+1, self.epochs, val_loss, "pypuf_accuracy", val_metric))

    # Executed by Experiment
    def learn(self):
        self.prepare()

        self.fit(verbose=True)

        def predict(chals):
            with torch.no_grad():
                x = torch.from_numpy(self.linearize(chals))
                y = torch.sign(self.model(chals))

        # Create Simulation object and return it
        sim = type('PerceptronSimulation', (Simulation,), {})
        sim.eval = predict
        sim.n = self.input_len
        return sim

