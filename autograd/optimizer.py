class GradientDescent:
    def __init__(self, n_epochs, loss_fn, model, learning_rate=1e-4):
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.model = model
        self.learning_rate = learning_rate

    def step(self, param):
        param.data -= self.learning_rate * param.grad

    def zero_grad(self, param):
        param.grad = 0

    def __call__(self, xis, yis):
        for i in range(self.n_epochs):
            ypis = [[self.model(xi)] for xi in xis]
            loss = self.loss_fn(y_true=yis, y_pred=ypis)
            loss.backward()

            for j, param in enumerate(self.model.parameters()):
                param.data -= self.learning_rate * param.grad
                param.grad = 0

            if i%20 == 0:
                print(f"Epoch {i+1}: {loss.data:.4f}")
        print(f"Epoch {i+1}: {loss.data:.4f}")

