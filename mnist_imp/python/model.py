from exceptions import ShapeMismatchError

class LinearLayer:
    def __init__(self, d_inp, d_out):
        self.d_inp = d_inp
        self.d_out = d_out
        self.weights = np.random.randn(self.d_inp, self.d_out) * np.sqrt(2 / self.d_inp)
        self.biases = np.zeros(self.d_out)
        self.first = False
        self.intermediate = False
        self.final = False
        self.mom_weights = np.zeros((self.d_inp, self.d_out))
        self.mom_bias = np.zeros((self.d_out))
        self.vel_weights = np.zeros((self.d_inp, self.d_out))
        self.vel_bias = np.zeros((self.d_out))

    def forward(self, X):
        z1 = np.matmul(X, self.weights) + self.biases
        return z1

    def __call__(self, X):
        return self.forward(X)

class MLP:
    def __init__(self, *layers):
        self.layer_list = []
        for layernum, layer in enumerate(layers):

            if layernum == 0:
                layer.first = True
            elif layernum == len(layers) - 1:
                layer.final = True
            else:
                layer.intermediate = True

            self.layer_list.append(layer)
        self.num_layers = len(self.layer_list)

        for layernum in range(self.num_layers - 1):
            layerprev, layernext = self.layer_list[layernum], self.layer_list[layernum + 1]
            if layerprev.d_out != layernext.d_inp:
                raise ShapeMismatchError(f"Shape mismatch between output of layer {layernum}: {layerprev.d_out} and layer {layernum + 1}: {layernext.d_inp}")

    def forward(self, X, activation, final_activation=None):
        out = X.copy()
        for layernum in range(self.num_layers - 1):
            layer = self.layer_list[layernum]
            out = layer(out)
            out = activation(out)
        out = self.layer_list[-1](out)
        if final_activation:
            out = final_activation(out)
        return out

    def __call__(self, X, activation):
        return self.forward(X, activation)
