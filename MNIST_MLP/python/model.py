from exceptions import ShapeMismatchError
import numpy as np

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
        self.output = None
        self.del_bias = None

    def forward(self, X):
        z1 = np.matmul(X, self.weights) + self.biases
        return z1

    def __call__(self, X):
        return self.forward(X)

class ShapeMismatchError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

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
        self.parameters = [[layer.weights, layer.biases] for layer in self.layer_list]

    def forward(self, X, activation, final_activation):
        out = X.copy()
        for layernum in range(self.num_layers - 1):
            layer = self.layer_list[layernum]
            out = layer(out)
            out = activation(out)
            layer.output = out

        out = self.layer_list[-1](out)
        self.layer_list[-1].output = out

        if final_activation:
            out = final_activation(out)
            self.layer_list[-1].output = out
        return out

    def __call__(self, X, activation, final_activation=None):
        return self.forward(X, activation, final_activation)

    def backward(self, train_loss, probs, one_hot_encoded, epoch, batch_num, total_batches, batch_input, beta1, beta2, alpha, eps):
        original_weights = [layer.weights.copy() for layer in self.layer_list]
        for index, layer in enumerate(reversed(self.layer_list)):

            original_index = len(self.layer_list) - 1 - index

            if layer.final:
                del_bias = probs - one_hot_encoded
            else:
                prev_layer_weights = original_weights[original_index + 1]
                del_bias = np.matmul(self.layer_list[original_index+1].del_bias, prev_layer_weights.T)
                del_bias[self.layer_list[original_index].output==0] = 0

            layer.del_bias = del_bias

            if not layer.first:
                del_l_del_final = np.dot(self.layer_list[original_index-1].output.T, del_bias)
            else:
                del_l_del_final = np.dot(batch_input.T, del_bias)

            np.clip(del_l_del_final, -5, 5, out=del_l_del_final)

            t = epoch*(total_batches) + batch_num  + 1

            layer.mom_weights = layer.mom_weights*beta1 + (1-beta1)*del_l_del_final
            mom_cap_weights = layer.mom_weights/(1-beta1**t)
            layer.vel_weights = layer.vel_weights*beta2 + (1-beta2)*del_l_del_final**2
            vel_cap_weights = layer.vel_weights/(1-beta2**t)

            layer.weights = layer.weights - alpha*((mom_cap_weights)/(np.sqrt(vel_cap_weights) + eps))

            layer.mom_bias = layer.mom_bias*beta1 + (1-beta1)*np.sum(del_bias, axis=0)
            mom_cap_bias = layer.mom_bias/(1-beta1**t)
            layer.vel_bias = layer.vel_bias*beta2 + (1-beta2)*np.sum(del_bias**2, axis=0)
            vel_cap_bias = layer.vel_bias/(1-beta2**t)

            layer.biases = layer.biases - alpha*((mom_cap_bias)/(np.sqrt(vel_cap_bias) + eps))
            prev_layer_weights = layer.weights.copy()
