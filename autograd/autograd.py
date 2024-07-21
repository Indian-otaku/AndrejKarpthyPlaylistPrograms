import math
import random


class Element():
    element_id = 0

    def __init__(self, data, label=None, operands=(), operation=""):
        assert isinstance(data, (int, float)), "Only supports int/float type"
        self.data = data
        self.grad = 0.0
        self.backward_ = lambda: None
        self.label = label if label else self._label_node()
        self.operands = operands
        self.operation = operation

    def _label_node(self):
        Element.element_id += 1
        return f"e_{Element.element_id}"

    def __repr__(self) -> str:
        return f"Element(\n\tdata={self.data},\n\tlabel={self.label}\n)"

    def __add__(self, other):
        other = other if isinstance(other, Element) else Element(other, label=str(other))
        out = Element(self.data + other.data, label=f"({self.label}+{other.label})", operands=(self, other), operation="+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward_ = _backward
        return out

    def __radd__(self, other):
        other = other if isinstance(other, Element) else Element(other, label=str(other))
        out = self + other
        out.label = f"({other.label}+{self.label})"
        return out    
    
    def __sub__(self, other):
        other = other if isinstance(other, Element) else Element(other, label=str(other))
        out = Element(self.data - other.data, label=f"({self.label}-{other.label})", operands=(self, other), operation="-")
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out.backward_ = _backward
        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, Element) else Element(other, label=str(other))
        out = other - self
        out.label = f"({other.label}-{self.label})"
        def _backward():
            self.grad -= out.grad
            other.grad += out.grad
        out.backward_ = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Element) else Element(other, label=str(other))
        out = Element(self.data * other.data, label=f"({self.label}*{other.label})", operands=(self, other), operation="*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward_ = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Element) else Element(other, label=str(other))
        out = Element(self.data / other.data, label=f"({self.label}/{other.label})", operands=(self, other), operation="/")
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data * (other.data ** (-2))) * out.grad
        out.backward_ = _backward
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Element) else Element(other, label=str(other))
        out = other / self
        out.label = f"({other.label}/{self.label})"
        def _backward():
            self.grad += (-other.data * (self.data ** (-2))) * out.grad
            other.grad += (1 / self.data) * out.grad
            out.backward_ = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supports power of int/float type"
        other = Element(other, label=str(other))
        def _backward():
                self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
        out = Element(self.data ** other.data, label=f"({self.label}**{other.label})", operands=(self, other), operation="**")
        out.backward_ = _backward
        return out
    
    def tanh(self):
        out = Element(math.tanh(self.data), label=f"tanh({self.label})", operands=(self,), operation="tanh")
        def _backward():
            self.grad += (1 - (out.data ** 2)) * out.grad
        out.backward_ = _backward
        return out
    
    def sigmoid(self):
        out = Element(1 / ( 1 + math.exp(-self.data)), label=f"sigmoid({self.label})", operands=(self,), operation="sigmoid")
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out.backward_ = _backward
        return out
    
    def relu(self):
        out = Element(max(0, self.data), label=f"relu({self.label})", operands=(self,), operation="relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out.backward_ = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def exp(self):
        out = Element(math.exp(self.data), label=f"exp({self.label})", operands=(self,), operation="exp")
        def _backward():
            self.grad += out.data * out.grad
        out.backward_ = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for operand in v.operands:
                    build_topo(operand)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node.backward_()
    


class Neuron:
    def __init__(self, nin, activation="tanh", neuron_no=0, layer_no=0):
        self.nin = nin
        self.activation = activation
        self.layer_no = layer_no
        self.neuron_no = neuron_no
        self.w = [Element(random.uniform(-1, 1), label=f"w[{layer_no},{neuron_no},{i}]") for i in range(nin)]
        self.b = Element(random.uniform(-1, 1), label=f"b[{layer_no},{neuron_no}]")
    
    def __call__(self, x):
        assert len(x) == self.nin, "Length of input must match input size of neuron"
        act = sum((wi * xi for wi, xi in zip(self.w, x)),start=self.b)
        if self.activation == "tanh":
            act = act.tanh()
        elif self.activation == "sigmoid":
            act = act.sigmoid()
        elif self.activation == "relu":
            act = act.relu()
        act.label = f"Neuron[{self.layer_no},{self.neuron_no}]"
        return act
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, activation="tanh", layer_no=0):
        self.nin = nin
        self.nout = nout
        self.activation = activation
        self.neurons = [Neuron(nin, activation=activation, neuron_no=i, layer_no=layer_no) for i in range(nout)]

    def __call__(self, x):
        assert len(x) == self.nin, "Length of input must match input size of layer"
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) > 1 else outs[0]
        
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts, activation="tanh"):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation=activation, layer_no=i) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    

