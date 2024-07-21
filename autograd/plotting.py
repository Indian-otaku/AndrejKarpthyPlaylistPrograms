from graphviz import Digraph


def trace(root):
    nodes = set()
    edges = set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for operand in v.operands:
                edges.add((operand, v))
                build(operand)
    
    build(root)
    return nodes, edges

def draw_dot(root, graph_format="svg", graph_attr={'rankdir': 'TB'}):
    dot = Digraph(
        name="Tensor graph",
        format=graph_format,
        graph_attr=graph_attr
    )
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"{n.label}|{n.data:.4f}|{n.grad:.4f}", shape="record")
        if n.operation:
            dot.node(name=uid+n.operation, label=f"{n.operation}")
            dot.edge(uid+n.operation, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2))+n2.operation)

    return dot