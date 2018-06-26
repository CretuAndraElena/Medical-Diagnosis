import ID3Algorithm as ID3
import networkx as nx

tree=ID3.id3.tree
preOrderList=[]

def preorder(tree):
    if isinstance(tree,ID3.Leaf):
        pass
    if isinstance(tree,ID3.Decision_Node):
        preOrderList.append((str(tree),str(tree.false_branch)))
        preOrderList.append((str(tree),str(tree.true_branch)))
        preorder(tree.false_branch)
        preorder(tree.true_branch)

preorder(tree)

g=nx.DiGraph()
g.add_edges_from(preOrderList)
p=nx.drawing.nx_pydot.to_pydot(g)
p.write_png('example.png')