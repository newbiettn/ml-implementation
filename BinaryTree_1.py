####################################################################################################
# BinaryTree_1.py
# AUTHOR:           NGOC TRAN
# CREATED:          12 Apr 2021
# DESCRIPTION:      Classes to represent and visualise decision tree (see decisiontree_1.py). In
#                   this implementation, I use ete3 package to support tree visualisation in
#                   ASCII format (see http://etetoolkit.org/docs/latest/tutorial/tutorial_trees
#                   .html).
####################################################################################################
# Load libraries
from ete3 import Tree


# Classes
class Node:
    """
    A tree node object
    """

    def __init__(self,
                 name="",
                 category=None,
                 data=None,
                 gini=None,
                 lnode=None,
                 rnode=None):
        """
        :param name:
        :param category:
        :param data:
        :param gini:
        :param lnode:
        :param rnode:
        """
        self._gini = gini
        self._category = category
        self._data = data
        self._name = name
        self._rnode = rnode
        self._lnode = lnode

    def traverse(self):
        """
        Traverse all sub-nodes
        :return: a representation of the tree in the Newick format.
        """
        s = ""
        print(s)
        if self._lnode is not None:
            s = ")" + s
            s = self._lnode.traverse() + s
        if self._rnode is not None:
            s = "," + s
            s = self._rnode.traverse() + s
        if self._lnode is not None:
            s = "(" + s
        s = s + self._name
        return s


class BinaryTree:
    """
    Represent binary tree objects
    """

    def __init__(self, root):
        """
        :param root: root of the tree
        """
        self._root = root

    def traverse(self):
        """
        Print the tree in the Newick format
        See http://etetoolkit.org/docs/latest/tutorial/tutorial_trees.html
        """
        return self._root.traverse() + ";"

    def visualize(self):
        """
        Visualise the tree in ASCII format
        :return:
        """
        s = self.traverse()
        t = Tree(s, format=1)
        print(t.get_ascii(attributes=["name", "label", "complex"]))


# Test
# Define nodes
a_node = Node(name='A')
a1_node = Node(name='A1')
a2_node = Node(name='A2')
b_node = Node(name='B')
b1_node = Node(name='B1')
b2_node = Node(name='B2')
b11_node = Node(name='B11')
b12_node = Node(name='B12')
b111_node = Node(name='B111')
b112_node = Node(name='B112')
root = Node(name='root', rnode=a_node, lnode=b_node)

a_node._lnode = a1_node
a_node._rnode = a2_node
b_node._lnode = b1_node
b_node._rnode = b2_node
b1_node._lnode = b11_node
b1_node._rnode = b12_node
b11_node._lnode = b111_node
b11_node._rnode = b112_node

# Define tree
tree = BinaryTree(root=root)
tree.visualize()