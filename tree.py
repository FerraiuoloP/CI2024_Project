import math
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt  
from functools import lru_cache


class NodeType(Enum):
    B_OP = 1
    U_OP = 2
    VAR = 3
    CONST = 4

class Node:
    def __init__(self, node_type, value=None):
        self.node_type = node_type
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        if self.node_type in {NodeType.B_OP, NodeType.U_OP}:
            return str(self.value.__name__) if callable(self.value) else str(self.value)
        elif self.node_type == NodeType.CONST:
            return str(int(round(self.value)))  #TODO: REMOVE round while printing
        else:    
            return str(self.value)
    def clone(self):
        """Creates a deep copy of the current node."""
        new_node = Node(self.node_type, self.value)
        new_node.left = self.left.clone() if self.left else None
        new_node.right = self.right.clone() if self.right else None
        return new_node
    

    def to_np_formula(self):
        if self.value is None:
            return None
        if self.node_type == NodeType.CONST:
            return str(self.value)
        if self.node_type == NodeType.VAR:
            return self.value
        if self.node_type == NodeType.U_OP:
            operand = self.left.to_np_formula() if self.right is None else self.right.to_np_formula()
            return f"np.{self.value.__name__}({operand})"
        if self.node_type == NodeType.B_OP:
            left = self.left.to_np_formula()
            right = self.right.to_np_formula()
            return f"np.{self.value.__name__}({left}, {right})"

class Tree:
    _memo_cache = {}
    _cache_limit = 1000  # Set the cache size limit
    def __init__(self, method="full",require_valid_tree=True,empty=False):
        #Valid tree means a tree that has a computable fitness (no division by zero, no overflow, etc.)
        
        self.fitness = np.inf
        vars = [f'x{i}' for i in range(Tree.n_var)]
        self.root = None
        while not empty and self.fitness == np.inf:
            if method == "full":
                self.root = self.create_random_tree_full_method(Tree.max_depth, vars)
            elif method == "grow":
                self.root = self.create_random_tree_grow_method(Tree.max_depth, vars)
            self.compute_fitness()
            if not require_valid_tree:
                break
            

    @staticmethod
    def set_params(unary_ops, binary_ops, n_var, max_const,max_depth, x_train, y_train, x_test=None, y_test=None):
        Tree.unary_ops = unary_ops
        Tree.binary_ops = binary_ops
        Tree.n_var = n_var
        Tree.max_const = max_const
        Tree.x_train = x_train
        Tree.y_train = y_train
        Tree.x_test = x_test
        Tree.y_test = y_test
        Tree.max_depth = max_depth
    
    @staticmethod
    def create_random_tree_full_method(depth, var_to_place=[]):
        leaves = []
        for var in var_to_place:
            leaves.append(Node(NodeType.VAR, value=var))
        while len(leaves) < 2 ** depth:
            leaves.append(Node(NodeType.CONST, value=(-Tree.max_const + (Tree.max_const - (-Tree.max_const)) * np.random.random())))

        def build_tree(nodes, current_depth):
            if current_depth == depth:
                return nodes.pop(0)

            node = Node(NodeType.B_OP, value=np.random.choice(Tree.binary_ops))
            node.left = build_tree(nodes, current_depth + 1)
            node.right = build_tree(nodes, current_depth + 1)
            return node

        np.random.shuffle(leaves)
        return build_tree(leaves, 0)

    @staticmethod
    def create_random_tree_grow_method(depth, var_to_place=[]):
        #create a tree with full method then prune randomly some branches that do not contain variables
        #TODO: fare in modo che la depth degli unary_ops non sia 1 (e fare il refactor di questo schifo :) )
        tree = Tree("full")
        # print("Albero full:")
        # tree.print_tree()
        # tree.add_drawing()
        nodes = tree.collect_nodes(tree.root)
        # nodes2=[node.clone() for node in nodes]
        
        for node in nodes:
            # print(node)
            var_in_subtree = Tree.find_var_in_subtree(node)
            if np.random.rand()>0.5  and node and node.node_type != NodeType.VAR and len(var_in_subtree)<2: #if the subtree has oly one variable we can prune it
                if(np.random.rand()>0.5):
                    node.node_type = NodeType.U_OP
                    node.value = np.random.choice(Tree.unary_ops)
                    if(len(var_in_subtree))==1:
                        node.left=Node(NodeType.VAR,value=var_in_subtree[0])
                    else:
                        var_left= (-Tree.max_const + (Tree.max_const - (-Tree.max_const)) * np.random.random())
                        node.left=Node(NodeType.CONST,value=var_left)

                    node.right = None
                    node.left.left=None
                    node.left.right=None
                else:
                    if(len(var_in_subtree))==1:
                        node.node_type = NodeType.VAR
                        node.value = var_in_subtree[0]
                   
                    else:
                        node.node_type = NodeType.CONST
                        node.value = (-Tree.max_const + (Tree.max_const - (-Tree.max_const)) * np.random.random())
                    node.left=None
                    node.right=None  
                    

                  
                
        return tree.root
    


    def print_tree(self):
        self.print_tree_recursive(self.root, 0)

    def print_tree_recursive(self, node, depth):
        if node is not None:
            print("  " * depth + f"{depth}-{str(node)}")  
            self.print_tree_recursive(node.left, depth + 1)
            self.print_tree_recursive(node.right, depth + 1)
        else:
            print("  " * depth + "None")
    
    def mutate_subtree(self):
        nodes = self.collect_nodes(self.root)
        index = np.random.randint(0, len(nodes))
        node_to_replace = nodes[index]
        depth_of_node = self.get_depth(self.root, node_to_replace)
        var_in_subtree = Tree.find_var_in_subtree(node_to_replace)
        min_depth = 0
        if var_in_subtree:  
            min_depth = math.ceil(math.log2(len(var_in_subtree)))  
        max_possible_depth = Tree.max_depth - depth_of_node
        subtree_depth = max(min_depth, 0) if min_depth >= max_possible_depth else np.random.randint(min_depth, max_possible_depth)
        new_subtree = Tree.create_random_tree_full_method(subtree_depth, var_in_subtree)
        node_to_replace.node_type = new_subtree.node_type
        node_to_replace.value = new_subtree.value
        node_to_replace.left = new_subtree.left
        node_to_replace.right = new_subtree.right

    def copy_tree(self):
        new_tree = Tree(empty=True)
        new_tree.root = self.root.clone()
        new_tree.fitness = self.fitness
        return new_tree

    def mutate_single_node(self):
        nodes = self.collect_nodes(self.root)
        nodes_to_mutate = [node for node in nodes if node.node_type != NodeType.VAR]
        node_to_mutate = np.random.choice(nodes_to_mutate)
        if node_to_mutate.node_type == NodeType.CONST:
            node_to_mutate.value = (-Tree.max_const + (Tree.max_const - (-Tree.max_const)) * np.random.random())
        elif node_to_mutate.node_type == NodeType.B_OP:
            node_to_mutate.value = np.random.choice(Tree.binary_ops)
        elif node_to_mutate.node_type == NodeType.U_OP:
            node_to_mutate.value = np.random.choice(Tree.unary_ops)
    
  

    @staticmethod
    def crossover(tree1, tree2):
        new_tree1 = Tree(empty=True)
        new_tree2 = Tree(empty=True)
        new_tree1.root = tree1.root.clone()
        new_tree2.root = tree2.root.clone()

        subtrees1 = new_tree1.find_subtree_without_var(new_tree1.root) #find subtrees without var so we can swap
        subtrees2 = new_tree2.find_subtree_without_var(new_tree2.root)

        if not subtrees1 or not subtrees2:
            return None,None
        
       
        subtree1 = np.random.choice(subtrees1)
        subtree2 = np.random.choice(subtrees2)

        subtree1.node_type, subtree2.node_type = subtree2.node_type, subtree1.node_type
        subtree1.value, subtree2.value = subtree2.value, subtree1.value
        subtree1.left, subtree2.left = subtree2.left, subtree1.left
        subtree1.right, subtree2.right = subtree2.right, subtree1.right

        return new_tree1, new_tree2

    def find_subtree_without_var(self, node):
        if node is None:
            return []
        subtrees = []
        if node.node_type != NodeType.VAR and not Tree.find_var_in_subtree(node):
            subtrees.append(node)
        subtrees += self.find_subtree_without_var(node.left)
        subtrees += self.find_subtree_without_var(node.right)
        return subtrees

    def collect_nodes(self, node):
        if node is None:
            return []
        return [node] + self.collect_nodes(node.left) + self.collect_nodes(node.right)
    
    @staticmethod
    def find_var_in_subtree(node):
        if node is None or node.node_type == NodeType.CONST:
            return []
        if node.node_type == NodeType.VAR:
            return [node.value]  
        var_l = Tree.find_var_in_subtree(node.left)
        var_r = Tree.find_var_in_subtree(node.right)
        return list(var_l + var_r)
    
    def get_depth(self, root, target_node, depth=0):
        if root is None:
            return -1
        if root == target_node:
            return depth
        left_depth = self.get_depth(root.left, target_node, depth + 1)
        if left_depth != -1:
            return left_depth
        return self.get_depth(root.right, target_node, depth + 1)
    
    def evaluate_tree(self, x):
        return Tree._evaluate_tree_recursive(self.root, x)
    
    #OLD IMPLEMENTATION, can be deleted?
    # @staticmethod
    # def _evaluate_tree_recursive(node, x):
    #     try:
    #         with np.errstate(all='raise'):#to catch exceptions in numpy
    #             if node.node_type == NodeType.VAR:
    #                 number = int(node.value[1:])
    #                 return x[number]
    #             if node.node_type == NodeType.CONST:
    #                 return node.value
    #             if node.node_type == NodeType.U_OP:
    #                 if node.right is None:
    #                     return node.value(Tree._evaluate_tree_recursive(node.left, x))
    #                 return node.value(Tree._evaluate_tree_recursive(node.right, x))
    #             if node.node_type == NodeType.B_OP:
    #                 # print(f"calculating {node.value.__name__} of {self._evaluate_tree_recursive(node.left, x)} and {self._evaluate_tree_recursive(node.right, x)}, the result is {node.value(self._evaluate_tree_recursive(node.left, x), self._evaluate_tree_recursive(node.right, x))}")
                

    #                 return node.value(
    #                     Tree._evaluate_tree_recursive(node.left, x), 
    #                     Tree._evaluate_tree_recursive(node.right, x)
    #                 )
    #     except (OverflowError, ZeroDivisionError, ValueError, RuntimeError, FloatingPointError):
    #         return np.nan
    
    
    @staticmethod
    def _evaluate_tree_recursive(node, x):
        # Create a unique cache key for this node and input
        # Check if result is already cached
        if node.node_type == NodeType.VAR:
            number = int(node.value[1:])
            result = x[number]
        elif node.node_type == NodeType.CONST:
            result = node.value
        else:
                #check if result is already cached
                cache_key = (node.value.__name__, id(node.left), id(node.right), x.tobytes())
                if cache_key in Tree._memo_cache:
                    return Tree._memo_cache[cache_key]
                if node.node_type == NodeType.U_OP:
                    if node.right is None:
                        result = node.value(Tree._evaluate_tree_recursive(node.left, x))
                    else:
                        result = node.value(Tree._evaluate_tree_recursive(node.right, x))
                elif node.node_type == NodeType.B_OP:
                    result = node.value(
                        Tree._evaluate_tree_recursive(node.left, x),
                        Tree._evaluate_tree_recursive(node.right, x)
                    )
                
                Tree._memo_cache[cache_key] = result
                if len(Tree._memo_cache) > Tree._cache_limit:
                 Tree._memo_cache.popitem() 
        
        return result
        

    #OLD IMPLEMENTATION, can be deleted?
    # #mean squared error
    # def compute_fitness(self,test=False):
    #     x_data = Tree.x_test if test else Tree.x_train
    #     y_data = Tree.y_test if test else Tree.y_train
    #     try:
    #         if x_data.shape[0] == 0:
    #             self.fitness = np.inf
    #             return
    #         squared_errors = 0
    #         # print(x_data.shape[1])
    #         for i in range(x_data.shape[1]):
    #             y_pred = self.evaluate_tree(x_data[:, i])
    #             if np.isnan(y_pred) or np.isinf(y_pred):
    #                 self.fitness = np.inf
    #                 return
    #             with np.errstate(all='raise'): #to raise exceptions in numpy
    #                 squared_errors += (y_data[i] - y_pred) ** 2
                
    #                 self.fitness = squared_errors / x_data.shape[1]

    #     except (OverflowError, ZeroDivisionError, ValueError, RuntimeError, FloatingPointError):  # Catch RuntimeWarning as RuntimeError
    #         self.fitness = np.inf
    #         return
        


    #mean squared error
    def compute_fitness(self,test=False):
        x_data = Tree.x_test if test else Tree.x_train
        y_data = Tree.y_test if test else Tree.y_train
        if x_data.shape[0] == 0 or x_data.shape[1] == 0:
            self.fitness = np.inf
            return
        squared_errors = 0
        # print(x_data.shape[1])
        for i in range(x_data.shape[1]):
            y_pred = self.evaluate_tree(x_data[:, i])
            if np.isnan(y_pred) or np.isinf(y_pred):
                self.fitness = np.inf
                return
            squared_errors += (y_data[i] - y_pred) ** 2
        self.fitness = squared_errors / x_data.shape[1]

        return       
    
    def add_drawing(self):
        """Draws the tree using matplotlib."""
        def draw_node(node, x, y, dx, dy):
            if node is not None:
                color = 'red' if node.node_type == NodeType.VAR else 'lightblue'  # VAR nodes are red
                plt.text(x, y, str(node), ha='center', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=color))
                if node.left is not None:
                    plt.plot([x, x - dx], [y - dy / 2, y - dy], color='black')
                    draw_node(node.left, x - dx, y - dy, dx / 2, dy)
                if node.right is not None:
                    plt.plot([x, x + dx], [y - dy / 2, y - dy], color='black')
                    draw_node(node.right, x + dx, y - dy, dx / 2, dy)

        plt.figure(figsize=(12, 8))
        plt.axis('off')
        draw_node(self.root, 0, 0, 20, 2)
        plt.show()
    

    def to_np_formula(self):
        return self.root.to_np_formula()
    

    
    
    #if the branches are too deep (over max_depth) collapse the ones that do not contain variables replacing them with their constant value
    @staticmethod
    def collapse_branch(node, current_depth=0,force_collapse=False):
        if node is None:
            return None
        if current_depth >= Tree.max_depth or force_collapse: 
            vars_in_subtree = Tree.find_var_in_subtree(node)
            if len(vars_in_subtree) == 0 and node.node_type != NodeType.CONST:
                
                # print("collapsed")
                
                ev = Tree._evaluate_tree_recursive(node, np.zeros(Tree.n_var))

                node.node_type = NodeType.CONST
                node.value = float(ev)  
                node.left = None
                node.right = None
                return node
        node.left = Tree.collapse_branch(node.left, current_depth + 1,force_collapse=force_collapse)
        node.right = Tree.collapse_branch(node.right, current_depth + 1,force_collapse=force_collapse)
        return node
    


    def __lt__(self, other):
        return self.fitness > other.fitness
    



     

# unary_ops = [
#     np.negative,
#     np.abs,
#     np.sqrt,
#     np.exp,
#     np.log,
#     np.sin,
#     np.cos,
#     np.tan,
#     np.arcsin,
#     np.arccos,
#     np.arctan,
#     np.ceil,
#     np.floor
# ]

# binary_ops = [
#     np.add,
#     np.subtract,
#     np.multiply,
#     np.divide,
#     np.power,
#     np.maximum,
#     np.minimum,
#     np.mod
# ]

# Tree.set_params(unary_ops, binary_ops, 3, 10)  
# t = Tree(2)
# t.print_tree()
def main():
    Tree.set_params(unary_ops, binary_ops, 3, 100,4, np.array([[1,2,3],[1,2,3],[1,2,3]]),np.array([1,2,3]))
    t=Tree("grow")
  
    t.print_tree()
    t.add_drawing()
if __name__ == "__main__":
    main()
   
# print(t.evaluate_tree([1, 2, 3]))