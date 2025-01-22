import math
import re
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt  
from functools import lru_cache
import cProfile 
import random

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
            return str(int(round(self.value)))  
        else:    
            return str(self.value)
    def clone(self):
        """Creates a deep copy of the current node."""
        new_node = Node(self.node_type, self.value)
        new_node.left = self.left.clone() if self.left else None
        new_node.right = self.right.clone() if self.right else None
        return new_node
    

    def to_np_formula_rec(self,use_std_operators=False):
        if self.value is None:
            return None
        if self.node_type == NodeType.CONST:
            return str(self.value)
        if self.node_type == NodeType.VAR:
            return "x["+self.value[1:]+"]"
            # return self.value
        if self.node_type == NodeType.U_OP:
            operand = self.right.to_np_formula_rec(use_std_operators) if self.left is None else self.left.to_np_formula_rec(use_std_operators)
            if use_std_operators:
                if(self.value.__name__=="negative"):
                    return f"-({operand})"
            return f"np.{self.value.__name__}({operand})"
        if self.node_type == NodeType.B_OP:
            left = self.left.to_np_formula_rec(use_std_operators)
            right = self.right.to_np_formula_rec(use_std_operators)
            if use_std_operators:
                if(self.value.__name__=="add"):
                    return f"({left} + {right})"
                if(self.value.__name__=="subtract"):
                    return f"({left} - {right})"
                if(self.value.__name__=="multiply"):
                    return f"({left} * {right})"
                if(self.value.__name__=="divide"):
                    return f"({left} / {right})"
            return f"np.{self.value.__name__}({left}, {right})"
  


class Tree:
    _VAR_DUP_PROB = 0.1 


    @staticmethod
    def set_params(unary_ops, binary_ops, n_var, max_const,max_depth,spawn_depth, x_train_norm, y_train_norm, x_test_norm=None, y_test_norm=None):
        Tree.unary_ops = unary_ops
        Tree.binary_ops = binary_ops
        Tree.n_var = n_var
        Tree.vars = [f'x{i}' for i in range(Tree.n_var)]
        Tree.max_const = max_const
        Tree.max_depth = max_depth
        Tree.spawn_depth= spawn_depth
        Tree.x_train = x_train_norm
        Tree.y_train = y_train_norm
        Tree.x_test = x_test_norm
        Tree.y_test = y_test_norm
     



    def __init__(self, method="full", require_valid_tree=True, empty=False):
        # Valid tree means a tree that has a computable fitness (no division by zero, no overflow, etc.)
        # print(f"--Creating tree with max_depth:{max_depth} and spawn_depth:{spawn_depth}")
        self.age = 0    # NOTE: it's just a test for select_parents_fitness_age
        self.fitness = np.inf
        self.root = None
        while not empty and self.fitness == np.inf:
            if method == "full":
                self.root = self.populate_tree_full_method()
            elif method == "grow":
                self.root = self.populate_tree_grow_method()
            self.compute_fitness()
            if not require_valid_tree:
                break
            
    
    
    def populate_tree_full_method(self):
        leaves = []
        #First create the leaves of the tree. Every variable should be placed in the tree at least once!
        for var in  Tree.vars:
            leaves.append(Node(NodeType.VAR, value=var))
        while len(leaves) < 2 ** Tree.spawn_depth:
            if(np.random.rand()<Tree._VAR_DUP_PROB): #Duplicate a variable
                value_idx=random.randint(0,Tree.n_var-1)
                leaves.append(Node(NodeType.VAR, value=Tree.vars[value_idx]))
            else:
                leaves.append(Node(NodeType.CONST, value=(-Tree.max_const + (Tree.max_const - (-Tree.max_const)) * random.random())))

        #Then build the tree recursively
        def build_tree(leaves_to_place, current_depth):
            if current_depth == Tree.spawn_depth:
                return leaves_to_place.pop(0)
            value_idx=random.randint(0,len(Tree.binary_ops)-1)
            node = Node(NodeType.B_OP, value=Tree.binary_ops[value_idx])
            node.left = build_tree(leaves_to_place, current_depth + 1)
            node.right = build_tree(leaves_to_place, current_depth + 1)
            return node

        np.random.shuffle(leaves)
        return build_tree(leaves, 0)

  
    def populate_tree_grow_method(self,max_depth=None,must_include_vars=None): 
        #max_depth and must_include_vars are set if creating a subtree: a tree wtih custom depth and do not need to include all the variables
        # print(f"Creating tree with max_depth:{Tree.max_depth} and spawn_depth:{Tree.spawn_depth}")
        if max_depth is None:
            #Pick a number of leaves between Tree.n_var and 2^max_depth
            n_leaves = np.random.randint(Tree.n_var, 2 ** Tree.spawn_depth +1)#+1 cause high val is exclusive
        else:
            if must_include_vars is None or len(must_include_vars) == 0:
                n_leaves = np.random.randint(1, 2 ** max_depth +1)
            else:
                n_leaves = np.random.randint(len(must_include_vars), 2 ** max_depth +1)

      
        leaves = []

        if(must_include_vars is None):
            vars_to_place = Tree.vars
        else:
            vars_to_place = must_include_vars
        #Create the leaves of the tree. Every variable should be placed in the tree at least once!
        for var in vars_to_place:
            leaves.append(Node(NodeType.VAR, value=var))
        for _ in range(n_leaves - len(vars_to_place)):
            if(np.random.rand()<Tree._VAR_DUP_PROB): #Duplicate a variable
                value_idx=random.randint(0,Tree.n_var-1)
                leaves.append(Node(NodeType.VAR, value=Tree.vars[value_idx])) 
            else:
                leaves.append(Node(NodeType.CONST, value=(-Tree.max_const + (Tree.max_const - (-Tree.max_const)) * random.random())))


        #Then build the tree recursively
        def build_tree(leaves_to_place, current_depth,max_depth=None):
            if max_depth is None:
                max_depth=Tree.spawn_depth

            if current_depth == max_depth:#enter here only if we reached the max depth, place the last leaf
                return leaves_to_place.pop(0)
            subtree_max_leaves=2**(max_depth-(current_depth+1)) #max number of leaves that can be placed in a subtree

            #place a leaf with a certain probability if the subtree has only one leaf to place
            if(np.random.rand()<0.5 and len(leaves_to_place)==1): 
                node=leaves_to_place.pop(0)
                return node
            
            #if the subtree has only one leaf to place (and it has not been placed in the crrent node) 
            # OR np.random.rand()<PROB and the single subtree has enough space to place all the leaves
            #place a unary operator
            if(len(leaves_to_place)==1 or (np.random.rand()<0.3 and len(leaves_to_place)>0 and len(leaves_to_place)<=subtree_max_leaves) ): 
                value_idx=random.randint(0,len(Tree.unary_ops)-1)
                node=Node(NodeType.U_OP,value=Tree.unary_ops[value_idx])
                node.left=build_tree(leaves_to_place, current_depth + 1,max_depth=max_depth)
                return node
            
            #code from now on is executed only if the subtree has more than one leaf to place and will place a binary operator
            if(len(leaves_to_place)<=1):
                print("Error: not enough leaves to place")
            place_on_right=None #setting to None to enter the while loop
            while place_on_right is None or place_on_right> subtree_max_leaves: #loop until also the right subtree has enough space to place the leaves
                #pick number of leaves to place on the left then calculate the number of leaves to place on the right
                if(len(leaves_to_place)>subtree_max_leaves):
                    place_on_left=np.random.randint(1,subtree_max_leaves+1)
                else:
                    place_on_left=np.random.randint(1,len(leaves_to_place)) #not using +1 even if high value is exclusive because otherwise we risk having right subtree with 0 leaves
                place_on_right=len(leaves_to_place)-place_on_left
                
            value_idx=random.randint(0,len(Tree.binary_ops)-1)
            node = Node(NodeType.B_OP, value=Tree.binary_ops[value_idx])
            node.left = build_tree(leaves_to_place[0:place_on_left], current_depth + 1,max_depth=max_depth)
            node.right = build_tree(leaves_to_place[place_on_left:], current_depth + 1,max_depth=max_depth)
            return node
                
        np.random.shuffle(leaves)
        if(len(leaves)==0):
            print("Error: no leaves to place!")
   
        return build_tree(leaves, 0,max_depth=max_depth)
           
            
                              
                
     
    def print_tree(self):
        self.print_tree_recursive(self.root, 0)

    def print_tree_recursive(self, node, depth):
        if node is not None:
            print("  " * depth + f"{depth}-{str(node)}")  
            self.print_tree_recursive(node.left, depth + 1)
            self.print_tree_recursive(node.right, depth + 1)
        else:
            print("  " * depth + "None")
    

    #count the instances of each variable in the vars_list
    #input: list of triplets of the form (node,depth,len)
    @staticmethod
    def count_vars(vars_list):
        var_count = {var: 0 for var in Tree.vars}
        for var in vars_list:
            var_count[var[0].value] += 1
        return var_count



    def mutate_subtree(self):
        self.age += 1

        variables_tree_tripe,other_nodes_triple = self.collect_nodes(self.root)
        variables_tree=Tree.count_vars(variables_tree_tripe)

        #choose a node that is not at the max depth
        valid_nodes=[i for i in variables_tree_tripe+other_nodes_triple if i[1]<Tree.max_depth]
        pick_idx=random.randint(0,len(valid_nodes)-1)
        picked_node_triple=valid_nodes[pick_idx]
        picked_node,picked_depth,_ = picked_node_triple

        # print("Picked node: ",str(picked_node))

        #get the variables in the subtree of the picked node and compare them with the variables in the tree
        subtree_vars_triple,_= self.collect_nodes(picked_node,depth=picked_depth)
        subtree_vars=Tree.count_vars(subtree_vars_triple)
        diff = {k: variables_tree[k] - subtree_vars[k] for k in Tree.vars}
        #diff is a dictionary that will have 0 as value for each variable that is present ONLY in the subtree. We must include these variables in the new subtree.
        must_include_vars = [k for k,v in diff.items() if v==0]
       
        
        # res= map(lambda x: variables_tree.remove(x), subtree_vars)
        # next(res)
    
 
        max_possible_depth = Tree.max_depth - picked_depth
        # print("Max possible depth: ",max_possible_depth)
 
        
        new_subtree = self.populate_tree_grow_method(max_possible_depth,must_include_vars=must_include_vars)
        if new_subtree is not None:
            picked_node.node_type = new_subtree.node_type
            picked_node.value = new_subtree.value
            picked_node.left = new_subtree.left
            picked_node.right = new_subtree.right

    def copy_tree(self):
        new_tree = Tree(empty=True)
        new_tree.root = self.root.clone()
        new_tree.fitness = self.fitness
        return new_tree



    def mutate_single_node(self, num_mutations=1):
        self.age += 1

        _,nodes_triple = self.collect_nodes(self.root)
        if(len(nodes_triple)==0): #if there are no nodes to mutate but the tree is made only of a variable
            return
        if(num_mutations>len(nodes_triple)):
            num_mutations=len(nodes_triple)
        for _ in range(num_mutations):
            index = np.random.randint(0, len(nodes_triple))
            node_to_mutate = nodes_triple[index][0]
            if node_to_mutate.node_type == NodeType.CONST:
                node_to_mutate.value = (-Tree.max_const + (Tree.max_const - (-Tree.max_const)) * np.random.random())
            elif node_to_mutate.node_type == NodeType.B_OP:
                op_idx=random.randint(0,len(Tree.binary_ops)-1)
                node_to_mutate.value = Tree.binary_ops[op_idx]
            elif node_to_mutate.node_type == NodeType.U_OP:
                op_idx=random.randint(0,len(Tree.unary_ops)-1)
                node_to_mutate.value = Tree.unary_ops[op_idx]
    
  
    #NOTE: the logic of this crossover is kinda convoluted because, while swapping the 2 subtrees, it assures that:
    #1) After the crossover the resulting trees have at least 1 of each variable
    #2) The resulting trees have a depth <= max_depth

    def crossover(self, tree2):
        # TODO: increment tree age in crossover for select_parents_fitness_age (?)
        new_tree1 = Tree(empty=True)
        new_tree2 = Tree(empty=True)
        new_tree1.root = self.root.clone()
        new_tree2.root = tree2.root.clone()

        tree1_vars, tree1_nodes = new_tree1.collect_nodes(new_tree1.root)
        tree2_vars, tree2_nodes = new_tree2.collect_nodes(new_tree2.root)

        #shuffle is important because we will iterate over the nodes and pick the first couple of subtrees that we find to be valid
        np.random.shuffle(tree1_nodes)
        np.random.shuffle(tree2_nodes)

        
        
    
        tree1_var_count = Tree.count_vars(tree1_vars)
        tree2_var_count = Tree.count_vars(tree2_vars)


        found_subtree1=None
        found_subtree1=None
        found_valid=False

        #Iterate for each node of the first tree
        while len(tree1_nodes) > 0 and found_valid==False:
            subtree1_triple = tree1_nodes.pop(0)
            subtree1_node,subtree1_depth,subtree1_len = subtree1_triple 

            #We check that the subtree we are considering can be swapped: the variables in it are also present somewhere else in the same tree
            subtree1_vars,_=self.collect_nodes(subtree1_node)
            subtree1_var_count=Tree.count_vars(subtree1_vars)

            #Difference between the 2 dictionaries: if the difference for one of the variables is <0 means that the variable only appears in the subtree, so we can't swap it
            diff1 = {k: tree1_var_count[k] - subtree1_var_count[k] for k in Tree.vars}
            #if each of the values in diff1 is >0 then we set found_subtree to True
            valid=all(v > 0 for v in diff1.values())
            if not valid:
                continue #choose another subtree1_node

            found_subtree1=subtree1_node

            #NOTE: Check the collect_nodes method for definition of LEN and DEPTH
            #General assumption: LEN_subtree1<=MAX_DEPTH-DEPTH_subtree2. This must hold for each subtree in the other tree
            #The LEN_subtree1 must be <= MAX_DEPTH-DEPTH_subtree2 that is: if we add the subtree2 to subtree1, the resulting tree must have a depth <= MAX_DEPTH
            #Then, we check also the other way around (after the 'end'): LEN_subtree2<=MAX_DEPTH-DEPTH_subtree1
            subtree_to_consider_in_tree2=[i[0] for i in tree2_nodes if (subtree1_len <=Tree.max_depth -i[1]) and i[2]<=Tree.max_depth- subtree1_depth]

            #Iterate for each suitable node of the second tree
            for subtree2_node in subtree_to_consider_in_tree2: 

                #We check that the subtree we are considering can be swapped: the variables in it are also present somewhere else in the same tree
                subtree2_vars,_=tree2.collect_nodes(subtree2_node)
                subtree2_var_count=Tree.count_vars(subtree2_vars)

                #Difference between the 2 dictionaries: if the difference for one of the variables is <0 means that the variable only appears in the subtree, so we can't swap it
                diff2 = {k: tree2_var_count[k] - subtree2_var_count[k] for k in Tree.vars}

                #if each of the values in diff2 is >0 then we set found_subtree to True
                valid=all(v > 0 for v in diff2.values())
                if valid:
                    found_valid=True
                    found_subtree2=subtree2_node 
                    break #We exit the loop because we found a valid subtree1_node, subtree2_node 
            
        if not found_valid:
            return None,None
          
        found_subtree1.node_type, found_subtree2.node_type = found_subtree2.node_type, found_subtree1.node_type
        found_subtree1.value, found_subtree2.value = found_subtree2.value, found_subtree1.value
        found_subtree1.left, found_subtree2.left = found_subtree2.left, found_subtree1.left
        found_subtree1.right, found_subtree2.right = found_subtree2.right, found_subtree1.right

        return new_tree1, new_tree2





    """
    @return: Two lists of tuples. The first list contains the variables in the tree, the second list contains the other nodes.
    Each tuple has the following structure: (node, depth, len).
    Node: the node itself
    Depth: the depth of the node in the tree (so root has depth 0, its children have depth 1, etc.)
    Len: the max length of the subtree of the current node (so a leaf has depth 0, its parent has depth 1, etc.)

    """
    def collect_nodes(self, node,depth=0):
        if node is None:
            return []
        if(node.node_type==NodeType.VAR):
            return [(node,depth,0)],[]
        if(node.node_type==NodeType.CONST):
            return [],[(node,depth,0)]
        
        if(node.left is not None):
            left_variables,left_others=self.collect_nodes(node.left,depth+1)
        else:
            left_variables,left_others=[],[]
        if(node.right is not None): 
            right_variables,right_others=self.collect_nodes(node.right,depth+1)
        else:
            right_variables,right_others=[],[]

        lenghts=[i[2] for i in left_variables+right_variables+left_others+right_others]
        max_len=max(lenghts)

        variables=left_variables+right_variables
        others=left_others+right_others

        return variables,others+[(node,depth,max_len+1)]
    
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
        


    


    def compute_fitness(self,test="train"):
        if(test=="train"):
            x_data = Tree.x_train
            y_data = Tree.y_train
        if(test=="test"):
            x_data = Tree.x_test
            y_data = Tree.y_test
        if(test=="all"):
            x_data = np.concatenate((Tree.x_train,Tree.x_test),axis=1)
            y_data = np.concatenate((Tree.y_train,Tree.y_test))

        # Conver the formula in a pre-compiled lambda function
        formula = self.root.to_np_formula_rec()  
        eval_formula = eval(f"lambda x: {formula}",{"np": np, "nan": np.nan, "inf": np.inf}) 

        # Exploiting np broadcasting
        y_pred = eval_formula(x_data)  


        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            self.fitness = np.inf
            return

        #Broadcasting is used to calculate the squared errors
        squared_errors = np.square(y_data - y_pred)

   
        self.fitness = np.sum(squared_errors) / x_data.shape[1]

    
    def add_drawing(self):
        """Draws the tree using matplotlib."""
        def draw_node(node, x, y, dx, dy):
            if node is not None:
                color = 'red' if node.node_type == NodeType.VAR else 'lightblue'  # VAR nodes are red
                plt.text(x, y, str(node), ha='center', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=color))
                if node.left is not None:
                    plt.plot([x, x - dx], [y - dy / 2, y - dy], color='black') #line to left
                    draw_node(node.left, x - dx, y - dy, dx / 2, dy)
                if node.right is not None:
                    plt.plot([x, x + dx], [y - dy / 2, y - dy], color='black') #line to right
                    draw_node(node.right, x + dx, y - dy, dx / 2, dy)

        plt.figure(figsize=(10, 7))
        plt.axis('off')
        draw_node(self.root, 0, 0, 20, 2)
        plt.show()


    
    def to_np_formula(self,use_std_operators=False):
        return self.root.to_np_formula_rec(use_std_operators=use_std_operators)
    
    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def split_arguments(arguments):
        """
        Split np function arguments based on parenthesis and commas.
        Es:"np.multiple(x[1],2),np.div(1,2)" -> ["np.multiple(x[1],2)", "np.div(1,2)"]
        """
        args = []
        bracket_level = 0
        current_arg = []
        for char in arguments:
            if char == ',' and bracket_level == 0:
                args.append("".join(current_arg))
                current_arg = []
            else:
                if char == '(':
                    bracket_level += 1
                elif char == ')':
                    bracket_level -= 1
                current_arg.append(char)
        if current_arg:
            args.append("".join(current_arg))
        return args

    

    @staticmethod
    def parse_expression(expression):
        expression = expression.replace(" ", "")
        if expression.startswith("np."):
            # Extracting function name and arguments
            match = re.match(r"np\.(\w+)\((.*)\)", expression)
            if not match:
                raise ValueError(f"Espressione non valida: {expression}")
            
            operation = match.group(1)  # Function name
            arguments = match.group(2)  # Args
            
            args = Tree.split_arguments(arguments)
            
            np_func = getattr(np, operation)
           
            _node_type = NodeType.B_OP if np_func.nargs-1 == 2 else NodeType.U_OP
            root = Node(value=np_func,node_type=_node_type)
            
            # Recursive calls
            if len(args) > 0:
                root.left = Tree.parse_expression(args[0])  
            if len(args) > 1:
                root.right = Tree.parse_expression(args[1])
            
            return root
        
        elif expression.isdigit() or Tree.is_float(expression):
            return Node(node_type=NodeType.CONST, value=float(expression))
        elif expression.startswith("x[") and expression.endswith("]"):
            return Node(node_type=NodeType.VAR, value="x"+expression[2:-1])
        else:
            raise ValueError(f"Invalid expression: {expression}")

    @staticmethod
    def create_tree_from_np_formula(formula):
        empty = Tree(empty=True)
        empty.root = Tree.parse_expression(formula)
        empty.compute_fitness()
        return empty

    
    
    #if the branches are too deep (over max_depth) or force_collapse is set to True collapse the ones that do not contain variables replacing them with their constant value
    @staticmethod
    def collapse_branch(node, current_depth=0,force_collapse=False,max_depth=10):
        if node is None:
            return None
        
        if current_depth > max_depth+1 or force_collapse: 
            vars_in_subtree = Tree.find_var_in_subtree(node)

            if len(vars_in_subtree) == 0 and node.node_type != NodeType.CONST:
                eval_formula = eval(f"lambda x: {node.to_np_formula_rec()}")
                # print("collapsed")
                ev = eval_formula(np.zeros(Tree.n_var))
                
                node.node_type = NodeType.CONST
                node.value = float(ev)  
                node.left = None
                node.right = None
                return node
        node.left = Tree.collapse_branch(node.left, current_depth + 1,force_collapse=force_collapse)
        node.right = Tree.collapse_branch(node.right, current_depth + 1,force_collapse=force_collapse)
        return node
    


    def __lt__(self, other):
        return self.fitness < other.fitness
    def __eq__(self, other):
        return self.fitness == other.fitness
    



     




def main():
    return



if __name__ == "__main__":
    main()


