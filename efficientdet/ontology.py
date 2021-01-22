from __future__ import absolute_import, division, print_function

import pickle
import yaml

from concept import ConceptEnum


class Ontology:
    """
    Ontology
    params:
        leaf_to_root: Dict[leaf, root], where the root of tree -> None

    Example:
        leaf2root = {
            'object': None, # the root
            'personnel': 'object',
            'vehicle': 'object',
            'male': 'personnel',
            'female': 'personnel',
            'car': 'vehicle',
            'truck': 'vehicle'
        }

        ontology = Ontology(leaf2root)
        print(ontology)

        >>> Ontology
            Connected components: 1

            object:
              personnel:
                female: 2
                male: 1
              vehicle:
                car: 3
                truck: 4

    """
    def __init__(self, leaf_to_root):

        roots, root_to_leaf = get_root_to_leaf(leaf_to_root)

        self.roots = roots

        self.leaf_to_root = leaf_to_root
        self.root_to_leaf = root_to_leaf
        self.n_components = len(roots)

        self.levels = \
        ontology_levels(roots, root_to_leaf)

        all_names = [n for l in list(self.levels.values()) for n in l]
        all_names.sort()
        self.all_names = all_names

        all_leaves = get_finest_leaves(leaf_to_root, root_to_leaf)
        all_leaves.sort()
        self.finest_leaves = all_leaves

        self.sumrules = get_sumrules(self.root_to_leaf)

    def __repr__(self):
        tree_str = self.__class__.__name__
        tree_str += "\nConnected components: {}".format(self.n_components)
        tree_str += "\n"
        for root in self.roots:
            tree_str += "\n"
            tree = expand_tree(root, self.root_to_leaf)
            tree_str += yaml.dump(tree, default_flow_style=False)
            tree_str += "\n"
        return tree_str

    def save(self, filename):
        with open(filename, 'bw') as f:
            pickle.dump(self.leaf_to_root, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'br') as f:
            leaf_to_root = pickle.load(f)
        return cls(leaf_to_root)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as f:
            leaf_to_root = yaml.safe_load(f)
        return cls(leaf_to_root)

    def to_yaml(self, filename):
        with open(filename, 'w') as f:
            f.write(yaml.dump(self.leaf_to_root))

    def export_finest_leaves(self, filename):
        with open(filename, 'w') as f:
            f.write(yaml.dump(self.finest_leaves))

    def export_all_names(self, filename):
        with open(filename, 'w') as f:
            f.write(yaml.dump(self.all_names))

    @property
    def enum_leaves(self):
        d = {
            'name': 'finest_leaves',
            'concepts': self.finest_leaves
        }

        return ConceptEnum.from_dict(d)

    @property
    def enum_all(self):
        d = {
            'name': 'all_concepts',
            'concepts': self.all_names
        }

        return ConceptEnum.from_dict(d)

    @property
    def support(self):
        en = self.enum_leaves
        su = {}

        for k, v in self.sumrules.items():
            su[k] = en.to_indices(v)

        for k in self.finest_leaves:
            su[k] = en.to_indices([k])

        return su


def connected_components(leaf_to_root):
    """Returns the number of roots in onotology
    Params:
        leaf_to_root, Dict[str, str] leaf_string -> root_string
    Outputs:
        num_roots, int
    """
    num_roots = 0
    for l, r in leaf_to_root.items():
        if r is None:
            num_roots += 1
    return num_roots


def get_root_to_leaf(leaf_to_root):
    """Converts leaf-to-root map to inverse map root-to-leaf
    Params:
        leaf_to_root, Dict[str, List[str]] leaf_string -> list of root_string
    Outputs:
        roots, List[str] root_strings if the tree has several connected_components
        root_to_leaf, Dict[str, str]
    """
    root_to_leaf = {}
    roots = []
    for l, r in leaf_to_root.items():
        if r is None:
            roots.append(l)
            continue

        if r in root_to_leaf:
            root_to_leaf[r].append(l)
        else:
            root_to_leaf[r] = [l]

    return roots, root_to_leaf


def get_finest_leaves(leaf_to_root, root_to_leaf):
    """Returns the finest mutually exclusive leaf strings
    Params:
        leaf_to_root: Dict[str, str], leaf_string -> root_string
        root_to_leaf: Dict[str, List[str]], root_string -> list of leaf_string
    Outputs:
        fine_leaves: List[str]
    """
    leaves = list(leaf_to_root.keys())
    fine_leaves = []
    for leaf in leaves:
        if leaf not in root_to_leaf:
            fine_leaves.append(leaf)
    return fine_leaves


def expand_tree(root, root_to_leaf):
    """Expands the tree into a nested Dict where the finest_leaves stores the enumeration
    Params:
        root, List[str] list of roots of the tree
        root_to_leaf, Dict[str, List[str]] map root_string -> list of leaf_string
    Output:
        tree, nested Dict
    Example:
        ontology = {'object':
                            {
                            'personnel': 1,
                            'vehicle': 2
                            }
                   }
    """

    def helper(root, tree, count=0):

        if root not in root_to_leaf:
            count += 1
            tree[root] = count
            return count

        leaves = root_to_leaf[root]

        tree[root] = {}
        for leaf in leaves:
            count = helper(leaf, tree[root], count)
        return count

    tree = {}
    count = helper(root, tree)

    return tree


def ontology_levels(roots, root_to_leaf):
    """Returns list of leaf strings at each ontology level
    Params:
        roots: List[str], list of tree root_string
        root_to_leaf: Dict[str, List[str]], map root_string -> list of leaf_string
    Outputs:
        memo: Dict[int, List[str]] dictionary of ontology level e.g. 0, 1, 2, ... -> list of leaf_string
    """

    def helper(roots, memo, level=0):
        if level in memo:
            memo[level] += roots
        else:
            memo[level] = roots
        level += 1
        for root in roots:
            if root in root_to_leaf:
                leaves = root_to_leaf[root][:]
                helper(leaves, memo, level)
    memo = {}
    helper(roots, memo)

    return memo


def get_sumrules(root_to_leaf):
    """Returns the sumrules of each class name
    Params:
        root_to_leaf: Dict[str, List[str]], map root_string -> list of leaf_string
    Outputs:
        memo: Dict[str, List[str]] class name -> list of finest leaf strings
    Example:
        personnel -> [male, female]
    """

    def helper(node, memo):
        if node in memo:
            return memo[node]
        if node not in root_to_leaf:
            return [node]

        memo[node] = []
        leaves = root_to_leaf[node]
        for l in leaves:
            memo[node] += helper(l, memo)
        return memo[node]

    memo = {}
    for r in root_to_leaf:
        _ = helper(r, memo)

    return memo
