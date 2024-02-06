class LabelTreeNode:
    def __init__(self, name: str, name_index: int = 1, src_path: str | None = None):
        self.name = name
        self.name_index = name_index
        self.children: list["LabelTreeNode"] = []
        self.src_path = src_path
    def add_child(self, child: "LabelTreeNode") -> None:
        self.children.append(child)
    def find_similar_child(self, other: "LabelTreeNode") -> "LabelTreeNode | None":
        findings : list["LabelTreeNode"] = []
        for child in self.children:
            if child.name == other.name and child.name_index == other.name_index:
                findings.append(child)

        if len(findings) == 1:
            return findings[0]
        elif len(findings) == 0:
            print(f"Could not find a similar child for {other.name}")
            return None
        else:
            raise ValueError(f"Found multiple similar children for {other.name}")

SQUEEZABLE_NAMES = [
    "DropoutAdapter"
]

def match_trees(tree1: LabelTreeNode, tree2: LabelTreeNode) -> None | dict[str, str]:
    if tree1.name != tree2.name:
        print(f"Name mismatch: {tree1.name} != {tree2.name}")
        return None
    if len(tree1.children) != len(tree2.children):
        print(f"Children mismatch: {len(tree1.children)} != {len(tree2.children)}")
        return None
    
    out : dict[str, str] = {}
    
    for i in range(len(tree1.children)):
        child2 = tree2.find_similar_child(tree1.children[i])
        if child2 is None:
            return None
        child_match = match_trees(tree1.children[i], child2)
        if child_match is None:
            return None
        out.update(child_match)
        
    if (tree1.src_path is None and tree2.src_path is not None) or (tree1.src_path is not None and tree2.src_path is None):
        raise ValueError(f"Source path mismatch: {tree1.src_path} != {tree2.src_path}")
        
    if tree1.src_path is not None and tree2.src_path is not None:
        out[tree1.src_path] = tree2.src_path
    
    return out

def get_or_create_from_path(path: str, registry: dict[str, LabelTreeNode]):
    components = path.split('.')
    parent = registry['']
    for i in range(1, len(components) + 1):
        label = '.'.join(components[:i])
        full_name = components[i-1]
        underscore_split = full_name.split('_')
        if len(underscore_split) == 1:
            name = full_name
            name_index = 1
        elif len(underscore_split) == 2:
            name = underscore_split[0]
            name_index = int(underscore_split[1])
        else:
            raise ValueError(f"Component {full_name} has more than one underscore")
        
        if label not in registry:
            src_path = None
            if(i == len(components)):
                src_path = path
            node = LabelTreeNode(name, name_index, src_path)
            parent.add_child(node)
            registry[label] = node

        parent = registry[label]
    return parent

def paths_to_tree(paths) -> LabelTreeNode:
    node_registry : dict[str, LabelTreeNode] = {}
    node_registry[''] = LabelTreeNode('')
    for path in paths:
        get_or_create_from_path(path, node_registry)
    return node_registry['']

def simplify_node(node: LabelTreeNode, index : int | None = None) -> "LabelTree":
                
    if index is not None and node.name_index != 1:
        raise ValueError(f"Index {index} does not match the index of {parent.name}")
    
    if index is None:
        index = node.name_index
    else:
        node.name_index = index
    
    if node.name in SQUEEZABLE_NAMES:
        if len(node.children) == 1:
            return simplify_node(node.children[0], index)
        else:
            raise ValueError(f"Node {parent.name} has more than one child")
    else:
        for i, child in enumerate(node.children):
            node.children[i] = simplify_node(child)
        return node

