from typing import Any

class LabelTree:
    def __init__(self, name: str, name_index: int = 1, src_path: str | None = None):
        self.name = name
        self.name_index = name_index
        self.children: list["LabelTree"] = []
        self.src_path = src_path

    def add_child(self, child: "LabelTree") -> None:
        self.children.append(child)

    def find_similar_child(self, other: "LabelTree") -> "LabelTree | None":
        findings: list["LabelTree"] = []
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

SQUEEZABLE_NAMES = ["DropoutAdapter", "Chain"]

class AssignementError(Exception):
    def __init__(self, message: str, path: str = "") -> None:
        self.message = message
        self.path = path
        super().__init__(message)

    def extend_path(self, prefix: str) -> "AssignementError":
        if self.path == "":
            self.path = prefix
        else:
            self.path = prefix + "." + self.path

        return self

AssignementErrors = list[dict[str, str]]
Assignement = dict[str, str]

def match_trees(tree1: LabelTree, tree2: LabelTree) -> tuple[Assignement, list[AssignementError]]:
    out: Assignement = {}
    if tree1.name != tree2.name:
        error = AssignementError(message=f"Name mismatch: {tree1.name} != {tree2.name}")
        return (out, [error])
    if len(tree1.children) != len(tree2.children):
        error = AssignementError(
            message=f"Children mismatch in {tree1.name}: {len(tree1.children)} != {len(tree2.children)}"
        )
        return (out, [error])

    errors: list[AssignementError] = []
    for i in range(len(tree1.children)):
        child1 = tree1.children[i]
        child2 = tree2.children[i]

        if child1.name != child2.name:
            errors.append(AssignementError(path=tree1.name, message=f"Name mismatch: {child1.name} != {child2.name}"))
        else:
            child_match, child_errors = match_trees(child1, child2)
            out.update(child_match)
            errors.extend([err.extend_path(tree1.name) for err in child_errors])

            out.update(child_match)

    if (tree1.src_path is None and tree2.src_path is not None) or (
        tree1.src_path is not None and tree2.src_path is None
    ):
        errors.append(AssignementError(message=f"Source path mismatch: {tree1.src_path} != {tree2.src_path}"))

    if tree1.src_path is not None and tree2.src_path is not None:
        out[tree1.src_path] = tree2.src_path

    return (out, errors)


def get_or_create_from_path(path: str, registry: dict[str, LabelTree]):
    components = path.split(".")
    parent = registry[""]
    for i in range(1, len(components) + 1):
        label = ".".join(components[:i])
        full_name = components[i - 1]
        underscore_split = full_name.split("_")
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
            if i == len(components):
                src_path = path
            node = LabelTree(name, name_index, src_path)
            parent.add_child(node)
            registry[label] = node

        parent = registry[label]
    return parent


def paths_to_tree(paths: list[str]) -> LabelTree:
    node_registry: dict[str, LabelTree] = {}
    node_registry[""] = LabelTree("")
    for path in paths:
        get_or_create_from_path(path, node_registry)
    return node_registry[""]


def collapse_empty(node: LabelTree) -> list[LabelTree]:
    if node.name == "_empty":
        out: list[LabelTree] = []
        for child in node.children:
            out.extend(collapse_empty(child))
        return out
    else:
        return [node]


def simplify_node(node: LabelTree) -> LabelTree:
    if node.name in SQUEEZABLE_NAMES:
        simple_children: list[LabelTree] = []
        for child in node.children:
            child_node = simplify_node(child)
            simple_children.extend(collapse_empty(child_node))

        if len(simple_children) == 1:
            return simple_children[0]
        else:
            res_node = LabelTree("_empty")
            res_node.children = simple_children
            return res_node
    else:
        simple_children: list[LabelTree] = []
        for child in node.children:
            child_node = simplify_node(child)
            simple_children.extend(collapse_empty(child_node))
        node.children = simple_children
        return node

StateDictLike = dict[str, Any]

def match_state_dict(source: StateDictLike, target: StateDictLike) -> StateDictLike:
    source_keys = list(source.keys())
    target_keys = list(target.keys())
    simplified_source = simplify_node(paths_to_tree(source_keys))
    simplified_target = simplify_node(paths_to_tree(target_keys))
    (assignment, errors) = match_trees(simplified_source, simplified_target)

    if len(errors) > 0:
        raise Exception(errors)

    out: StateDictLike = {}

    for key in source_keys:
        if key in assignment:
            out[assignment[key]] = source[key]
    return out
