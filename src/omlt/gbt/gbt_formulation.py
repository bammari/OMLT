import collections

import numpy as np
import pyomo.environ as pe

from omlt.formulation import _PyomoFormulation, _setup_scaled_inputs_outputs
from omlt.gbt.model import GradientBoostedTreeModel
from collections import defaultdict
from collections import deque

class GBTBigMFormulation(_PyomoFormulation):
    """
    This class is the entry-point to build gradient-boosted trees formulations.

    This class iterates over all trees in the ensemble and generates
    constraints to enforce splitting rules according to:

    References
    ----------
     * Misic, V. "Optimization of tree ensembles."
       Operations Research 68.5 (2020): 1605-1624.
     * Mistry, M., et al. "Mixed-integer convex nonlinear optimization with gradient-boosted trees embedded."
       INFORMS Journal on Computing (2020).

    Parameters
    ----------
    tree_ensemble_structure : GradientBoostedTreeModel
        the tree ensemble definition
    """

    def __init__(self, gbt_model):
        super().__init__()
        self.model_definition = gbt_model

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        return list(range(self.model_definition.n_inputs))

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        return list(range(self.model_definition.n_outputs))

    def _build_formulation(self):
        """This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        _setup_scaled_inputs_outputs(
            self.block,
            self.model_definition.scaling_object,
            self.model_definition.scaled_input_bounds,
        )

        add_formulation_to_block(
            block=self.block,
            model_definition=self.model_definition,
            input_vars=self.block.scaled_inputs,
            output_vars=self.block.scaled_outputs,
        )

class GBTSimpleFormulation(_PyomoFormulation):
    """
    This class is the entry-point to build gradient-boosted trees formulations.

    This class iterates over all trees in the ensemble and generates
    constraints to enforce splitting rules according to:

    References
    ----------
     * Misic, V. "Optimization of tree ensembles."
       Operations Research 68.5 (2020): 1605-1624.
     * Mistry, M., et al. "Mixed-integer convex nonlinear optimization with gradient-boosted trees embedded."
       INFORMS Journal on Computing (2020).

    Parameters
    ----------
    tree_ensemble_structure : GradientBoostedTreeModel
        the tree ensemble definition
    """

    def __init__(self, gbt_model):
        super().__init__()
        self.model_definition = gbt_model

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        return list(range(self.model_definition.n_inputs))

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        return list(range(self.model_definition.n_outputs))

    def _build_formulation(self):
        """This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        _setup_scaled_inputs_outputs(
            self.block,
            self.model_definition.scaling_object,
            self.model_definition.scaled_input_bounds,
        )

        add_simple_formulation_to_block(
            block=self.block,
            model_definition=self.model_definition,
            input_vars=self.block.scaled_inputs,
            output_vars=self.block.scaled_outputs,
        )

def add_formulation_to_block(block, model_definition, input_vars, output_vars):
    r"""
    Adds the gradient-boosted trees formulation to the given Pyomo block.

    .. math::
        \begin{align*}
        \hat{\mu} &= \sum\limits_{t \in T} \sum\limits_{l \in {L_t}}
            F_{t,l} z_{t,l}, && \\
        \sum\limits_{l \in L_t} z_{t,l} &= 1, && \forall t \in T, \\
        \sum\limits_{l \in \text{Left}_{t,s}} z_{t,l} &\leq y_{i(s),j(s)},
            && \forall t \in T, \forall s \in V_t, \\
        \sum\limits_{l \in \text{Right}_{t,s}} z_{t,l} &\leq 1 - y_{i(s),j(s)},
            && \forall t \in T, \forall s \in V_t, \\
        y_{i,j} &\leq y_{i,j+1},
            && \forall i \in \left [ n \right ], \forall j \in \left [ m_i - 1 \right ], \\
        x_{i} &\geq v_{i,0} +
            \sum\limits_{j=1}^{m_i} \left (v_{i,j} -
            v_{i,j-1} \right ) \left ( 1 - y_{i,j} \right ),
            && \forall i \in \left [ n \right ], \\
        x_{i} &\leq v_{i,m_i+1} +
            \sum\limits_{j=1}^{m_i} \left (v_{i,j} - v_{i,j+1} \right ) y_{i,j},
            && \forall i \in \left [ n \right ]. \\
        \end{align*}


    References
    ----------
     * Misic, V. "Optimization of tree ensembles."
       Operations Research 68.5 (2020): 1605-1624.
     * Mistry, M., et al. "Mixed-integer convex nonlinear optimization with gradient-boosted trees embedded."
       INFORMS Journal on Computing (2020).

    Parameters
    ----------
    block : Block
        the Pyomo block
    tree_ensemble_structure : GradientBoostedTreeModel
        the tree ensemble definition
    input_vars : Var
        the input variables of the Pyomo block
    output_vars : Var
        the output variables of the Pyomo block

    """
    if isinstance(model_definition, GradientBoostedTreeModel):
        gbt = model_definition.onnx_model
    else:
        gbt = model_definition
    graph = gbt.graph

    root_node = graph.node[0]
    attr = _node_attributes(root_node)

    # base_values don't apply to lgbm models
    base_value = (
        np.array(attr["base_values"].floats)[0] if "base_values" in attr else 0.0
    )

    nodes_feature_ids = np.array(attr["nodes_featureids"].ints)
    nodes_values = np.array(attr["nodes_values"].floats)
    nodes_modes = np.array(attr["nodes_modes"].strings)
    nodes_tree_ids = np.array(attr["nodes_treeids"].ints)
    nodes_node_ids = np.array(attr["nodes_nodeids"].ints)
    nodes_false_node_ids = np.array(attr["nodes_falsenodeids"].ints)
    nodes_true_node_ids = np.array(attr["nodes_truenodeids"].ints)
    nodes_hitrates = np.array(attr["nodes_hitrates"].floats)
    nodes_missing_value_tracks_true = np.array(
        attr["nodes_missing_value_tracks_true"].ints
    )

    n_targets = attr["n_targets"].i
    target_ids = np.array(attr["target_ids"].ints)
    target_node_ids = np.array(attr["target_nodeids"].ints)
    target_tree_ids = np.array(attr["target_treeids"].ints)
    target_weights = np.array(attr["target_weights"].floats)

    # Compute derived data
    nodes_leaf_mask = nodes_modes == b"LEAF"
    nodes_branch_mask = nodes_modes == b"BRANCH_LEQ"

    tree_ids = set(nodes_tree_ids)
    feature_ids = set(nodes_feature_ids)

    continuous_vars = dict()

    for var_idx in input_vars:
        var = input_vars[var_idx]
        continuous_vars[var_idx] = var

    block.z_l = pe.Var(
        list(zip(nodes_tree_ids[nodes_leaf_mask], nodes_node_ids[nodes_leaf_mask])),
        bounds=(0, None),
        domain=pe.Reals,
    )

    branch_value_by_feature_id = dict()
    branch_value_by_feature_id = collections.defaultdict(list)

    for f in feature_ids:
        nodes_feature_mask = nodes_feature_ids == f
        branch_values = nodes_values[nodes_feature_mask & nodes_branch_mask]
        branch_value_by_feature_id[f] = np.unique(np.sort(branch_values))

    y_index = [
        (f, bi)
        for f in continuous_vars.keys()
        for bi, _ in enumerate(branch_value_by_feature_id[f])
    ]
    block.y = pe.Var(y_index, domain=pe.Binary)

    @block.Constraint(tree_ids)
    def single_leaf(b, tree_id):
        r"""
        Add constraint to ensure that only one leaf per tree is active, Mistry et al. Equ. (3b).
        .. math::
            \begin{align*}
            \sum\limits_{l \in L_t} z_{t,l} &= 1, && \forall t \in T
            \end{align*}
        """
        tree_mask = nodes_tree_ids == tree_id
        return (
            sum(
                b.z_l[tree_id, node_id]
                for node_id in nodes_node_ids[nodes_leaf_mask & tree_mask]
            )
            == 1
        )

    nodes_tree_branch_ids = [
        (t, b)
        for t in tree_ids
        for b in nodes_node_ids[(nodes_tree_ids == t) & nodes_branch_mask]
    ]

    def _branching_y(tree_id, branch_node_id):
        node_mask = (nodes_tree_ids == tree_id) & (nodes_node_ids == branch_node_id)
        feature_id = nodes_feature_ids[node_mask]
        branch_value = nodes_values[node_mask]
        assert len(feature_id) == 1 and len(branch_value) == 1
        feature_id = feature_id[0]
        branch_value = branch_value[0]
        (branch_y_idx,) = np.where(
            branch_value_by_feature_id[feature_id] == branch_value
        )
        assert len(branch_y_idx) == 1
        return block.y[feature_id, branch_y_idx[0]]

    def _sum_of_z_l(tree_id, start_node_id):
        tree_mask = nodes_tree_ids == tree_id
        local_false_node_ids = nodes_false_node_ids[tree_mask]
        local_true_node_ids = nodes_true_node_ids[tree_mask]
        local_mode = nodes_modes[tree_mask]
        visit_queue = [start_node_id]
        sum_of_z_l = 0.0
        while visit_queue:
            node_id = visit_queue.pop()
            if local_mode[node_id] == b"LEAF":
                sum_of_z_l += block.z_l[tree_id, node_id]
            else:
                # add left and right child to list of nodes to visit
                visit_queue.append(local_false_node_ids[node_id])
                visit_queue.append(local_true_node_ids[node_id])
        return sum_of_z_l

    @block.Constraint(nodes_tree_branch_ids)
    def left_split(b, tree_id, branch_node_id):
        r"""
        Add constraint to activate all left splits leading to an active leaf,
        Mistry et al. Equ. (3c).
        .. math::
            \begin{align*}
            \sum\limits_{l \in \text{Left}_{t,s}} z_{t,l} &\leq y_{i(s),j(s)},
            && \forall t \in T, \forall s \in V_t
            \end{align*}
        """
        node_mask = (nodes_tree_ids == tree_id) & (nodes_node_ids == branch_node_id)
        y = _branching_y(tree_id, branch_node_id)

        subtree_root = nodes_true_node_ids[node_mask][0]
        return _sum_of_z_l(tree_id, subtree_root) <= y

    @block.Constraint(nodes_tree_branch_ids)
    def right_split(b, tree_id, branch_node_id):
        r"""
        Add constraint to activate all right splits leading to an active leaf,
        Mistry et al. Equ. (3d).
        .. math::
            \begin{align*}
            \sum\limits_{l \in \text{Right}_{t,s}} z_{t,l} &\leq 1 - y_{i(s),j(s)},
            && \forall t \in T, \forall s \in V_t
            \end{align*}
        """
        node_mask = (nodes_tree_ids == tree_id) & (nodes_node_ids == branch_node_id)
        y = _branching_y(tree_id, branch_node_id)

        subtree_root = nodes_false_node_ids[node_mask][0]
        return _sum_of_z_l(tree_id, subtree_root) <= 1 - y

    @block.Constraint(y_index)
    def order_y(b, feature_id, branch_y_idx):
        r"""
        Add constraint to activate splits in the correct order.
        Mistry et al. Equ. (3e).
        .. math::
            \begin{align*}
            y_{i,j} &\leq y_{i,j+1},
            && \forall i \in \left [ n \right ], \forall j \in \left [ m_i - 1 \right ]
            \end{align*}
        """
        branch_values = branch_value_by_feature_id[feature_id]
        if branch_y_idx >= len(branch_values) - 1:
            return pe.Constraint.Skip
        return b.y[feature_id, branch_y_idx] <= b.y[feature_id, branch_y_idx + 1]

    @block.Constraint(y_index)
    def var_lower(b, feature_id, branch_y_idx):
        r"""
        Add constraint to link discrete tree splits to lower bound of continuous variables.
        Mistry et al. Equ. (4a).
        .. math::
            \begin{align*}
            x_{i} &\geq v_{i,0} +
            \sum\limits_{j=1}^{m_i} \left (v_{i,j} -
            v_{i,j-1} \right ) \left ( 1 - y_{i,j} \right ),
            && \forall i \in \left [ n \right ]
            \end{align*}
        """
        x = input_vars[feature_id]
        if x.lb is None:
            return pe.Constraint.Skip
        branch_value = branch_value_by_feature_id[feature_id][branch_y_idx]
        return x >= x.lb + (branch_value - x.lb) * (1 - b.y[feature_id, branch_y_idx])

    @block.Constraint(y_index)
    def var_upper(b, feature_id, branch_y_idx):
        r"""
        Add constraint to link discrete tree splits to upper bound of continuous variables.
        Mistry et al. Equ. (4b).
        .. math::
            \begin{align*}
            x_{i} &\leq v_{i,m_i+1} +
            \sum\limits_{j=1}^{m_i} \left (v_{i,j} - v_{i,j+1} \right ) y_{i,j},
            && \forall i \in \left [ n \right ]
            \end{align*}
        """
        x = input_vars[feature_id]
        if x.ub is None:
            return pe.Constraint.Skip
        branch_value = branch_value_by_feature_id[feature_id][branch_y_idx]
        return x <= x.ub + (branch_value - x.ub) * b.y[feature_id, branch_y_idx]

    @block.Constraint()
    def tree_mean_value(b):
        r"""
        Add constraint to link block output tree model mean.
        Mistry et al. Equ. (3a).
        .. math::
            \begin{align*}
            \hat{\mu} &= \sum\limits_{t \in T} \sum\limits_{l \in {L_t}}
            F_{t,l} z_{t,l}
            \end{align*}
        """
        return (
            output_vars[0]
            == sum(
                weight * b.z_l[tree_id, node_id]
                for tree_id, node_id, weight in zip(
                    target_tree_ids, target_node_ids, target_weights
                )
            )
            + base_value
        )


def _node_attributes(node):
    attr = dict()
    for at in node.attribute:
        attr[at.name] = at
    return attr

def add_simple_formulation_to_block(block, model_definition, input_vars, output_vars):
    if isinstance(model_definition, GradientBoostedTreeModel):
        gbt = model_definition.onnx_model
    else:
        gbt = model_definition
    graph = gbt.graph

    root_node = graph.node[0]
    attr = _node_attributes(root_node)

    # base_values don't apply to lgbm models
    base_value = (
        np.array(attr["base_values"].floats)[0] if "base_values" in attr else 0.0
    )

    nodes_feature_ids = np.array(attr["nodes_featureids"].ints)
    nodes_values = np.array(attr["nodes_values"].floats)
    nodes_modes = np.array(attr["nodes_modes"].strings)
    nodes_tree_ids = np.array(attr["nodes_treeids"].ints)
    nodes_node_ids = np.array(attr["nodes_nodeids"].ints)
    nodes_false_node_ids = np.array(attr["nodes_falsenodeids"].ints)
    nodes_true_node_ids = np.array(attr["nodes_truenodeids"].ints)
    nodes_hitrates = np.array(attr["nodes_hitrates"].floats)
    nodes_missing_value_tracks_true = np.array(
        attr["nodes_missing_value_tracks_true"].ints
    )

    n_targets = attr["n_targets"].i
    target_ids = np.array(attr["target_ids"].ints)
    target_node_ids = np.array(attr["target_nodeids"].ints)
    target_tree_ids = np.array(attr["target_treeids"].ints)
    target_weights = np.array(attr["target_weights"].floats)
    nodes_leaf_mask = nodes_modes == b"LEAF"
    nodes_branch_mask = nodes_modes == b"BRANCH_LEQ"

    tree_ids = set(nodes_tree_ids)
    feature_ids = set(nodes_feature_ids)
    splits_dic = defaultdict(dict)
    leaves_dic = defaultdict(dict)
    for i in tree_ids:
        # splits_dic[i] = {"node": nodes_node_ids[nodes_tree_ids==i]}
        node = nodes_node_ids[nodes_tree_ids == i]
        feature = nodes_feature_ids[nodes_tree_ids == i]
        value = nodes_values[nodes_tree_ids == i]
        mode = nodes_modes[nodes_tree_ids == i]
        target_weight = target_weights[target_tree_ids == i]
        count = 0
        count_leaf = 0
        queue = deque([node[count]])
        while queue:
            cur = queue[0]
            queue.popleft()
            if mode[cur] == b'BRANCH_LEQ':
                splits_dic[i][cur] = {'th': value[cur],
                                      'col': feature[cur],
                                      'children': [None, None]}
                queue.appendleft(node[count + 2])
                splits_dic[i][cur]['children'][0] = node[count + 1]
                queue.appendleft(node[count + 1])
                splits_dic[i][cur]['children'][1] = node[count + 2]
                count += 2
            else:
                leaves_dic[i][cur] = {'val': target_weight[count_leaf]}
                count_leaf += 1

    for i in tree_ids:
        splits = splits_dic[i]
        leaves = leaves_dic[i]
        for split in splits:
            left_child = splits[split]['children'][0]
            right_child = splits[split]['children'][1]
            if left_child in splits:
                splits[left_child]['parent'] = split
            else:
                leaves[left_child]['parent'] = split

            if right_child in splits:
                splits[right_child]['parent'] = split
            else:
                leaves[right_child]['parent'] = split

    for i in tree_ids:
        splits = splits_dic[i]
        leaves = leaves_dic[i]
        for split in splits:
            # print("split:" + str(split))
            left_child = splits[split]['children'][0]
            right_child = splits[split]['children'][1]

            if left_child in splits:
                # means left_child is split
                splits[split]['left_leaves'] = find_all_children_leaves(
                    left_child, splits, leaves
                )
            else:
                # means left_child is leaf
                splits[split]['left_leaves'] = [left_child]
                # print("left_child" + str(left_child))

            if right_child in splits:
                splits[split]['right_leaves'] = find_all_children_leaves(
                    right_child, splits, leaves
                )
            else:
                splits[split]['right_leaves'] = [right_child]
                # print("right_child" + str(right_child))

    features = np.arange(0, len(set(nodes_feature_ids)))
    for i in tree_ids:
        splits = splits_dic[i]
        leaves = leaves_dic[i]
        for leaf in leaves:
            leaves[leaf]['bounds'] = {}
        for th in features:
            for leaf in leaves:
                leaves[leaf]['bounds'][th] = [None, None]

    for i in tree_ids:
        splits = splits_dic[i]
        leaves = leaves_dic[i]
        for split in splits:
            var = splits[split]['col']
            for leaf in splits[split]['left_leaves']:
                leaves[leaf]['bounds'][var][1] = splits[split]['th']

            for leaf in splits[split]['right_leaves']:
                leaves[leaf]['bounds'][var][0] = splits[split]['th']
    for t in tree_ids:
        leaves_dic[t] = reassign_none_bounds(
            leaves_dic[t], model_definition.scaled_input_bounds, features)

    tree_leaf_set = []
    for t in tree_ids:
        for l in leaves_dic[t].keys():
            tree_leaf_set.append((t, l))
    block.z = pe.Var(tree_leaf_set, within=pe.Binary)
    block.d = pe.Var(tree_ids)

    def lowerBounds(m, t, f):
        # t = 1
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(leaves_dic[t][l]['bounds'][f][0] * m.z[t, l] for l in L) <= input_vars[f]
    block.lbCon = pe.Constraint(tree_ids, features, rule=lowerBounds)

    def upperBounds(m, t, f):
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(leaves_dic[t][l]['bounds'][f][1] * m.z[t, l] for l in L) >= input_vars[f]
    block.ubCon = pe.Constraint(tree_ids, features, rule=upperBounds)

    def outPuts(m, t):
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(m.z[t, l] * leaves_dic[t][l]['val'] for l in L) == block.d[t]
    block.outputCon = pe.Constraint(tree_ids, rule=outPuts)

    def onlyOne(m, t):
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(m.z[t, l] for l in L) == 1
    block.onlyOneCon = pe.Constraint(tree_ids, rule=onlyOne)

    block.final_sum = pe.Constraint(
        expr=output_vars[0] == sum(block.d[t] for t in tree_ids))


def find_all_children_splits(split, splits_dict):
    """
    This helper function finds all multigeneration children splits for an 
    argument split.

    Arguments:
        split --The split for which you are trying to find children splits
        splits_dict -- A dictionary of all the splits in the tree

    Returns:
        A list containing the Node IDs of all children splits
    """
    all_splits = []

    # Check if the immediate left child of the argument split is also a split.
    # If so append to the list then use recursion to generate the remainder
    left_child = splits_dict[split]['children'][0]
    if left_child in splits_dict:
        all_splits.append(left_child)
        all_splits.extend(find_all_children_splits(left_child, splits_dict))

    # Same as above but with right child
    right_child = splits_dict[split]['children'][1]
    if right_child in splits_dict:
        all_splits.append(right_child)
        all_splits.extend(find_all_children_splits(right_child, splits_dict))

    return all_splits


def find_all_children_leaves(split, splits_dict, leaves_dict):
    """
    This helper function finds all multigeneration children leaves for an 
    argument split.

    Arguments:
        split -- The split for which you are trying to find children leaves
        splits_dict -- A dictionary of all the split info in the tree
        leaves_dict -- A dictionary of all the leaf info in the tree

    Returns:
        A list containing all the Node IDs of all children leaves
    """
    all_leaves = []

    # Find all the splits that are children of the relevant split
    all_splits = find_all_children_splits(split, splits_dict)

    # Ensure the current split is included
    if split not in all_splits:
        all_splits.append(split)

    # For each leaf, check if the parents appear in the list of children
    # splits (all_splits). If so, it must be a leaf of the argument split
    for leaf in leaves_dict:
        if leaves_dict[leaf]['parent'] in all_splits:
            all_leaves.append(leaf)

    return all_leaves


def reassign_none_bounds(leaves, input_bounds, features):
    """
    This helper function reassigns bounds that are None to the bounds
    input by the user

    Arguments:
        leaves -- The dictionary of leaf information. Attribute of the 
            LinearTreeModel object
        input_bounds -- The nested dictionary

    Returns:
        The modified leaves dict without any bounds that are listed as None
    """
    L = np.array(list(leaves.keys()))
    # features = np.arange(0, len(set(nodes_feature_ids)))

    for l in L:
        for f in features:
            if leaves[l]['bounds'][f][0] == None:
                leaves[l]['bounds'][f][0] = input_bounds[f][0]
            if leaves[l]['bounds'][f][1] == None:
                leaves[l]['bounds'][f][1] = input_bounds[f][1]

    return leaves
