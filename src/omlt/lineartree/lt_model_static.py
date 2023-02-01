import numpy as np

class LinearTreeModel:

    def __init__(self, lt_model, scaling_object=None, scaled_input_bounds=None):
        r"""
        Create a network definition object used to create the gradient-boosted trees
        formulation in Pyomo

        Args:
           lt_model : linear-tree model
              An linear-tree model that is generated by the linear-tree package 
           scaling_object : ScalingInterface or None
              A scaling object to specify the scaling parameters for the
              linear model tree inputs and outputs. If None, then no
              scaling is performed.
           scaled_input_bounds : dict or None
              A dict that contains the bounds on the scaled variables (the
              direct inputs to the tree ensemble). If None, then no bounds
              are specified or they are generated using unscaled bounds.
        """
        self.__model = lt_model
        self._splits, self._leaves, self._thresholds =\
            LinearTreeModel._parse_Tree_Data(lt_model)
        self.__scaling_object = scaling_object
        self.__scaled_input_bounds = scaled_input_bounds

    @staticmethod
    def _find_all_children_splits(split, splits):
        r"""
        This helper function finds all multigeneration children splits for an argument split.
        """
        # We will store all the split ids in a list.
        all_splits = []

        # Check if the immediate left child of the argument split is also a split. If so
        # append to the list
        child0 = splits[split]['children'][0]
        if child0 in splits:
            all_splits.append(child0)
        
        # Same as above but with right child
        child1 = splits[split]['children'][1]
        if child1 in splits:
            all_splits.append(child1)
        
        # Now iterate through the list and continue to check the same for all subsequent splits.
        for child in all_splits:
            if splits[child]['children'][0] in splits:
                all_splits.append(splits[child]['children'][0])
            if splits[child]['children'][1] in splits:
                all_splits.append(splits[child]['children'][1])
        
        return all_splits

    @staticmethod
    def _find_all_children_leaves(split, leaves, splits):
        """
        This function finds all multigeneration children leaves for an argument split.
        """
        # Store the ids of the leaves in a list
        all_leaves = []
        # Find all the splits that are children of the relevant split
        all_splits = LinearTreeModel._find_all_children_splits(split, splits)
        # If the current split not in all splits, append it to the list
        if split not in all_splits:
            all_splits.append(split)
        # For each leaf, check if the parents appear in the list of splits. If so,
        # It must be a leaf of the split
        for leaf in leaves:
            if leaves[leaf]['parent'] in all_splits:
                all_leaves.append(leaf)
        
        return all_leaves

    @staticmethod
    def _parse_Tree_Data(model):

        # Create the initial leaves and splits dictionaries. These are attributes of the 
        # LinearModelTree Objet 
        leaves = model.summary(only_leaves=True)
        splits = model.summary()

        # This loop removes unnecessary entries from the leaves dictionary and adds keys
        # for the slope and intercept. Also removes leaves from the splits dictionary
        for leaf in leaves:
            del splits[leaf]
            del leaves[leaf]['samples']
            leaves[leaf]['slope'] = list(leaves[leaf]['models'].coef_)
            leaves[leaf]['intercept'] = leaves[leaf]['models'].intercept_
            del leaves[leaf]['models']
        
        # This loop removes unnecessary entries from the splits dictionary. This loop
        # also creates an entry for each leaf or split in the tree that indicates which split
        # is its parent
        for split in splits:
            del splits[split]['loss']
            del splits[split]['samples']
            del splits[split]['models']

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
        
        # This loop goes through all the splits and determines gets a list of all
        # the leaves to the left of a split and all the leaves to the right of a split
        for split in splits:
            left_child = splits[split]['children'][0]
            right_child = splits[split]['children'][1]
            if left_child in splits:
                splits[split]['left_leaves'] = LinearTreeModel._find_all_children_leaves(left_child, leaves, splits)
            else:
                splits[split]['left_leaves'] = [left_child]
            if right_child in splits:
                splits[split]['right_leaves'] = LinearTreeModel._find_all_children_leaves(right_child, splits)
            else:
                splits[split]['right_leaves'] = [right_child]

        # For each variable that appears in the tree, go through all the splits
        # and assign its splitting threshold to the correct entry in this nested dictionary
        splitting_thresholds = {}
        for split in splits:
            var = splits[split]['col']
            splitting_thresholds[var] = {}
        for split in splits:
            var = splits[split]['col']
            splitting_thresholds[var][split] = splits[split]['th']

        # Make sure every nested dictionary in the vars_dict dictionary is sorted by value since
        # this plays an important role in the ordering of the indices of binary variables y_ij in the Misic 
        # formulation
        for var in splitting_thresholds:
            splitting_thresholds[var] = dict(sorted(splitting_thresholds[var].items(), key=lambda x: x[1]))

        # Once the splitting threshold dictionary
        for split in splits:
            var = splits[split]['col']
            splits[split]['y_index'] = []
            splits[split]['y_index'].append(splits[split]['col'])
            splits[split]['y_index'].append(list(splitting_thresholds[var]).index(split))

        # Go through and 
        for leaf in leaves:
            leaves[leaf]['bounds'] = {}
        
        L = np.array(list(leaves.keys()))
        features = np.arange(0,len(leaves[L[0]]['slope']))
        for th in features:
            for leaf in leaves:
                leaves[leaf]['bounds'][th] = [None, None]

        for split in splits:
            for leaf in splits[split]['left_leaves']:
                leaves[leaf]['bounds'][splits[split]['col']][1] = splits[split]['th']
            
            for leaf in splits[split]['right_leaves']:
                leaves[leaf]['bounds'][splits[split]['col']][0] = splits[split]['th']

        return splits, leaves, splitting_thresholds