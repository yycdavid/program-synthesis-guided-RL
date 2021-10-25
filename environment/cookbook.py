from misc.util import Struct, Index

import copy
import numpy as np
import yaml


class Cookbook(object):
    '''
    Attributes:
    - environment: set of indices for environment items (boundary, water, wall, etc.)
    - primitives: set of indices for resource items
    '''

    def __init__(self, recipes_path):
        with open(recipes_path) as recipes_f:
            recipes = yaml.load(recipes_f)
        #self.environment = set(recipes["environment"])
        self.index = Index()
        self.environment = set(self.index.index(e)
                               for e in recipes["environment"])
        self.primitives = set(self.index.index(p)
                              for p in recipes["primitives"])
        self.recipes = {}
        for output, inputs in recipes["recipes"].items():
            d = {}
            for inp, count in inputs.items():
                # special keys
                if "_" in inp:
                    d[inp] = count
                else:
                    d[self.index.index(inp)] = count
            self.recipes[self.index.index(output)] = d
        kinds = self.environment | self.primitives | set(self.recipes.keys())
        self.n_kinds = len(self.index)

        # workshop_menus is a dict: {workshop: set(artifacts)}
        self.workshop_menus = {}
        for artifact in self.recipes:
            workshop_name = self.recipes[artifact]['_at']
            workshop = self.index.index(workshop_name)
            if workshop in self.workshop_menus:
                self.workshop_menus[workshop].add(artifact)
            else:
                self.workshop_menus[workshop] = set()
                self.workshop_menus[workshop].add(artifact)

    def amount_needed(self, artifact, primitive):
        if primitive in self.recipes[artifact]:
            return self.recipes[artifact][primitive]
        else:
            return 0

    def direct_ingredients(self, artifact):
        ingredients = []
        for ingredient in self.recipes[artifact]:
            if isinstance(ingredient, int):
                ingredients.append(ingredient)

        return ingredients

    '''
    Returns a dict of {ingredient: count} in order to make goal
    '''

    def primitives_for(self, goal):
        out = {}

        def insert(kind, count):
            assert kind in self.primitives
            if kind not in out:
                out[kind] = count
            else:
                out[kind] += count

        for ingredient, count in self.recipes[goal].items():
            if not isinstance(ingredient, int):
                assert ingredient[0] == "_"
                continue
            elif ingredient in self.primitives:
                insert(ingredient, count)
            else:
                sub_recipe = self.recipes[ingredient]
                # Always n_produce=1 for now
                n_produce = sub_recipe["_yield"] if "_yield" in sub_recipe else 1
                n_needed = int(np.ceil(1. * count / n_produce))
                expanded = self.primitives_for(ingredient)
                for k, v in expanded.items():
                    insert(k, v * n_needed)

        return out

    '''
    Returns:
        - a dict of {ingredient (index): count} in order to make goal
        - a set including workshops needed (str)
    '''

    def primitives_all_for(self, goal):
        if goal in self.primitives:
            out = {goal: 1}
            out_workshop = set()
        else:
            out = {}
            out_workshop = set()

            def insert(kind, count):
                assert kind in self.primitives
                if kind not in out:
                    out[kind] = count
                else:
                    out[kind] += count

            for ingredient, count in self.recipes[goal].items():
                if not isinstance(ingredient, int):
                    assert ingredient[0] == "_"
                    out_workshop.add(count)
                elif ingredient in self.primitives:
                    insert(ingredient, count)
                else:
                    sub_recipe = self.recipes[ingredient]
                    # Always n_produce=1 for now
                    n_produce = sub_recipe["_yield"] if "_yield" in sub_recipe else 1
                    n_needed = int(np.ceil(1. * count / n_produce))
                    expanded, expanded_workshop = self.primitives_all_for(
                        ingredient)
                    for k, v in expanded.items():
                        insert(k, v * n_needed)
                    for w in expanded_workshop:
                        out_workshop.add(w)

        return (out, out_workshop)
