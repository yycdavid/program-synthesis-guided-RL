from z3 import *
import copy

N_TYPES = 4
CONNECTED = 0
WATER = 1
STONE = 2
NOT_ADJACENT = 3

EITHER = -1

NOOP = 0


class CraftSynthesizer(object):
    """
    Two functionalities:
    - synthesize a plan
    - Keep track of progress and decide when to advance to the next command

    Variable Namings:
        - b: boundary variables, b[t][(i,j)][boundary_type]
        - z: zone variables, z[t][i]
        - r: resource variables, r[t][resource][i]
        - w: workshop variables, w[t][workshop][i]
        - a: agent inventory variables, a[t][object]
        - n: object made at step t, n[t][object]. This is only meaningful for steps where the action is use workshop
        - p: action variables, p[t][action]
    """

    def __init__(self):
        super(CraftSynthesizer, self).__init__()
        self.cached_models = {}

    '''
    Solve max sat over states. Return None if no plan can solve any of the states. Otherwise return the plan, condition, and number of sat
    - abs_states: completed abstract states by the C-VAE
    - concrete_state: Concrete state, use it to get inventory
    '''

    def plan_max_sat(
            self,
            abs_states,
            concrete_state,
            world,
            goal,
            max_steps=10):
        # Construct formula that shares the p variables
        self.max_steps = max_steps
        # Get useful stuff
        self.world = world
        self.cookbook = self.world.cookbook
        action_indices = copy.copy(self.world.action_indices)
        action_indices.append(NOOP)
        self.action_indices = action_indices

        # Store handles to variables
        self.b_all = {}
        self.z_all = {}
        self.r_all = {}
        self.w_all = {}
        self.a_all = {}
        self.n_all = {}

        optimizer = Optimize()

        # Construct p
        self.p, constr = self._construct_p(self.max_steps)
        optimizer.add(constr)

        # Add soft constraints
        indicator_vars = {}
        for i, abs_state in enumerate(abs_states):
            constraint = self._cons_for_scenario(
                abs_state, concrete_state, i, goal)
            optimizer.add_soft(constraint)
            indicator_vars[i] = Bool("indicator_%s" % i)
            optimizer.add(indicator_vars[i] == constraint)

        # Solve
        optimizer.check()

        # Get number of sat
        m = optimizer.model()

        last_i = -1
        n_sat = 0
        for i in indicator_vars.keys():
            if is_true(m[indicator_vars[i]]):
                last_i = i
                n_sat += 1

        if last_i >= 0:
            # Set registered vars to last_i
            self.b = self.b_all[last_i]
            self.z = self.z_all[last_i]
            self.r = self.r_all[last_i]
            self.w = self.w_all[last_i]
            self.a = self.a_all[last_i]
            self.n = self.n_all[last_i]

            # Return solutions
            plan_conds = self._extract_solution(m)
            plan = [pc[0] for pc in plan_conds]
            conditions = [pc[1] for pc in plan_conds]
            plan_in_name = [self.cookbook.index.get(action) for action in plan]

            solved_results = {}
            solved_results['solved'] = True
            solved_results['plan'] = plan
            solved_results['plan_in_name'] = plan_in_name
            solved_results['transition_conditions'] = conditions
            solved_results['n_sat'] = n_sat

            return solved_results

        else:
            return None

    def _cons_for_scenario(self, abs_state, concrete_state, scenario_id, goal):
        self.n_zones = abs_state['n_zones']

        cons = []
        constr = self._construct_vars_other(scenario_id)
        cons.append(constr)

        # Add transition constraints
        constr = self._get_transition_constraints()
        cons.append(constr)

        # Add start state constraints
        constr = self._get_start_constraints_from_abs(
            concrete_state, abs_state)
        cons.append(constr)

        # Add goal constraints
        constr = self._get_goal_constraints(goal)
        cons.append(constr)

        return And(cons)

    def _construct_p(self, max_steps):
        # Action vars
        p = {}
        cons = []
        for t in range(max_steps):
            p[t] = {}
            for action in self.action_indices:
                p[t][action] = Bool(
                    "p_%s_%s" %
                    (t, self.cookbook.index.get(action) if action != NOOP else "noop"))
            # Cardinality constraints
            constr = self._singleton_constraint(
                [p[t][key] for key in p[t]])
            cons.append(constr)

        return p, And(cons)

    def _construct_vars_other(self, scenario_id):
        cons = []
        # Define state variables
        self.b = {}
        self.z = {}
        self.r = {}
        self.w = {}
        self.a = {}
        self.n = {}
        for t in range(self.max_steps + 1):
            # Boundary vars
            self.b[t] = {}
            for i in range(self.n_zones):
                for j in range(i, self.n_zones):
                    self.b[t][(i, j)] = [Bool("%s_b_%s_%s_%s_%s" % (
                        scenario_id, t, i, j, type)) for type in range(N_TYPES)]
                    self.b[t][(j, i)] = self.b[t][(i, j)]
                    # Cardinality constraints
                    constr = self._singleton_constraint(self.b[t][(i, j)])
                    cons.append(constr)

            # Zone vars
            self.z[t] = [
                Bool(
                    "%s_z_%s_%s" %
                    (scenario_id, t, i)) for i in range(
                    self.n_zones)]
            constr = self._singleton_constraint(self.z[t])
            cons.append(constr)

            # Resource vars
            self.r[t] = {}
            for primitive in self.world.primitive_indices:
                self.r[t][primitive] = [Int("%s_r_%s_%s_%s" % (
                    scenario_id, t, self.cookbook.index.get(primitive), i)) for i in range(self.n_zones)]
                # Non-negative constraints
                constr = And([var >= 0 for var in self.r[t][primitive]])
                cons.append(constr)

            # Workshop vars
            self.w[t] = {}
            for workshop in self.world.workshop_indices:
                self.w[t][workshop] = [
                    Bool(
                        "%s_w_%s_%s_%s" %
                        (scenario_id,
                         t,
                         self.cookbook.index.get(workshop),
                         i)) for i in range(
                        self.n_zones)]

            # Agent vars
            self.a[t] = {}
            for obj in self.world.grabbable_indices:
                self.a[t][obj] = Int(
                    "%s_a_%s_%s" %
                    (scenario_id, t, self.cookbook.index.get(obj)))
                cons.append(self.a[t][obj] >= 0)

            # Object made vars
            self.n[t] = {}
            for obj in self.world.artifact_indices:
                self.n[t][obj] = Int(
                    "%s_n_%s_%s" %
                    (scenario_id, t, self.cookbook.index.get(obj)))
                cons.append(self.n[t][obj] >= 0)

        # Store a handle for this set of variables
        self.b_all[scenario_id] = self.b
        self.z_all[scenario_id] = self.z
        self.r_all[scenario_id] = self.r
        self.w_all[scenario_id] = self.w
        self.a_all[scenario_id] = self.a
        self.n_all[scenario_id] = self.n

        return And(cons)

    '''
    init_state: CraftState
    goal: Index of the goal object
    '''

    def synthesize_plan(
            self,
            init_state,
            goal,
            max_steps=10,
            optimistic=False):
        self.max_steps = max_steps
        # Get useful stuff
        self.world = init_state.world
        self.cookbook = self.world.cookbook
        action_indices = copy.copy(self.world.action_indices)
        action_indices.append(NOOP)
        self.action_indices = action_indices

        # Store handles to variables
        self.b_all = {}
        self.z_all = {}
        self.r_all = {}
        self.w_all = {}
        self.a_all = {}
        self.n_all = {}

        # Construct solver
        self.solver = Solver()

        # Get zones
        region_result = self.world._get_zones_and_boundaries(init_state.grid)

        # Define state variables and basic constraints like cardinality and
        # non-negativity, and transition constraints
        constr = self._construct_vars_and_constraints(
            init_state, region_result)
        self.solver.add(constr)

        # Add start state constraints
        if optimistic:
            constr = self._get_start_constraints_optim(
                init_state, region_result)
        else:
            constr = self._get_start_constraints(init_state, region_result)
        self.solver.add(constr)

        # Add goal constraints
        constr = self._get_goal_constraints(goal)
        self.solver.add(constr)

        # Solve
        status = self.solver.check()
        solved_results = {}
        solved_results['solved'] = (status == sat)
        if solved_results['solved']:
            m = self.solver.model()
            # Return solutions
            plan_conds = self._extract_solution(m)
            plan = [pc[0] for pc in plan_conds]
            conditions = [pc[1] for pc in plan_conds]
            plan_in_name = [self.cookbook.index.get(action) for action in plan]
            solved_results['plan'] = plan
            solved_results['plan_in_name'] = plan_in_name
            solved_results['transition_conditions'] = conditions

        return solved_results

    def _extract_solution(self, model):
        actions = [
            self._extract_action_at(
                model,
                t) for t in range(
                self.max_steps)]
        actual_actions = [action for action in actions if action[0] != NOOP]
        return actual_actions

    def _extract_action_at(self, model, t):
        for action in self.action_indices:
            if is_true(model[self.p[t][action]]):
                condition = self._get_condition(model, t, action)
                return (action, condition)

    '''
    Condition is the minimum number of key item in the inventory to advance to the next command. For usable_indices is the exact number
    '''

    def _get_condition(self, model, t, action):
        condition = {}
        if action == NOOP:
            return condition
        elif action in self.cookbook.primitives:
            condition[action] = model[self.a[t + 1][action]].as_long()
        elif action in self.world.workshop_set:
            for artifact in self.cookbook.workshop_menus[action]:
                condition[artifact] = model[self.a[t + 1][artifact]].as_long()
        else:  # usable items
            condition[action] = model[self.a[t + 1][action]].as_long()

        return condition

    def _get_transition_constraints(self):
        cons = []
        for t in range(self.max_steps):
            constr = self._get_constr_get_resource(t)
            cons.append(constr)
            constr = self._get_constr_use_workshop(t)
            cons.append(constr)
            constr = self._get_constr_use_tool(
                t, self.cookbook.index.index("bridge"), WATER)
            cons.append(constr)
            constr = self._get_constr_use_tool(
                t, self.cookbook.index.index("axe"), STONE)
            cons.append(constr)
            if len(self.world.usable_set) == 3:
                constr = self._get_constr_use_tool(
                    t, self.cookbook.index.index("ladder"), EITHER)
                cons.append(constr)
            constr = self._get_constr_noop(t)
            cons.append(constr)
        return And(cons)

    '''
    Add transition constraints for use tool actions
    '''

    def _get_constr_use_tool(self, t, tool, boundary_type):
        change_condition = []
        for i in range(self.n_zones):
            for j in range(self.n_zones):
                lhs = And(self.z[t][i], self.z[t + 1][j])
                rhs_conds = []
                if boundary_type == EITHER:
                    rhs_conds.append(Or(self.b[t][(i, j)][WATER], self.b[t][(i, j)][STONE]))
                else:
                    rhs_conds.append(self.b[t][(i, j)][boundary_type])
                rhs_conds.append(self.b[t + 1][(i, j)][CONNECTED])
                rhs_conds.append(self.a[t + 1][tool] == self.a[t][tool] - 1)
                rhs_conds.append(self._propagate_boundary(t, i, j))

                rhs = And(rhs_conds)
                cond = Implies(lhs, rhs)
                change_condition.append(cond)

        unchange_condition = []
        unchange_condition.append(self._w_unchange(t))
        unchange_condition.append(self._r_unchange(t))
        unchange_condition.append(self._a_unchange(t, except_case=tool))

        clause = And(change_condition + unchange_condition)

        return Implies(self.p[t][tool], clause)

    def _get_constr_noop(self, t):
        unchange_condition = []
        unchange_condition.append(self._w_unchange(t))
        unchange_condition.append(self._r_unchange(t))
        unchange_condition.append(self._b_unchange(t))
        unchange_condition.append(self._a_unchange(t))
        unchange_condition.append(self._z_unchange(t))

        clause = And(unchange_condition)

        return Implies(self.p[t][NOOP], clause)

    def _propagate_boundary(self, t, i, j):
        conds = []
        for ip in range(self.n_zones):
            for jp in range(ip, self.n_zones):
                connect_by_ij = And(
                    self._connect_to_one(
                        t, ip, i, j), self._connect_to_one(
                        t, jp, i, j))
                conds.append(self.b[t + 1][(ip, jp)][CONNECTED] ==
                             Or(self.b[t][(ip, jp)][CONNECTED], connect_by_ij))
                # Remains the previous boundary if not connected
                not_connected = Not(self.b[t + 1][(ip, jp)][CONNECTED])
                same_cond = And([self.b[t + 1][(ip, jp)][type] == self.b[t]
                                 [(ip, jp)][type] for type in range(1, N_TYPES)])
                conds.append(Implies(not_connected, same_cond))

        return And(conds)

    '''
    Condition that zone is connected with at least one of i or j
    '''

    def _connect_to_one(self, t, zone, i, j):
        return Or(self.b[t][(zone, i)][CONNECTED],
                  self.b[t][(zone, j)][CONNECTED])

    '''
    Add transition constraints for use workshop actions
    '''

    def _get_constr_use_workshop(self, t):
        cons = []
        for workshop in self.world.workshop_indices:
            change_condition = []
            for i in range(self.n_zones):
                for j in range(self.n_zones):
                    lhs = And(self.z[t][i], self.z[t + 1][j])
                    rhs_conds = []
                    rhs_conds.append(self.b[t][(i, j)][CONNECTED])
                    rhs_conds.append(self.w[t][workshop][j])

                    # Something needs to be made
                    at_this_workshop = self.cookbook.workshop_menus[workshop]
                    total_made = Sum([self.n[t][obj]
                                      for obj in at_this_workshop])
                    rhs_conds.append(total_made >= 1)

                    # Can't make things not provided at this workshop
                    made_outside = Sum(
                        [self.n[t][obj] for obj in self.world.artifact_indices if obj not in at_this_workshop])
                    rhs_conds.append(made_outside == 0)

                    # Update agent's inventory
                    for primitive in self.world.primitive_indices:
                        used = self._get_amount_used(
                            t, primitive, at_this_workshop)
                        rhs_conds.append(
                            self.a[t + 1][primitive] == self.a[t][primitive] - used)
                    for obj in self.world.artifact_indices:
                        used = self._get_amount_used(t, obj, at_this_workshop)
                        rhs_conds.append(
                            self.a[t + 1][obj] == self.a[t][obj] - used + self.n[t][obj])

                    # Can't make any more artifacts
                    for obj in at_this_workshop:
                        can_still_make = And([self.a[t + 1][res] >= self.cookbook.amount_needed(
                            obj, res) for res in self.cookbook.direct_ingredients(obj)])
                        rhs_conds.append(Not(can_still_make))

                    rhs = And(rhs_conds)
                    cond = Implies(lhs, rhs)
                    change_condition.append(cond)

            unchange_condition = []
            unchange_condition.append(self._b_unchange(t))
            unchange_condition.append(self._w_unchange(t))
            unchange_condition.append(self._r_unchange(t))

            clause = And(change_condition + unchange_condition)

            cons.append(Implies(self.p[t][workshop], clause))

        return And(cons)

    def _get_amount_used(self, t, primitive, at_this_workshop):
        return Sum([self.cookbook.amount_needed(artifact, primitive)
                    * self.n[t][artifact] for artifact in at_this_workshop])

    '''
    Add transition constraints for get resource actions
    '''

    def _get_constr_get_resource(self, t):
        cons = []
        for primitive in self.world.primitive_indices:
            change_condition = []
            for i in range(self.n_zones):
                for j in range(self.n_zones):
                    lhs = And(self.z[t][i], self.z[t + 1][j])
                    rhs_conds = []
                    rhs_conds.append(self.b[t][(i, j)][CONNECTED])
                    rhs_conds.append(self.r[t][primitive][j] >= 1)
                    for res in self.world.primitive_indices:
                        if res != primitive:
                            rhs_conds.append(
                                self.r[t + 1][res][j] == self.r[t][res][j])
                    rhs_conds.append(
                        self.r[t + 1][primitive][j] == self.r[t][primitive][j] - 1)
                    rhs_conds.append(
                        self.a[t + 1][primitive] == self.a[t][primitive] + 1)
                    rhs = And(rhs_conds)
                    cond = Implies(lhs, rhs)
                    change_condition.append(cond)

            unchange_condition = []
            unchange_condition.append(self._b_unchange(t))
            unchange_condition.append(self._w_unchange(t))
            unchange_condition.append(
                self._a_unchange(
                    t, except_case=primitive))
            for j in range(self.n_zones):
                lhs = Not(self.z[t + 1][j])
                rhs_conds = []
                for res in self.world.primitive_indices:
                    rhs_conds.append(self.r[t + 1][res]
                                     [j] == self.r[t][res][j])
                rhs = And(rhs_conds)
                unchange_condition.append(Implies(lhs, rhs))

            clause = And(change_condition + unchange_condition)

            cons.append(Implies(self.p[t][primitive], clause))

        return And(cons)

    '''
    Get the condition saying b is unchanged from t to t+1
    '''

    def _b_unchange(self, t):
        conds = []
        for i in range(self.n_zones):
            for j in range(i, self.n_zones):
                for type in range(N_TYPES):
                    conds.append(self.b[t + 1][(i, j)][type]
                                 == self.b[t][(i, j)][type])
        return And(conds)

    '''
    Get the condition saying r is unchanged from t to t+1
    '''

    def _r_unchange(self, t):
        conds = []
        for i in range(self.n_zones):
            for primitive in self.world.primitive_indices:
                conds.append(self.r[t + 1][primitive][i]
                             == self.r[t][primitive][i])
        return And(conds)

    '''
    Get the condition saying w is unchanged from t to t+1
    '''

    def _w_unchange(self, t):
        conds = []
        for i in range(self.n_zones):
            for workshop in self.world.workshop_indices:
                conds.append(self.w[t + 1][workshop][i]
                             == self.w[t][workshop][i])
        return And(conds)

    '''
    Get the condition saying a is unchanged from t to t+1
    '''

    def _a_unchange(self, t, except_case=None):
        conds = []
        for item in self.world.grabbable_indices:
            if item != except_case:
                conds.append(self.a[t + 1][item] == self.a[t][item])
        return And(conds)

    '''
    Get the condition saying z is unchanged from t to t+1
    '''

    def _z_unchange(self, t):
        conds = []
        for i in range(self.n_zones):
            conds.append(self.z[t + 1][i] == self.z[t][i])
        return And(conds)

    def _get_goal_constraints(self, goal):
        return self.a[self.max_steps][goal] >= 1

    def _get_start_constraints(self, init_state, region_result):
        cons = []
        # Inventory
        for obj in self.world.grabbable_indices:
            cons.append(self.a[0][obj] == init_state.inventory[obj])

        # Boundary
        init_boundary = region_result['boundaries']
        for i in range(self.n_zones):
            for j in range(i, self.n_zones):
                type = init_boundary[(i, j)]
                cons.append(self.b[0][(i, j)][type])

        # Resources, workshops, and zone
        for i in range(self.n_zones):
            region = region_result['regions'][i]
            counts = self.world._count_things(init_state.grid, region)

            for primitive in self.world.primitive_indices:
                cons.append(self.r[0][primitive][i] == counts[primitive])

            for workshop in self.world.workshop_indices:
                if counts[workshop] > 0:
                    cons.append(self.w[0][workshop][i])
                else:
                    cons.append(Not(self.w[0][workshop][i]))

            if init_state.pos in region:
                cons.append(self.z[0][i])

        return And(cons)

    '''
    Get start constraints in optimistic synthesis
    '''

    def _get_start_constraints_optim(self, init_state, region_result):
        cons = []
        # Inventory
        for obj in self.world.grabbable_indices:
            cons.append(self.a[0][obj] == init_state.inventory[obj])

        # Boundary
        init_boundary = region_result['boundaries']
        for i in range(self.n_zones):
            for j in range(i, self.n_zones):
                type = init_boundary[(i, j)]
                cons.append(self.b[0][(i, j)][type])

        # Resources, workshops, and zone
        for i in range(self.n_zones):
            region = region_result['regions'][i]
            counts = self.world._count_things(
                init_state.grid,
                region,
                obs_mask=init_state.mask,
                count_unobs=True)

            for primitive in self.world.primitive_indices:
                cons.append(self.r[0][primitive][i] >= counts[primitive])
                cons.append(
                    self.r[0][primitive][i] <= (
                        counts[primitive] +
                        counts['unobserved']))

            for workshop in self.world.workshop_indices:
                if counts[workshop] > 0:
                    cons.append(self.w[0][workshop][i])
                else:
                    if counts['unobserved'] == 0:
                        cons.append(Not(self.w[0][workshop][i]))

            if init_state.pos in region:
                cons.append(self.z[0][i])

        return And(cons)

    '''
    Get start constraints from completed abstract state directly
    '''

    def _get_start_constraints_from_abs(self, init_state, abs_state):
        cons = []
        # Inventory
        for obj in self.world.grabbable_indices:
            cons.append(self.a[0][obj] == init_state.inventory[obj])

        # Boundary
        init_boundary = abs_state['boundaries']
        for i in range(self.n_zones):
            for j in range(i, self.n_zones):
                type = init_boundary[(i, j)]
                cons.append(self.b[0][(i, j)][type])

        # Resources, workshops, and zone
        for i in range(self.n_zones):
            counts = abs_state['counts'][i]
            for primitive in self.world.primitive_indices:
                cons.append(self.r[0][primitive][i] == counts[primitive])

            for workshop in self.world.workshop_indices:
                if counts[workshop] > 0:
                    cons.append(self.w[0][workshop][i])
                else:
                    cons.append(Not(self.w[0][workshop][i]))

        # Agent is always in zone 0 because we use abstract state directly
        cons.append(self.z[0][0])

        return And(cons)

    '''
    Construct variables, and construct constraint. Use cache to store constructed variables and constraints.
    '''

    def _construct_vars_and_constraints(self, init_state, region_result):
        self.n_zones = region_result['n_zones']
        if (self.max_steps, self.n_zones) in self.cached_models:
            cached_model = self.cached_models[(self.max_steps, self.n_zones)]
            self.b = cached_model['b']
            self.z = cached_model['z']
            self.r = cached_model['r']
            self.w = cached_model['w']
            self.a = cached_model['a']
            self.n = cached_model['n']
            self.p = cached_model['p']
            return And(cached_model['cons'])

        else:
            cons = []
            self.p, constr = self._construct_p(self.max_steps)
            cons.append(constr)

            constr = self._construct_vars_other(0)
            cons.append(constr)

            # Add transition constraints
            constr = self._get_transition_constraints()
            cons.append(constr)

            # Save model to cached
            self.cached_models[(self.max_steps, self.n_zones)] = {
                'b': self.b,
                'z': self.z,
                'r': self.r,
                'w': self.w,
                'a': self.a,
                'n': self.n,
                'p': self.p,
                'cons': cons,
            }
            return And(cons)

    '''
    One and only one var in vars is true
    '''

    def _singleton_constraint(self, vars):
        return And(AtMost(*vars, 1), Or(vars))
