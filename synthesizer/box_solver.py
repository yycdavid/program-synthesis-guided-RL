from z3 import *
import copy


class BoxSynthesizer(object):
    """
    Variable Namings:
    """

    def __init__(self):
        super(BoxSynthesizer, self).__init__()
        self.cached_models = {}
        self.cached_models_maxsat = {}

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
            max_steps=10):
        # Construct formula that shares the p variables
        self.max_steps = max_steps
        # Get useful stuff
        self.world = world
        action_indices = copy.copy(self.world.action_indices)
        self.NOOP = len(action_indices)
        action_indices.append(self.NOOP)
        self.action_indices = action_indices

        # Store handles to variables
        optimizer = Optimize()

        # Construct or use cached model
        n_completions = len(abs_states)
        cons, p_cons = self._construct_vars_and_trans_maxsat(n_completions)
        # After the above, self.p, self.b_all, l_all, a_all are defined
        optimizer.add(p_cons)

        # Add soft constraints
        indicator_vars = {}
        for i, abs_state in enumerate(abs_states):
            self.b = self.b_all[i]
            self.l = self.l_all[i]
            self.a = self.a_all[i]
            cons_start = self._get_start_constraints_from_abs(
                concrete_state, abs_state)
            cons_goal = self._get_goal_constraints()
            constraint = And(cons[i], cons_start, cons_goal)
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
            self.l = self.l_all[last_i]
            self.a = self.a_all[last_i]

            # Return solutions
            plan = self._extract_solution(m)

            solved_results = {}
            solved_results['solved'] = True
            solved_results['plan'] = plan
            solved_results['n_sat'] = n_sat

            return solved_results

        else:
            return None

    def synthesize_plan(self, init_state, max_steps=10):
        self.max_steps = max_steps
        # Get useful stuff
        self.world = init_state.world
        action_indices = copy.copy(self.world.action_indices)
        self.NOOP = len(action_indices)
        action_indices.append(self.NOOP)
        self.action_indices = action_indices

        # Store handles to variables
        self.b_all = {}
        self.l_all = {}
        self.a_all = {}

        # Construct solver
        self.solver = Solver()

        # Define state variables and basic constraints like cardinality and
        # non-negativity, and transition constraints
        constr = self._construct_vars_and_constraints()
        self.solver.add(constr)

        # Add start state constraints
        constr = self._get_start_constraints(init_state)
        self.solver.add(constr)

        # Add goal constraints
        constr = self._get_goal_constraints()
        self.solver.add(constr)

        # Solve
        status = self.solver.check()
        solved_results = {}
        solved_results['solved'] = (status == sat)
        if solved_results['solved']:
            m = self.solver.model()
            # Return solutions
            plan = self._extract_solution(m)
            solved_results['plan'] = plan

        return solved_results

    def _cons_for_scenario(self, abs_state, concrete_state, scenario_id):
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
        constr = self._get_goal_constraints()
        cons.append(constr)

        return And(cons)

    '''
    Get start constraints from completed abstract state directly
    '''

    def _get_start_constraints_from_abs(self, init_state, abs_state):
        '''
        Abstract state format:
        - loose: {color}
        - box: {(k1, k2): count}
        - key: color
        '''
        cons = []
        # inventory
        if init_state.key == -1:
            # No key
            cons.append(Not(Or(self.a[0])))
        else:
            cons.append(self.a[0][init_state.key])

        self.add_observed = True
        if self.add_observed:
            # Union partially observed things to hallucinated things
            abs_partial_state = self.world.get_abstract_partial_state(
                init_state)

        # loose key
        #assert len(abs_state['loose']) <= 1, "Should not have more than one loose key"
        for k in self.world.action_indices:
            if k in abs_state['loose'] or (
                    self.add_observed and k in abs_partial_state['loose']):
                cons.append(self.l[0][k])
            else:
                cons.append(Not(self.l[0][k]))

        # boxes
        for k1 in self.world.action_indices:
            for k2 in self.world.action_indices:
                if k1 == k2:
                    continue

                if (k1, k2) in abs_state['box'] or (
                        self.add_observed and (k1, k2) in abs_partial_state['box']):
                    if self.add_observed and (
                            k1, k2) in abs_partial_state['box']:
                        if (k1, k2) in abs_state['box']:
                            count = max(
                                abs_state['box'][(k1, k2)], abs_partial_state['box'][(k1, k2)])
                        else:
                            count = abs_partial_state['box'][(k1, k2)]
                    else:
                        count = abs_state['box'][(k1, k2)]
                    cons.append(self.b[0][(k1, k2)] == count)
                else:
                    cons.append(self.b[0][(k1, k2)] == 0)

        return And(cons)

    def _extract_solution(self, model):
        actions = [
            self._extract_action_at(
                model,
                t) for t in range(
                self.max_steps)]
        actual_actions = [action for action in actions if action != self.NOOP]
        return actual_actions

    '''
    action is the color index
    '''

    def _extract_action_at(self, model, t):
        for action in self.action_indices:
            if is_true(model[self.p[t][action]]):
                return action

    def _get_start_constraints(self, init_state):
        abstract_state = self.world.get_abstract_state(init_state)
        cons = []
        # inventory
        if abstract_state['key'] == -1:
            # No key
            cons.append(Not(Or(self.a[0])))
        else:
            cons.append(self.a[0][abstract_state['key']])

        # loose key
        assert len(
            abstract_state['loose']) <= 1, "Should not have more than one loose key"
        for k in self.world.action_indices:
            if k in abstract_state['loose']:
                cons.append(self.l[0][k])
            else:
                cons.append(Not(self.l[0][k]))

        # boxes
        for k1 in self.world.action_indices:
            for k2 in self.world.action_indices:
                if k1 == k2:
                    continue

                if (k1, k2) in abstract_state['box']:
                    cons.append(self.b[0][(k1, k2)] ==
                                abstract_state['box'][(k1, k2)])
                else:
                    cons.append(self.b[0][(k1, k2)] == 0)

        return And(cons)

    def _get_goal_constraints(self):
        return self.a[self.max_steps][self.world.goal_color_id]

    '''
    Construct variables, and construct constraint. Use cache to store constructed variables and constraints.
    '''

    def _construct_vars_and_constraints(self):
        if self.max_steps in self.cached_models:
            cached_model = self.cached_models[self.max_steps]
            self.b = cached_model['b']
            self.l = cached_model['l']
            self.a = cached_model['a']
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
            self.cached_models[self.max_steps] = {
                'b': self.b,
                'l': self.l,
                'a': self.a,
                'p': self.p,
                'cons': cons,
            }
            return And(cons)

    '''
    After calling this, the following should be ready:
    - self.p
    - self.b_all, l_all, a_all
    - cons[scenario_id]

    Return:
    - cons: list of constraints, each is for a scenario_id
    - p_cons: constraint for p var
    '''

    def _construct_vars_and_trans_maxsat(self, n_completions):
        if (self.max_steps, n_completions) in self.cached_models_maxsat:
            cached_model = self.cached_models_maxsat[(
                self.max_steps, n_completions)]
            self.p = cached_model['p']
            self.b_all = cached_model['b_all']
            self.l_all = cached_model['l_all']
            self.a_all = cached_model['a_all']
            return cached_model['cons'], cached_model['p_cons']

        else:
            self.b_all = {}
            self.l_all = {}
            self.a_all = {}
            cons = []
            # Construct p
            self.p, p_cons = self._construct_p(self.max_steps)

            # Construct other vars
            for scenario_id in range(n_completions):
                var_cons = self._construct_vars_other(scenario_id)
                trans_cons = self._get_transition_constraints()
                cons.append(And(var_cons, trans_cons))

            self.cached_models_maxsat[(self.max_steps, n_completions)] = {
                'p': self.p,
                'b_all': self.b_all,
                'l_all': self.l_all,
                'a_all': self.a_all,
                'p_cons': p_cons,
                'cons': cons,
            }
            return cons, p_cons

    def _get_transition_constraints(self):
        cons = []
        for t in range(self.max_steps):
            constr = self._get_constr_color(t)
            cons.append(constr)
            constr = self._get_constr_noop(t)
            cons.append(constr)
        return And(cons)

    def _get_constr_color(self, t):
        cons = []
        for color in self.world.action_indices:
            # Pick up a loose key
            loose_conditions = []
            loose_conditions.append(self.l[t][color])
            loose_conditions.append(self.a[t + 1][color])
            loose_conditions.append(Not(Or(self.l[t + 1])))
            loose_conditions.append(self._b_unchange(t))
            loose_condition = And(loose_conditions)

            # Open a lock and get key
            unlock_conditions = []
            unlock_conditions.append(self._singleton_constraint(self.a[t]))
            unlock_conditions.append(self.a[t + 1][color])
            unlock_conditions.append(Not(self.a[t][color]))
            unlock_conditions.append(self._l_unchange(t))
            for k1 in self.world.action_indices:
                if k1 == color:
                    cond = Not(self.a[t][k1])
                else:
                    rhs_conds = []
                    rhs_conds.append(
                        self.b[t + 1][(color, k1)] == self.b[t][(color, k1)] - 1)
                    rhs_conds.append(
                        self._b_unchange(
                            t, except_case=(
                                color, k1)))
                    cond = Implies(self.a[t][k1], And(rhs_conds))

                unlock_conditions.append(cond)

            unlock_condition = And(unlock_conditions)

            clause = Or(loose_condition, unlock_condition)
            cons.append(Implies(self.p[t][color], clause))

        return And(cons)

    def _get_constr_noop(self, t):
        unchange_condition = []
        unchange_condition.append(self._b_unchange(t))
        unchange_condition.append(self._l_unchange(t))
        unchange_condition.append(self._a_unchange(t))

        clause = And(unchange_condition)

        return Implies(self.p[t][self.NOOP], clause)

    '''
    Get the condition saying b is unchanged from t to t+1
    '''

    def _b_unchange(self, t, except_case=None):
        conds = []
        for k1 in self.world.action_indices:
            for k2 in self.world.action_indices:
                if (k1, k2) != except_case and k1 != k2:
                    conds.append(self.b[t + 1][(k1, k2)]
                                 == self.b[t][(k1, k2)])
        return And(conds)

    '''
    Get the condition saying a is unchanged from t to t+1
    '''

    def _a_unchange(self, t, except_case=None):
        conds = []
        for item in self.world.action_indices:
            if item != except_case:
                conds.append(self.a[t + 1][item] == self.a[t][item])
        return And(conds)

    '''
    Get the condition saying l is unchanged from t to t+1
    '''

    def _l_unchange(self, t, except_case=None):
        conds = []
        for item in self.world.action_indices:
            if item != except_case:
                conds.append(self.l[t + 1][item] == self.l[t][item])
        return And(conds)

    def _construct_vars_other(self, scenario_id):
        cons = []
        # Define state variables
        self.b = {}
        self.l = {}
        self.a = {}
        for t in range(self.max_steps + 1):
            # Key-lock pairs
            self.b[t] = {}
            for k1 in self.world.action_indices:
                for k2 in self.world.action_indices:
                    if k1 == k2:
                        continue
                    self.b[t][(k1, k2)] = Int("%s_b_%s_%s_%s" %
                                              (scenario_id, t, k1, k2))
                    cons.append(self.b[t][(k1, k2)] >= 0)

            # loose key
            self.l[t] = [Bool("%s_l_%s_%s" % (scenario_id, t, k))
                         for k in self.world.action_indices]
            constr = AtMost(*self.l[t], 1)
            cons.append(constr)

            # Agent's key
            self.a[t] = [Bool("%s_a_%s_%s" % (scenario_id, t, k))
                         for k in self.world.action_indices]
            constr = AtMost(*self.a[t], 1)
            cons.append(constr)

        # Store a handle for this set of variables
        self.b_all[scenario_id] = self.b
        self.l_all[scenario_id] = self.l
        self.a_all[scenario_id] = self.a

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
                    (t, action if action != self.NOOP else "noop"))
            # Cardinality constraints
            constr = self._singleton_constraint(
                [p[t][key] for key in p[t]])
            cons.append(constr)

        return p, And(cons)

    '''
    One and only one var in vars is true
    '''

    def _singleton_constraint(self, vars):
        return And(AtMost(*vars, 1), Or(vars))
