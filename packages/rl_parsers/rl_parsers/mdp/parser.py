from collections import namedtuple

from ply import lex, yacc
from rl_parsers import ParserError
from . import tokrules

import numpy as np


# LEXER

lexer = lex.lex(module=tokrules)


# MDP

MDP = namedtuple('MDP', 'discount, values, states, actions, start, T, R, reset')


# PARSER


class MDP_Parser:
    tokens = tokrules.tokens

    def __init__(self):
        self.discount = None
        self.values = None

        self.states = None
        self.actions = None

        self.start = None
        self.T = None
        self.R = None

        self.reset = None

    def p_error(self, p):
        # TODO send all printsto stderr or smth like that
        print('Parsing Error:', p.lineno, p.lexpos, p.type, p.value)

    def p_mdp(self, p):
        """ mdp : preamble start structure
                | preamble structure """
        self.mdp = MDP(
            discount=self.discount,
            values=self.values,
            states=self.states,
            actions=self.actions,
            start=self.start,
            T=self.T,
            R=self.R,
            reset=self.reset,
        )

    ###

    def p_preamble(self, p):
        """ preamble : preamble_list """
        self.T = np.zeros((self.nactions, self.nstates, self.nstates))
        self.R = np.zeros((self.nactions, self.nstates, self.nstates))
        self.reset = np.zeros((self.nactions, self.nstates), dtype=np.bool)

    def p_preamble_list(self, p):
        """ preamble_list : preamble_list preamble_item
                          | preamble_item """

    def p_preamble_discount(self, p):
        """ preamble_item : DISCOUNT COLON FLOAT """
        self.discount = p[3]

    def p_preamble_values(self, p):
        """ preamble_item : VALUES COLON REWARD
                          | VALUES COLON COST """
        self.values = p[3]

    def p_preamble_states_N(self, p):
        """ preamble_item : STATES COLON INT """
        N = p[3]
        self.states = tuple(range(N))
        self.nstates = N

    def p_preamble_states_names(self, p):
        """ preamble_item : STATES COLON id_list """
        idlist = p[3]
        self.states = tuple(idlist)
        self.nstates = len(idlist)

    def p_preamble_actions_N(self, p):
        """ preamble_item : ACTIONS COLON INT """
        N = p[3]
        self.actions = tuple(range(N))
        self.nactions = N

    def p_preamble_actions_names(self, p):
        """ preamble_item : ACTIONS COLON id_list """
        idlist = p[3]
        self.actions = tuple(idlist)
        self.nactions = len(idlist)

    ###

    def p_start_uniform(self, p):
        """ start : START COLON UNIFORM """
        self.start = np.full(self.nstates, 1 / self.nstates)

    # NOTE reduce/reduce conflict solved by enforcing pmatrix contains at least
    # 2 probabilities
    def p_start_dist(self, p):
        """ start : START COLON pmatrix """
        pm = np.array(p[3])
        pmsum = pm.sum()
        if not np.isclose(pmsum, 1.):
            raise ParserError(
                f'Start distribution is not normalized (sums to {pmsum}).')
        self.start = pm

    def p_start_state(self, p):
        """ start : START COLON state """
        s = p[3]
        self.start = np.zeros(self.nstates)
        self.start[s] = 1

    def p_start_include(self, p):
        """ start : START INCLUDE COLON state_list """
        slist = p[4]
        self.start = np.zeros(self.nstates)
        self.start[slist] = 1 / len(slist)

    def p_start_exclude(self, p):
        """ start : START EXCLUDE COLON state_list """
        slist = p[4]
        self.start = np.full(self.nstates, 1 / (self.nstates - len(slist)))
        self.start[slist] = 0

    ###

    def p_id_list(self, p):
        """ id_list : id_list ID """
        p[0] = p[1] + [p[2]]

    def p_id_list_base(self, p):
        """ id_list : ID """
        p[0] = [p[1]]

    ###

    def p_state_list(self, p):
        """ state_list : state_list state """
        p[0] = p[1] + [p[2]]

    def p_state_list_base(self, p):
        """ state_list : state """
        p[0] = [p[1]]

    ###

    def p_state_idx(self, p):
        """ state : INT """
        p[0] = p[1]

    def p_state_id(self, p):
        """ state : ID """
        p[0] = self.states.index(p[1])

    def p_state_all(self, p):
        """ state : ASTERISK """
        p[0] = slice(None)

    ###

    def p_action_idx(self, p):
        """ action : INT """
        p[0] = p[1]

    def p_action_id(self, p):
        """ action : ID """
        p[0] = self.actions.index(p[1])

    def p_action_all(self, p):
        """ action : ASTERISK """
        p[0] = slice(None)

    ###

    def p_structure(self, p):
        """ structure : structure_list """

    def p_structure_list(self, p):
        """ structure_list : structure_list structure_item
                           | """

    def p_structure_t_ass(self, p):
        """ structure_item : T COLON action COLON state COLON state prob """
        a, s0, s1, prob = p[3], p[5], p[7], p[8]
        self.T[a, s0, s1] = prob

    def p_structure_t_as_uniform(self, p):
        """ structure_item : T COLON action COLON state UNIFORM """
        a, s0 = p[3], p[5]
        self.T[a, s0] = 1 / self.nstates

    def p_structure_t_as_reset(self, p):
        """ structure_item : T COLON action COLON state RESET """
        a, s0 = p[3], p[5]
        self.T[a, s0] = self.start
        self.reset[a, s0] = True

    def p_structure_t_as_dist(self, p):
        """ structure_item : T COLON action COLON state pmatrix """
        a, s0, pm = p[3], p[5], p[6]
        pm = np.array(pm)
        pmsum = pm.sum()
        if not np.isclose(pmsum, 1.):
            raise ParserError(f'Transition distribution (action={a}, '
                              f'state={s0}) is not normalized (sums to '
                              f'{pmsum}).')
        self.T[a, s0] = pm

    def p_structure_t_a_uniform(self, p):
        """ structure_item : T COLON action UNIFORM """
        a = p[3]
        self.T[a] = 1 / self.nstates

    def p_structure_t_a_identity(self, p):
        """ structure_item : T COLON action IDENTITY """
        a = p[3]
        self.T[a] = np.eye(self.nstates)

    def p_structure_t_a_dist(self, p):
        """ structure_item : T COLON action pmatrix """
        a, pm = p[3], p[4]
        pm = np.reshape(pm, (self.nstates, self.nstates))
        if not np.isclose(pm.sum(axis=1), 1.).all():
            raise ParserError(f'Transition state distribution (action={a}) is '
                              'not normalized;')
        self.T[a] = pm

    ###

    def p_structure_r_ass(self, p):
        """ structure_item : R COLON action COLON state COLON state number """
        a, s0, s1, r = p[3], p[5], p[7], p[8]
        self.R[a, s0, s1] = r

    def p_structure_r_as(self, p):
        """ structure_item : R COLON action COLON state nmatrix """
        a, s0, r = p[3], p[5], p[6]
        self.R[a, s0] = r

    ###

    # TODO move elsewhere
    def p_pmatrix_1(self, p):
        """ pmatrix : pmatrix prob """
        p[0] = p[1] + [p[2]]

    # NOTE enforce at least two probabilities;
    # solves reduce/reduce conflict in start_state rule!
    def p_pmatrix_2(self, p):
        """ pmatrix : prob prob """
        p[0] = [p[1], p[2]]

    # def p_pmatrix_2(self, p):
    #     """ pmatrix : prob """
    #     p[0] = [p[1]]

    def p_nmatrix_1(self, p):
        """ nmatrix : nmatrix number """
        p[0] = p[1] + [p[2]]

    def p_nmatrix(self, p):
        """ nmatrix : number """
        p[0] = [p[1]]

    # TODO improve this
    def p_number_1(self, p):
        """ number : PLUS number
                   | MINUS number """
        p[0] = p[2] if p[1] == '+' else -p[2]

    def p_number_2(self, p):
        """ number : FLOAT
                   | INT """
        p[0] = p[1]

    def p_prob(self, p):
        """ prob : FLOAT
                 | INT """
        prob = p[1]
        if not 0 <= prob <= 1:
            raise ParserError(f'Probability value ({prob}) out of bounds.')
        p[0] = prob


def parse(text, **kwargs):
    p = MDP_Parser()
    y = yacc.yacc(module=p)
    y.parse(text, lexer=lexer, **kwargs)
    return p.mdp
