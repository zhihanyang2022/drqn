from collections import namedtuple

from ply import lex, yacc
from rl_parsers import ParserError
from . import tokrules

import numpy as np


# LEXER

lexer = lex.lex(module=tokrules)


# FSC

FSC = namedtuple('FSC', 'nodes, actions, observations, start, A, T')


# PARSER


class FSC_Parser:
    tokens = tokrules.tokens

    def __init__(self):
        self.nodes = None
        self.actions = None
        self.observations = None
        self.start = None

        self.A = None
        self.T = None

    def p_error(self, p):
        # TODO send all printsto stderr or smth like that
        print('Parsing Error:', p.lineno, p.lexpos, p.type, p.value)

    def p_fsc(self, p):
        """ fsc : preamble start structure
                | preamble structure """
        self.fsc = FSC(nodes=self.nodes,
                       actions=self.actions,
                       observations=self.observations,
                       start=self.start,
                       A=self.A,
                       T=self.T
                       )

    ###

    def p_preamble(self, p):
        """ preamble : preamble_list """
        self.A = np.zeros((self.nnodes, self.nactions))
        self.T = np.zeros((self.nobservations, self.nnodes, self.nnodes))

    def p_preamble_list(self, p):
        """ preamble_list : preamble_list preamble_item
                          | preamble_item """

    def p_preamble_nodes_N(self, p):
        """ preamble_item : NODES COLON INT """
        N = p[3]
        self.nodes = tuple(range(N))
        self.nnodes = N

    def p_preamble_nodes_names(self, p):
        """ preamble_item : NODES COLON id_list """
        idlist = p[3]
        self.nodes = tuple(idlist)
        self.nnodes = len(idlist)

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

    def p_preamble_observations_N(self, p):
        """ preamble_item : OBSERVATIONS COLON INT """
        N = p[3]
        self.observations = tuple(range(N))
        self.nobservations = N

    def p_preamble_observations_names(self, p):
        """ preamble_item : OBSERVATIONS COLON id_list """
        idlist = p[3]
        self.observations = tuple(idlist)
        self.nobservations = len(idlist)

    ###

    def p_start_uniform(self, p):
        """ start : START COLON UNIFORM """
        self.start = np.full(self.nnodes, 1 / self.nnodes)

    # NOTE reduce/reduce conflict solved by enforcing pmatrix contains at least
    # 2 probabilities
    def p_start_dist(self, p):
        """ start : START COLON pmatrix """
        pm = np.array(p[3])
        if not np.isclose(pm.sum(), 1.):
            raise ParserError('Start distribution is not normalized (sums to '
                              f'{pm.sum()}).')
        self.start = pm

    def p_start_node(self, p):
        """ start : START COLON node """
        s = p[3]
        self.start = np.zeros(self.nnodes)
        self.start[s] = 1

    def p_start_include(self, p):
        """ start : START INCLUDE COLON node_list """
        slist = p[4]
        self.start = np.zeros(self.nnodes)
        self.start[slist] = 1 / len(slist)

    def p_start_exclude(self, p):
        """ start : START EXCLUDE COLON node_list """
        slist = p[4]
        self.start = np.full(self.nnodes, 1 / (self.nnodes - len(slist)))
        self.start[slist] = 0

    ###

    def p_id_list(self, p):
        """ id_list : id_list ID """
        p[0] = p[1] + [p[2]]

    def p_id_list_base(self, p):
        """ id_list : ID """
        p[0] = [p[1]]

    ###

    def p_node_list(self, p):
        """ node_list : node_list node """
        p[0] = p[1] + [p[2]]

    def p_node_list_base(self, p):
        """ node_list : node """
        p[0] = [p[1]]

    ###

    def p_node_idx(self, p):
        """ node : INT """
        p[0] = p[1]

    def p_node_id(self, p):
        """ node : ID """
        p[0] = self.nodes.index(p[1])

    def p_node_all(self, p):
        """ node : ASTERISK """
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

    def p_observation_idx(self, p):
        """ observation : INT """
        p[0] = p[1]

    def p_observation_id(self, p):
        """ observation : ID """
        p[0] = self.observations.index(p[1])

    def p_observation_all(self, p):
        """ observation : ASTERISK """
        p[0] = slice(None)

    ###

    def p_structure(self, p):
        """ structure : structure_list """

    def p_structure_list(self, p):
        """ structure_list : structure_list structure_item
                           | """

    def p_structure_a_na(self, p):
        """ structure_item : A COLON node COLON action prob """
        n0, a, prob = p[3], p[5], p[6]
        self.A[n0, a] = prob

    def p_structure_a_n_uniform(self, p):
        """ structure_item : A COLON node UNIFORM """
        n0 = p[3]
        self.T[n0] = 1 / self.nnodes

    def p_structure_a_n_dist(self, p):
        """ structure_item : A COLON node pmatrix """
        n0, pm = p[3], p[4]
        pm = np.array(pm)
        if not np.isclose(pm.sum(), 1.):
            raise ParserError(f'Action distribution (node={n0}) is not '
                              f'normalized (sums to {pm.sum()}).')
        self.A[n0] = pm

    def p_structure_t_ass(self, p):
        """ structure_item : T COLON observation COLON node COLON node prob """
        o, n0, n1, prob = p[3], p[5], p[7], p[8]
        self.T[o, n0, n1] = prob

    def p_structure_t_as_uniform(self, p):
        """ structure_item : T COLON observation COLON node UNIFORM """
        o, n0 = p[3], p[5]
        self.T[o, n0] = 1 / self.nnodes

    def p_structure_t_os_reset(self, p):
        """ structure_item : T COLON observation COLON node RESET """
        o, n0 = p[3], p[5]
        self.T[o, n0] = self.start

    def p_structure_t_os_dist(self, p):
        """ structure_item : T COLON observation COLON node pmatrix """
        o, n0, pm = p[3], p[5], p[6]
        pm = np.array(pm)
        if not np.isclose(pm.sum(), 1.):
            raise ParserError(
                f'Transition distribution (observation={o}, ode={n0}) is not '
                f'normalized (sums to {pm.sum()}).')
        self.T[o, n0] = pm

    def p_structure_t_o_uniform(self, p):
        """ structure_item : T COLON observation UNIFORM """
        o = p[3]
        self.T[o] = 1 / self.nnodes

    def p_structure_t_o_identity(self, p):
        """ structure_item : T COLON observation IDENTITY """
        o = p[3]
        self.T[o] = np.eye(self.nnodes)

    def p_structure_t_o_dist(self, p):
        """ structure_item : T COLON observation pmatrix """
        o, pm = p[3], p[4]
        pm = np.reshape(pm, (self.nnodes, self.nnodes))
        if not np.isclose(pm.sum(axis=1), 1.).all():
            raise ParserError(f'Transition node distribution (observation={o})'
                              ' is not normalized;')
        self.T[o] = pm

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

    def p_prob(self, p):
        """ prob : FLOAT
                 | INT """
        prob = p[1]
        if not 0 <= prob <= 1:
            raise ParserError(
                f'Probability value out of bounds;  Is ({prob}) instead.')
        p[0] = prob


def parse(text, **kwargs):
    p = FSC_Parser()
    y = yacc.yacc(module=p)
    y.parse(text, lexer=lexer, **kwargs)
    return p.fsc
