from collections import namedtuple

from ply import lex, yacc
from . import tokrules

import numpy as np


# LEXER

lexer = lex.lex(module=tokrules)


# FSS

FSS = namedtuple('FSS', 'nodes, actions, start, A, N')


# PARSER


class FSS_Parser:
    tokens = tokrules.tokens

    def __init__(self):
        self.nodes = None
        self.actions = None
        self.start = None

        self.A = None
        self.N = None

    def p_error(self, p):
        # TODO something else
        print('Parsing Error:', p.lineno, p.lexpos, p.type, p.value)

    def p_fsc(self, p):
        """ fsc : preamble start structure
                | preamble structure """
        self.fss = FSS(
            nodes=self.nodes,
            actions=self.actions,
            start=self.start,
            A=self.A,
            N=self.N,
        )

    ###

    def p_preamble(self, p):
        """ preamble : preamble_list """
        self.A = np.ones((self.nnodes, self.nactions), dtype=bool)
        self.N = np.ones((self.nnodes, self.nnodes), dtype=bool)

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

    ###

    def p_start(self, p):
        """ start : START COLON node """
        self.start = p[3]

    ###

    def p_id_list(self, p):
        """ id_list : id_list ID """
        p[0] = p[1] + [p[2]]

    def p_id_list_base(self, p):
        """ id_list : ID """
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

    def p_structure(self, p):
        """ structure : structure_list """

    def p_structure_list(self, p):
        """ structure_list : structure_list structure_item
                           | """

    def p_structure_a_na(self, p):
        """ structure_item : A COLON node COLON action bool """
        n, a, b = p[3], p[5], p[6]
        self.A[n, a] = b

    def p_structure_a_n_bm(self, p):
        """ structure_item : A COLON node bmatrix """
        n, bm = p[3], p[4]
        self.A[n] = bm

    def p_structure_a_n_all(self, p):
        """ structure_item : A COLON node ALL """
        n = p[3]
        self.A[n].fill(True)

    def p_structure_a_n_none(self, p):
        """ structure_item : A COLON node NONE """
        n = p[3]
        self.A[n].fill(False)

    # TODO this will not work, because bool can also be an index..
    # def p_structure_a_bm(self, p):
    #     """ structure_item : A COLON bmatrix """
    #     bm = p[3]
    # TODO not reshape.. but...?
    #     self.A = np.reshape(bm, (self.nnodes, self.nactions))

    def p_structure_a_all(self, p):
        """ structure_item : A COLON ALL """
        self.A.fill(True)

    def p_structure_a_none(self, p):
        """ structure_item : A COLON NONE """
        self.A.fill(False)

    def p_structure_n_nn(self, p):
        """ structure_item : N COLON node COLON node bool """
        n, n1, b = p[3], p[5], p[6]
        self.N[n, n1] = b

    def p_structure_n_n_bm(self, p):
        """ structure_item : N COLON node bmatrix """
        n, bm = p[3], p[4]
        self.N[n] = bm

    def p_structure_n_n_all(self, p):
        """ structure_item : N COLON node ALL """
        n = p[3]
        self.N[n].fill(True)

    def p_structure_n_n_none(self, p):
        """ structure_item : N COLON node NONE """
        n = p[3]
        self.N[n].fill(False)

    # NOTE this will not work because bool can also be an index
    # def p_n_structure_5(self, p):
    #     """ structure_item : N COLON bmatrix """
    #     bm = p[3]
    #     self.N = np.reshape(bm, (self.nfactory.nitems, self.nfactory.nitems))

    def p_structure_n_all(self, p):
        """ structure_item : N COLON ALL """
        self.N.fill(True)

    def p_structure_n_none(self, p):
        """ structure_item : N COLON NONE """
        self.N.fill(False)

    ###

    def p_bmatrix_1(self, p):
        """ bmatrix : bmatrix bool """
        p[0] = p[1] + [p[2]]

    def p_bmatrix_2(self, p):
        """ bmatrix : bool """
        p[0] = [p[1]]

    def p_bool(self, p):
        """ bool : INT """
        p[0] = bool(p[1])


def parse(text, **kwargs):
    p = FSS_Parser()
    y = yacc.yacc(module=p)
    y.parse(text, lexer=lexer, **kwargs)
    return p.fss
