tokens = (
    'COLON',
    'ASTERISK',
    'PLUS',
    'MINUS',
    'ID',
    'INT',
    'FLOAT',
)

reserved = {
    'discount': 'DISCOUNT',
    'values': 'VALUES',
    'states': 'STATES',
    'actions': 'ACTIONS',
    'observations': 'OBSERVATIONS',
    'T': 'T',
    'O': 'O',
    'R': 'R',
    'uniform': 'UNIFORM',
    'identity': 'IDENTITY',
    'reward': 'REWARD',
    'cost': 'COST',
    'start': 'START',
    'include': 'INCLUDE',
    'exclude': 'EXCLUDE',
    'reset': 'RESET',
}

tokens += tuple(reserved.values())

t_COLON = r':'
t_ASTERISK = r'\*'
t_PLUS = r'\+'
t_MINUS = r'-'


def t_ID(t):
    r'[a-zA-Z][a-zA-Z0-9\_\-]*'
    t.type = reserved.get(t.value, 'ID')
    return t


def t_NUMBER(t):
    r'[0-9]*\.?[0-9]+((E|e)(\+|-)?[0-9]+)?'
    try:
        t.value = int(t.value)
    except ValueError:
        pass
    else:
        t.type = 'INT'
        return t

    try:
        t.value = float(t.value)
    except ValueError:
        pass
    else:
        t.type = 'FLOAT'
        return t


def t_COMMENT(t):
    r'\#.*'
    pass


t_ignore = ' \t'


# updates line number
def t_newline(t):
    r'\n'
    t.lexer.lineno += 1


def t_error(t):
    print(f'Illegal character \'{t.value[0]}\'')
    t.lexer.skip(1)
