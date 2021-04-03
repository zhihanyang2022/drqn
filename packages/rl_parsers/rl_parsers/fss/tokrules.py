tokens = (
    'COLON',
    'ASTERISK',
    'ID',
    'INT',
)

reserved = {
    'nodes': 'NODES',
    'actions': 'ACTIONS',
    'A': 'A',
    'N': 'N',
    'all': 'ALL',
    'none': 'NONE',
    'start': 'START',
}

tokens += tuple(reserved.values())

t_COLON = r':'
t_ASTERISK = r'\*'
# t_BOOL = r'0|1'


def t_INT(t):
    r'[0-9]+'
    t.value = int(t.value)
    return t


def t_ID(t):
    r'[a-zA-Z][a-zA-Z0-9\_\-]*'
    t.type = reserved.get(t.value, 'ID')
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
