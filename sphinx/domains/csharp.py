""" C# sphinx domain """

import re
import os
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple,
                    TypeVar, Union, cast)

from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttribute, ASTIdAttribute, ASTParenAttribute, ASTBaseBase,
                                 ASTBaseParenExprList, BaseParser, DefinitionError, StringifyTransform,
                                 UnsupportedMultiCharacterCharLiteral, anon_identifier_re,
                                 binary_literal_re, char_literal_re, float_literal_re,
                                 float_literal_suffix_re, hex_literal_re, identifier_re,
                                 integer_literal_re, integers_literal_suffix_re,
                                 octal_literal_re, verify_description_mode)
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode

import requests

logger = logging.getLogger(__name__)
T = TypeVar('T')

udl_identifier_re = re.compile(r'''(?x)
    [a-zA-Z_][a-zA-Z0-9_]*\b   # note, no word boundary in the beginning
''')
_string_re = re.compile(r"[LuU8]?('([^'\\]*(?:\\.[^'\\]*)*)'"
                        r'|"([^"\\]*(?:\\.[^"\\]*)*)")', re.S)
_visibility_re = re.compile(r'\b(public|private|protected|internal)\b')
_array_indexer_re = re.compile(r'\[\s*(?:[a-zA-Z_][a-zA-Z0-9_]*)?\s?(?:[a-zA-Z_][a-zA-Z0-9_]*)?\]')
_property_accessor_re = re.compile(r'\b(init|get|set)\b')
# see https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/
_value_types = [
    'char', 'bool', 'sbyte', 'byte', 'short', 'ushort', 'int', 'uint',
    'long', 'ulong', 'nint', 'nuint', 'float', 'double', 'decimal',
    'enum', 'struct',
]

_reference_types = [
    'object', 'string', 'delegate', 'dynamic', 'record',
    'class', 'interface', 'void', 'var'
]

_modifiers = [
    'public', 'protected', 'internal', 'private', 'abstract', 'async',
    'const', 'event', 'extern', 'in', 'new', 'out', 'override',
    'readonly', 'sealed', 'static', 'unsafe', 'virtual', 'volatile'
]

_statement_keywords = [
    'if', 'else', 'switch', 'case', 'foreach', 'do', 'for', 'foreach', 'while',
    'break', 'continue', 'default', 'goto', 'return', 'yield', 'throw',
    'try', 'catch', 'finally', 'checked', 'unchecked', 'fixed', 'lock'
]

_method_parameters = [ 'in', 'ref', 'out', 'params' ]

_namespace_keywords = [ 'namespace', 'using', 'extern', 'alias' ]

_generic_type_contraint_keywords = [ 'new', 'where', 'notnull', 'unmanaged' ]

_access_keywords = [ 'base', 'this' ]

_literal_keywords = {
    'null': 'LDnE',
    'true': 'L1E',
    'false': 'L0E',
    'default': 'LDdE'
}

_contextual_keywords = [
    'add', 'and', 'async', 'await', 'dynamic', 'get', 'global', 'init', 'nint', 'not', 'nuint', 'or',
    'partial', 'record', 'remove', 'set', 'value', 'var', 'when', 'where', 'yield'
]

_query_keywords = [ 'from', 'where', 'select', 'group', 'into', 'orderby', 'join', 'let', 'in', 'on', 'equals', 'by', 'ascending', 'descending' ]

_operators = [ 'operator', 'explicit', 'implicit', 'sizeof', 'new', 'true', 'false', 'is', 'delegate', 'await', 'as', 'cast', 'typeof' ]

_expressions = [ 'default', 'nameof', 'stackalloc', 'switch', 'with' ]

_function_pointer_keywords = [ 'unsafe', 'managed', 'unmanaged' ]

_keywords = set(_value_types +
                _reference_types +
                _modifiers +
                _statement_keywords +
                _method_parameters +
                _namespace_keywords +
                _generic_type_contraint_keywords +
                _access_keywords +
                list(_literal_keywords.keys()) +
                _contextual_keywords +
                _query_keywords +
                _operators +
                _expressions +
                _function_pointer_keywords
)

_id_fundamental = {
    'void': 'v',
    'bool': 'b',
    'char': 'c',
    'byte': 'h',
    'sbyte': 'a',
    'short': 's',
    'ushort': 't',
    'int': 'i',
    'uint': 'j',
    'nint': 'Ni',
    'nuint': 'Nu',
    'long': 'l',
    'ulong': 'm',
    'float': 'f',
    'double': 'd',
    'decimal': 'D',
    'string': 'str',
    'var': 'Da'
}

_id_builtin_operator = {
    '~': 'co',
    '+': 'pl',
    '-': 'mi',
    '*': 'ml',
    '/': 'dv',
    '%': 'rm',
    '&': 'an',
    '|': 'or',
    '^': 'eo',
    '=': 'aS',
    '+=': 'pL',
    '-=': 'mI',
    '*=': 'mL',
    '/=': 'dV',
    '%=': 'rM',
    '&=': 'aN',
    '|=': 'oR',
    '^=': 'eO',
    '<<': 'ls',
    '>>': 'rs',
    '<<=': 'lS',
    '>>=': 'rS',
    '==': 'eq',
    '!=': 'ne',
    '<': 'lt',
    '>': 'gt',
    '<=': 'le',
    '>=': 'ge',
    '<=>': 'ss',
    '??': 'nc',
    '??=': 'Nc',
    '!': 'nt',
    '&&': 'aa',
    '||': 'oo',
    '++': 'pp',
    '--': 'mm',
    ',': 'cm',
    '()': 'cl',
    '[]': 'ix'
}

_id_operator_unary = {
    '++': 'pp_',
    '--': 'mm_',
    '*': 'de',
    '&': 'ad',
    '+': 'ps',
    '-': 'ng',
    '!': 'nt',
    '~': 'co',
	'^': 'xo',
	'(T)': 'ct'
}

_id_char_from_prefix = {
    None: 'c', 'u8': 'c',
    'u': 'Ds', 'U': 'Di', 'L': 'w'
}  # type: Dict[Any, str]
# these are ordered by preceedence
_expression_bin_ops = [
    ['||'],
    ['&&'],
    ['|'],
    ['^'],
    ['&'],
    ['==', '!='],
    ['<=', '>=', '<', '>'],
    ['<<', '>>'],
    ['+', '-'],
    ['*', '/', '%'],
    ['??']
]
_expression_unary_ops = ["++", "--", "*", "&", "+", "-", "!", "~"]
_expression_assignment_ops = ["=", "*=", "/=", "%=", "+=", "-=", ">>=", "<<=", "&=", "^=", "|=", "??="]

_msdn_url_base = 'https://docs.microsoft.com/en-us/dotnet/api/'
_temp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
_msdn_ref_cache_file = os.path.join(_temp_dir, "msdn_ref_cache")

def get_msdn_ref(name: str, child: Union[Node, List[Node]]) -> nodes.reference:
    """ Try and create a reference to a type on MSDN """
    skip_url_validation = False
    valid_url = False
    url = _msdn_url_base + name.lower()
    if os.path.exists(_msdn_ref_cache_file):
        with open(_msdn_ref_cache_file, 'r') as f:
            valid_url_pattern = re.escape(url) + r"\s--\s(?P<valid_url>\w+)\n"
            valid_url_match = re.search(valid_url_pattern, f.read())
            if valid_url_match:
                skip_url_validation = True
                valid_url = eval(valid_url_match.groupdict()["valid_url"])
    if not skip_url_validation:
        try:
            url_status_code = requests.get(url).status_code
            valid_url = url_status_code == 200
            if not os.path.isdir(_temp_dir):
                os.mkdir(_temp_dir)
            with open(_msdn_ref_cache_file, 'a') as f:
                f.write(url + " -- " + str(valid_url) + "\n")
        except:
            if os.name != 'nt': # Allow ConnectionError for Windows
                raise
            else:
                return None

    if not valid_url:
        return None

    node = nodes.reference('', '', internal=True)
    node['refuri'] = url
    node['reftitle'] = name
    node += child
    return node


class _DuplicateSymbolError(Exception):
    def __init__(self, symbol: "Symbol", declaration: "ASTDeclaration") -> None:
        assert symbol
        assert declaration
        self.symbol = symbol
        self.declaration = declaration

    def __str__(self) -> str:
        return "Internal C# duplicate symbol error:\n%s" % self.symbol.dump(0)


class ASTBase(ASTBaseBase):
    pass


# Names
################################################################################

class ASTIdentifier(ASTBase):
    def __init__(self, identifier: str) -> None:
        assert identifier is not None
        assert len(identifier) != 0
        self.identifier = identifier

    def is_anon(self) -> bool:
        return self.identifier[0] == '@'

    def get_id(self) -> str:
        txt = _array_indexer_re.sub('', self.identifier)
        if self.is_anon():
            return 'Ut%d_%s' % (len(txt) - 1, txt[1:])
        else:
            return str(len(txt)) + txt

    # and this is where we finally make a difference between __str__ and the display string

    def __str__(self) -> str:
        return self.identifier

    def get_display_string(self) -> str:
        return "[anonymous]" if self.is_anon() else self.identifier

    def describe_signature(self, signode: TextElement, mode: str, env: "BuildEnvironment",
                           prefix: str, templateArgs: str, symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if mode == 'markType':
            targetText = prefix + self.identifier + templateArgs
            pnode = addnodes.pending_xref('', refdomain='csharp',
                                          reftype='identifier',
                                          reftarget=targetText, modname=None,
                                          classname=None)
            pnode['csharp:parent_key'] = symbol.get_lookup_key()
            if self.is_anon():
                pnode += nodes.strong(text="[anonymous]")
            else:
                pnode += nodes.Text(self.identifier)
            signode += pnode
        elif mode == 'lastIsName':
            if self.is_anon():
                signode += nodes.strong(text="[anonymous]")
            else:
                signode += addnodes.desc_name(self.identifier, self.identifier)
        elif mode == 'noneIsName':
            if self.is_anon():
                signode += nodes.strong(text="[anonymous]")
            else:
                signode += nodes.Text(self.identifier)
        elif mode == 'udl':
            # the target is 'operator""id' instead of just 'id'
            assert len(prefix) == 0
            assert len(templateArgs) == 0
            assert not self.is_anon()
            targetText = 'operator""' + self.identifier
            pnode = addnodes.pending_xref('', refdomain='csharp',
                                          reftype='identifier',
                                          reftarget=targetText, modname=None,
                                          classname=None)
            pnode['csharp:parent_key'] = symbol.get_lookup_key()
            pnode += nodes.Text(self.identifier)
            signode += pnode
        else:
            raise Exception('Unknown description mode: %s' % mode)


class ASTNestedNameElement(ASTBase):
    def __init__(self, identOrOp: Union[ASTIdentifier, "ASTOperator"],
                 templateArgs: "ASTTemplateArgs") -> None:
        self.identOrOp = identOrOp
        self.templateArgs = templateArgs

    def is_operator(self) -> bool:
        return False

    def get_id(self) -> str:
        res = self.identOrOp.get_id()
        if self.templateArgs:
            res += self.templateArgs.get_id()
        return res

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.identOrOp)
        if self.templateArgs:
            res += transform(self.templateArgs)
        return res

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", prefix: str, symbol: "Symbol") -> None:
        tArgs = str(self.templateArgs) if self.templateArgs is not None else ''
        self.identOrOp.describe_signature(signode, mode, env, prefix, tArgs, symbol)
        if self.templateArgs is not None:
            self.templateArgs.describe_signature(signode, mode, env, symbol)


class ASTNestedName(ASTBase):
    def __init__(self, names: List[ASTNestedNameElement], rooted: bool) -> None:
        assert len(names) > 0
        self.names = names
        self.rooted = rooted

    @property
    def name(self) -> "ASTNestedName":
        return self

    def num_templates(self) -> int:
        count = 0
        for n in self.names:
            if n.is_operator():
                continue
            if n.templateArgs:
                count += 1
        return count

    def get_id(self, modifiers: str = '') -> str:
        res = []
        if len(self.names) > 1 or len(modifiers) > 0:
            res.append('N')
        res.append(modifiers)
        for n in self.names:
            res.append(n.get_id())
        if len(self.names) > 1 or len(modifiers) > 0:
            res.append('E')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.rooted:
            res.append('')
        for i in range(len(self.names)):
            res.append(transform(self.names[i]))
        return '.'.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        # just print the name part, with template args, not template params
        if mode == 'noneIsName':
            if self.rooted:
                assert False, "Can this happen?"  # TODO
                signode += nodes.Text('.')
            signode += nodes.Text(str(self))
        elif mode == 'param':
            name = str(self)
            signode += nodes.emphasis(name, name)
        elif mode == 'markType' or mode == 'lastIsName' or mode == 'markName':
            # Each element should be a pending xref targeting the complete
            # prefix. however, only the identifier part should be a link, such
            # that template args can be a link as well.
            # For 'lastIsName' we should also prepend template parameter lists.
            templateParams = []  # type: List[Any]
            if mode == 'lastIsName':
                assert symbol is not None
            iTemplateParams = 0
            templateParamsPrefix = ''
            prefix = ''
            first = True
            names = self.names[:-1] if mode == 'lastIsName' else self.names
            # If lastIsName, then wrap all of the prefix in a desc_addname,
            # else append directly to signode.
            # NOTE: Breathe previously relied on the prefix being in the desc_addname node,
            #       so it can remove it in inner declarations.
            dest = signode
            if mode == 'lastIsName':
                dest = addnodes.desc_addname()
            if self.rooted:
                prefix += '.'
                if mode == 'lastIsName' and len(names) == 0:
                    signode += addnodes.desc_sig_punctuation('.', '.')
                else:
                    dest += addnodes.desc_sig_punctuation('.', '.')
            for i in range(len(names)):
                nne = names[i]
                if not first:
                    dest += nodes.Text('.')
                    prefix += '.'
                first = False
                txt_nne = str(nne)
                if txt_nne != '':
                    if nne.templateArgs and iTemplateParams < len(templateParams):
                        templateParamsPrefix += str(templateParams[iTemplateParams])
                        iTemplateParams += 1
                    nne.describe_signature(dest, 'markType',
                                           env, templateParamsPrefix + prefix, symbol)
                prefix += txt_nne
            if mode == 'lastIsName':
                if len(self.names) > 1:
                    dest += addnodes.desc_addname('.', '.')
                    signode += dest
                self.names[-1].describe_signature(signode, mode, env, '', symbol)
        else:
            raise Exception('Unknown description mode: %s' % mode)


################################################################################
# Attributes
################################################################################

class ASTCSharpAttribute(ASTAttribute):
    def __init__(self, arg: str) -> None:
        self.arg = arg

    def _stringify(self, transform: StringifyTransform) -> str:
        return "[" + self.arg + "]"

    def describe_signature(self, signode: TextElement) -> None:
        txt = str(self)
        signode.append(nodes.Text(txt, txt))

class ASTNullableAttribute(ASTAttribute):
    def __init__(self) -> None:
        self.nullable_attribute = "?"

    def describe_signature(self, signode: TextElement) -> None:
        signode += addnodes.desc_annotation(self.nullable_attribute, self.nullable_attribute)

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.nullable_attribute


################################################################################
# Modifiers
################################################################################

class ASTModifier(ASTAttribute):
    def __init__(self, mod: str) -> None:
        self.mod = mod

    def describe_signature(self, signode: TextElement) -> None:
        signode += addnodes.desc_annotation(self.mod, self.mod)

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.mod


################################################################################
# Expressions
################################################################################

class ASTExpression(ASTBase):
    def get_id(self) -> str:
        raise NotImplementedError(repr(self))

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        raise NotImplementedError(repr(self))


# Primary expressions
################################################################################

class ASTLiteral(ASTExpression):
    pass


class ASTSimpleLiteral(ASTLiteral):
    def __init__(self, literal: str, id: str) -> None:
        self.literal = literal
        self.id = id

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.literal

    def get_id(self) -> str:
        return self.id

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text(self.literal))


class ASTNumberLiteral(ASTLiteral):
    def __init__(self, data: str) -> None:
        self.data = data

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.data

    def get_id(self) -> str:
        # TODO: floats should be mangled by writing the hex of the binary representation
        return "L%sE" % self.data

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        txt = str(self)
        signode.append(nodes.Text(txt, txt))


class ASTStringLiteral(ASTLiteral):
    def __init__(self, data: str) -> None:
        self.data = data

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.data

    def get_id(self) -> str:
        # note: the length is not really correct with escaping
        return "LA%d_KcE" % (len(self.data) - 2)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        txt = str(self)
        signode.append(nodes.Text(txt, txt))


class ASTCharLiteral(ASTLiteral):
    def __init__(self, prefix: str, data: str) -> None:
        self.prefix = prefix  # may be None when no prefix
        self.data = data
        assert prefix in _id_char_from_prefix
        self.type = _id_char_from_prefix[prefix]
        decoded = data.encode().decode('unicode-escape')
        if len(decoded) == 1:
            self.value = ord(decoded)
        else:
            raise UnsupportedMultiCharacterCharLiteral(decoded)

    def _stringify(self, transform: StringifyTransform) -> str:
        if self.prefix is None:
            return "'" + self.data + "'"
        else:
            return self.prefix + "'" + self.data + "'"

    def get_id(self) -> str:
        # TODO: the ID should be have L E around it
        return self.type + str(self.value)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        txt = str(self)
        signode.append(nodes.Text(txt, txt))


class ASTUserDefinedLiteral(ASTLiteral):
    def __init__(self, literal: ASTLiteral, ident: ASTIdentifier):
        self.literal = literal
        self.ident = ident

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.literal) + transform(self.ident)

    def get_id(self) -> str:
        # mangle as if it was a function call: ident(literal)
        return 'clL_Zli{}E{}E'.format(self.ident.get_id(), self.literal.get_id())

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.literal.describe_signature(signode, mode, env, symbol)
        self.ident.describe_signature(signode, "udl", env, "", "", symbol)


################################################################################

class ASTParenExpr(ASTExpression):
    def __init__(self, expr: ASTExpression):
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        return '(' + transform(self.expr) + ')'

    def get_id(self) -> str:
        return self.expr.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('(', '('))
        self.expr.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(')', ')'))


class ASTIdExpression(ASTExpression):
    def __init__(self, name: ASTNestedName):
        # note: this class is basically to cast a nested name as an expression
        self.name = name

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.name)

    def get_id(self) -> str:
        return self.name.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.name.describe_signature(signode, mode, env, symbol)


# Postfix expressions
################################################################################

class ASTPostfixOp(ASTBase):
    def get_id(self, idPrefix: str) -> str:
        raise NotImplementedError(repr(self))

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        raise NotImplementedError(repr(self))


class ASTPostfixArray(ASTPostfixOp):
    def __init__(self, expr: ASTExpression):
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        return '[' + transform(self.expr) + ']'

    def get_id(self, idPrefix: str) -> str:
        return 'ix' + idPrefix + self.expr.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('['))
        self.expr.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(']'))


class ASTPostfixMember(ASTPostfixOp):
    def __init__(self, name: ASTNestedName):
        self.name = name

    def _stringify(self, transform: StringifyTransform) -> str:
        return '.' + transform(self.name)

    def get_id(self, idPrefix: str) -> str:
        return 'dt' + idPrefix + self.name.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('.'))
        self.name.describe_signature(signode, 'noneIsName', env, symbol)


class ASTPostfixMemberOfPointer(ASTPostfixOp):
    def __init__(self, name: ASTNestedName):
        self.name = name

    def _stringify(self, transform: StringifyTransform) -> str:
        return '->' + transform(self.name)

    def get_id(self, idPrefix: str) -> str:
        return 'pt' + idPrefix + self.name.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('->'))
        self.name.describe_signature(signode, 'noneIsName', env, symbol)


class ASTPostfixInc(ASTPostfixOp):
    def _stringify(self, transform: StringifyTransform) -> str:
        return '++'

    def get_id(self, idPrefix: str) -> str:
        return 'pp' + idPrefix

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('++'))


class ASTPostfixDec(ASTPostfixOp):
    def _stringify(self, transform: StringifyTransform) -> str:
        return '--'

    def get_id(self, idPrefix: str) -> str:
        return 'mm' + idPrefix

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('--'))


class ASTPostfixCallExpr(ASTPostfixOp):
    def __init__(self, lst: Union["ASTParenExprList", "ASTBracedInitList"]) -> None:
        self.lst = lst

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.lst)

    def get_id(self, idPrefix: str) -> str:
        res = ['cl', idPrefix]
        for e in self.lst.exprs:
            res.append(e.get_id())
        res.append('E')
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.lst.describe_signature(signode, mode, env, symbol)


class ASTPostfixExpr(ASTExpression):
    def __init__(self, prefix: "ASTType", postFixes: List[ASTPostfixOp]):
        self.prefix = prefix
        self.postFixes = postFixes

    def _stringify(self, transform: StringifyTransform) -> str:
        res = [transform(self.prefix)]
        for p in self.postFixes:
            res.append(transform(p))
        return ''.join(res)

    def get_id(self) -> str:
        id = self.prefix.get_id()
        for p in self.postFixes:
            id = p.get_id(id)
        return id

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.prefix.describe_signature(signode, mode, env, symbol)
        for p in self.postFixes:
            p.describe_signature(signode, mode, env, symbol)


class ASTTypeId(ASTExpression):
    def __init__(self, typeOrExpr: Union["ASTType", ASTExpression], isType: bool):
        self.typeOrExpr = typeOrExpr
        self.isType = isType

    def _stringify(self, transform: StringifyTransform) -> str:
        return 'typeid(' + transform(self.typeOrExpr) + ')'

    def get_id(self) -> str:
        prefix = 'ti' if self.isType else 'te'
        return prefix + self.typeOrExpr.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('typeid'))
        signode.append(nodes.Text('('))
        self.typeOrExpr.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(')'))


# Unary expressions
################################################################################

class ASTUnaryOpExpr(ASTExpression):
    def __init__(self, op: str, expr: ASTExpression):
        self.op = op
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.op + transform(self.expr)

    def get_id(self) -> str:
        return _id_operator_unary[self.op] + self.expr.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text(self.op))
        self.expr.describe_signature(signode, mode, env, symbol)


class ASTSizeofType(ASTExpression):
    def __init__(self, typ: "ASTType"):
        self.typ = typ

    def _stringify(self, transform: StringifyTransform) -> str:
        return "sizeof(" + transform(self.typ) + ")"

    def get_id(self) -> str:
        return 'st' + self.typ.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('sizeof('))
        self.typ.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(')'))


class ASTSizeofExpr(ASTExpression):
    def __init__(self, expr: ASTExpression):
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        return "sizeof " + transform(self.expr)

    def get_id(self) -> str:
        return 'sz' + self.expr.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('sizeof '))
        self.expr.describe_signature(signode, mode, env, symbol)


class ASTNewExpr(ASTExpression):
    def __init__(self, rooted: bool, isNewTypeId: bool, typ: "ASTType",
                 initList: Union["ASTParenExprList", "ASTBracedInitList"]) -> None:
        self.rooted = rooted
        self.isNewTypeId = isNewTypeId
        self.typ = typ
        self.initList = initList

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.rooted:
            res.append('.')
        res.append('new ')
        # TODO: placement
        if self.isNewTypeId:
            res.append(transform(self.typ))
        else:
            assert False
        if self.initList is not None:
            res.append(transform(self.initList))
        return ''.join(res)

    def get_id(self) -> str:
        # the array part will be in the type mangling, so na is not used
        res = ['nw']
        # TODO: placement
        res.append('_')
        res.append(self.typ.get_id())
        if self.initList is not None:
            res.append(self.initList.get_id())
        else:
            res.append('E')
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        if self.rooted:
            signode.append(nodes.Text('.'))
        signode.append(nodes.Text('new '))
        # TODO: placement
        if self.isNewTypeId:
            self.typ.describe_signature(signode, mode, env, symbol)
        else:
            assert False
        if self.initList is not None:
            self.initList.describe_signature(signode, mode, env, symbol)


# Other expressions
################################################################################

class ASTCastExpr(ASTExpression):
    def __init__(self, typ: "ASTType", expr: ASTExpression):
        self.typ = typ
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['(']
        res.append(transform(self.typ))
        res.append(')')
        res.append(transform(self.expr))
        return ''.join(res)

    def get_id(self) -> str:
        return 'cv' + self.typ.get_id() + self.expr.get_id()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode.append(nodes.Text('('))
        self.typ.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(')'))
        self.expr.describe_signature(signode, mode, env, symbol)


class ASTBinOpExpr(ASTExpression):
    def __init__(self, exprs: List[ASTExpression], ops: List[str]):
        assert len(exprs) > 0
        assert len(exprs) == len(ops) + 1
        self.exprs = exprs
        self.ops = ops

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.exprs[0]))
        for i in range(1, len(self.exprs)):
            res.append(' ')
            res.append(self.ops[i - 1])
            res.append(' ')
            res.append(transform(self.exprs[i]))
        return ''.join(res)

    def get_id(self) -> str:
        res = []
        for i in range(len(self.ops)):
            res.append(_id_builtin_operator[self.ops[i]])
            res.append(self.exprs[i].get_id())
        res.append(self.exprs[-1].get_id())
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.exprs[0].describe_signature(signode, mode, env, symbol)
        for i in range(1, len(self.exprs)):
            signode.append(nodes.Text(' '))
            signode.append(nodes.Text(self.ops[i - 1]))
            signode.append(nodes.Text(' '))
            self.exprs[i].describe_signature(signode, mode, env, symbol)


class ASTBracedInitList(ASTBase):
    def __init__(self, exprs: List[Union[ASTExpression, "ASTBracedInitList"]],
                 trailingComma: bool) -> None:
        self.exprs = exprs
        self.trailingComma = trailingComma

    def get_id(self) -> str:
        return "il%sE" % ''.join(e.get_id() for e in self.exprs)

    def _stringify(self, transform: StringifyTransform) -> str:
        exprs = [transform(e) for e in self.exprs]
        trailingComma = ',' if self.trailingComma else ''
        return '{%s%s}' % (', '.join(exprs), trailingComma)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text('{'))
        first = True
        for e in self.exprs:
            if not first:
                signode.append(nodes.Text(', '))
            else:
                first = False
            e.describe_signature(signode, mode, env, symbol)
        if self.trailingComma:
            signode.append(nodes.Text(','))
        signode.append(nodes.Text('}'))


class ASTAssignmentExpr(ASTExpression):
    def __init__(self, exprs: List[Union[ASTExpression, ASTBracedInitList]], ops: List[str]):
        assert len(exprs) > 0
        assert len(exprs) == len(ops) + 1
        self.exprs = exprs
        self.ops = ops

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.exprs[0]))
        for i in range(1, len(self.exprs)):
            res.append(' ')
            res.append(self.ops[i - 1])
            res.append(' ')
            res.append(transform(self.exprs[i]))
        return ''.join(res)

    def get_id(self) -> str:
        res = []
        for i in range(len(self.ops)):
            res.append(_id_builtin_operator[self.ops[i]])
            res.append(self.exprs[i].get_id())
        res.append(self.exprs[-1].get_id())
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.exprs[0].describe_signature(signode, mode, env, symbol)
        for i in range(1, len(self.exprs)):
            signode.append(nodes.Text(' '))
            signode.append(nodes.Text(self.ops[i - 1]))
            signode.append(nodes.Text(' '))
            self.exprs[i].describe_signature(signode, mode, env, symbol)


class ASTCommaExpr(ASTExpression):
    def __init__(self, exprs: List[ASTExpression]):
        assert len(exprs) > 0
        self.exprs = exprs

    def _stringify(self, transform: StringifyTransform) -> str:
        return ', '.join(transform(e) for e in self.exprs)

    def get_id(self) -> str:
        id_ = _id_builtin_operator[',']
        res = []
        for i in range(len(self.exprs) - 1):
            res.append(id_)
            res.append(self.exprs[i].get_id())
        res.append(self.exprs[-1].get_id())
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.exprs[0].describe_signature(signode, mode, env, symbol)
        for i in range(1, len(self.exprs)):
            signode.append(nodes.Text(', '))
            self.exprs[i].describe_signature(signode, mode, env, symbol)


class ASTFallbackExpr(ASTExpression):
    def __init__(self, expr: str):
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.expr

    def get_id(self) -> str:
        return str(self.expr)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += nodes.Text(self.expr)


################################################################################
# Types
################################################################################

# Things for ASTNestedName
################################################################################

class ASTOperator(ASTBase):
    def __init__(self, op: str) -> None:
        self.op = op

    def is_anon(self) -> bool:
        return False

    def is_operator(self) -> bool:
        return True

    def get_id(self) -> str:
        return self.op

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.op

    def _describe_identifier(self, signode: TextElement, identnode: TextElement,
                             env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += addnodes.desc_sig_keyword(self.op, self.op)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", prefix: str, templateArgs: str,
                           symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if mode == 'lastIsName':
            mainName = addnodes.desc_name()
            self._describe_identifier(mainName, mainName, env, symbol)
            signode += mainName
        elif mode == 'markType':
            targetText = prefix + str(self) + templateArgs
            pnode = addnodes.pending_xref('', refdomain='csharp',
                                          reftype='identifier',
                                          reftarget=targetText, modname=None,
                                          classname=None)
            pnode['csharp:parent_key'] = symbol.get_lookup_key()
            # Render the identifier part, but collapse it into a string
            # and make that the a link to this operator.
            # E.g., if it is 'operator SomeType', then 'SomeType' becomes
            # a link to the operator, not to 'SomeType'.
            container = nodes.literal()
            self._describe_identifier(signode, container, env, symbol)
            txt = container.astext()
            pnode += addnodes.desc_name(txt, txt)
            signode += pnode
        else:
            addName = addnodes.desc_addname()
            self._describe_identifier(addName, addName, env, symbol)
            signode += addName


class ASTOperatorBuiltIn(ASTOperator):
    def __init__(self, op: str) -> None:
        ASTOperator.__init__(self, 'operator')
        self.builtin_op = op

    def get_id(self) -> str:
        if self.builtin_op not in _id_builtin_operator:
            raise Exception('Internal error: Built-in operator "%s" can not '
                            'be mapped to an id.' % self.op)
        return _id_builtin_operator[self.builtin_op]

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.op + ' ' + self.builtin_op

    def _describe_identifier(self, signode: TextElement, identnode: TextElement,
                             env: "BuildEnvironment", symbol: "Symbol") -> None:
        ASTOperator._describe_identifier(self, signode, identnode, env, symbol)
        signode += addnodes.desc_sig_space()
        identnode += addnodes.desc_sig_operator(self.builtin_op, self.builtin_op)


class ASTOperatorLiteral(ASTOperator):
    def __init__(self, identifier: ASTIdentifier) -> None:
        ASTOperator.__init__(self, 'operator')
        self.identifier = identifier

    def get_id(self) -> str:
        return 'li' + self.identifier.get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.op + ' ' + transform(self.identifier)

    def _describe_identifier(self, signode: TextElement, identnode: TextElement,
                             env: "BuildEnvironment", symbol: "Symbol") -> None:
        ASTOperator._describe_identifier(self, signode, identnode, env, symbol)
        signode += addnodes.desc_sig_space()
        self.identifier.describe_signature(identnode, 'markType', env, '', '', symbol)


class ASTTemplateArgConstant(ASTBase):
    def __init__(self, value: ASTExpression) -> None:
        self.value = value

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.value)

    def get_id(self) -> str:
        return 'X' + self.value.get_id() + 'E'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.value.describe_signature(signode, mode, env, symbol)


class ASTTemplateArgs(ASTBase):
    def __init__(self, args: List[Union["ASTType", ASTTemplateArgConstant]]) -> None:
        assert args is not None
        self.args = args

    def get_id(self) -> str:
        res = []
        res.append('I')
        if len(self.args) > 0:
            for a in self.args[:-1]:
                res.append(a.get_id())
            res.append(self.args[-1].get_id())
        res.append('E')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ', '.join(transform(a) for a in self.args)
        return '<' + res + '>'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text('<')
        first = True
        for a in self.args:
            if not first:
                signode += nodes.Text(', ')
            first = False
            a.describe_signature(signode, 'markType', env, symbol=symbol)
        signode += nodes.Text('>')


# Main part of declarations
################################################################################

class ASTTrailingTypeSpec(ASTBase):
    def get_id(self) -> str:
        raise NotImplementedError(repr(self))

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        raise NotImplementedError(repr(self))


class ASTTrailingTypeSpecFundamental(ASTTrailingTypeSpec):
    def __init__(self, name: str) -> None:
        self.name = name

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.name

    def get_id(self) -> str:
        if self.name not in _id_fundamental:
            raise Exception(
                'Semi-internal error: Fundamental type "%s" can not be mapped '
                'to an id. Is it a true fundamental type? If not so, the '
                'parser should have rejected it.' % self.name)
        return _id_fundamental[self.name]

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += nodes.Text(str(self.name))


class ASTTrailingTypeSpecName(ASTTrailingTypeSpec):
    def __init__(self, prefix: str, nestedName: ASTNestedName) -> None:
        self.prefix = prefix
        self.nestedName = nestedName

    @property
    def name(self) -> ASTNestedName:
        return self.nestedName

    def get_id(self) -> str:
        return self.nestedName.get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.prefix:
            res.append(self.prefix)
            res.append(' ')
        res.append(transform(self.nestedName))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        if self.prefix:
            signode += addnodes.desc_annotation(self.prefix, self.prefix)
            signode += nodes.Text(' ')
        self.nestedName.describe_signature(signode, mode, env, symbol=symbol)


class ASTFunctionParameter(ASTBase):
    def __init__(self, arg: Union["ASTTypeWithInit", "ASTTemplateParamConstrainedTypeWithInit"]) -> None:
        self.arg = arg

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C#
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id()
        # else, do the usual
        return self.arg.get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.arg)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.arg.describe_signature(signode, mode, env, symbol=symbol)


class ASTParametersQualifiers(ASTBase):
    def __init__(self, paramMode: str, args: List[ASTFunctionParameter]) -> None:
        self.paramMode = paramMode
        self.args = args

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.args

    def get_modifiers_id(self) -> str:
        return ''

    def get_param_id(self) -> str:
        if len(self.args) == 0:
            return 'v'
        else:
            return ''.join(a.get_id() for a in self.args)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.paramMode in ('method', 'delegate'):
            res.append('(')
        elif self.paramMode == 'property':
            res.append('{')
        else:
            self.fail("Unexpected paramMode %s" % self.paramMode)
        first = True
        for a in self.args:
            if not first:
                res.append(', ')
            first = False
            res.append(str(a))
        if self.paramMode in ('method', 'delegate'):
            res.append(')')
        elif self.paramMode == 'property':
            res.append('}')
        else:
            self.fail("Unexpected paramMode %s" % self.paramMode)
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        paramlist = addnodes.desc_parameterlist()
        for arg in self.args:
            param = addnodes.desc_parameter('', '', noemph=True)
            if mode == 'lastIsName':  # i.e., outer-function params
                arg.describe_signature(param, 'param', env, symbol=symbol)
            else:
                arg.describe_signature(param, 'markType', env, symbol=symbol)
            paramlist += param
        signode += paramlist


class ASTDeclSpecsSimple(ASTBase):
    def __init__(self, modifiers: List[Union[ASTModifier, ASTAttribute]]) -> None:
        self.modifiers = modifiers

    def mergeWith(self, other: "ASTDeclSpecsSimple") -> "ASTDeclSpecsSimple":
        if not other:
            return self
        return ASTDeclSpecsSimple(self.modifiers + other.modifiers)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []  # type: List[str]
        res.extend(transform(modifier) for modifier in self.modifiers)
        return ' '.join(res)

    def describe_signature(self, signode: TextElement) -> None:
        addSpace = False
        for modifier in self.modifiers:
            if addSpace:
                signode += nodes.Text(' ')
            addSpace = True
            modifier.describe_signature(signode)


class ASTDeclSpecs(ASTBase):
    def __init__(self, outer: str,
                 leftSpecs: ASTDeclSpecsSimple, rightSpecs: ASTDeclSpecsSimple,
                 trailing: ASTTrailingTypeSpec) -> None:
        # leftSpecs and rightSpecs are used for output
        # allSpecs are used for id generation
        self.outer = outer
        self.leftSpecs = leftSpecs
        self.rightSpecs = rightSpecs
        self.allSpecs = self.leftSpecs.mergeWith(self.rightSpecs)
        self.trailingTypeSpec = trailing

    def get_id(self) -> str:
        res = []
        if 'volatile' in self.allSpecs.modifiers:
            res.append('V')
        if 'const' in self.allSpecs.modifiers:
            res.append('K')
        if self.trailingTypeSpec is not None:
            res.append(self.trailingTypeSpec.get_id())
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []  # type: List[str]
        l = transform(self.leftSpecs)
        if len(l) > 0:
            res.append(l)
        if self.trailingTypeSpec:
            if len(res) > 0:
                res.append(" ")
            res.append(transform(self.trailingTypeSpec))
            r = str(self.rightSpecs)
            if len(r) > 0:
                if len(res) > 0:
                    res.append(" ")
                res.append(r)
        return "".join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        numChildren = len(signode)
        self.leftSpecs.describe_signature(signode)
        addSpace = len(signode) != numChildren

        if self.trailingTypeSpec:
            if addSpace:
                signode += nodes.Text(' ')
            numChildren = len(signode)
            self.trailingTypeSpec.describe_signature(signode, mode, env,
                                                     symbol=symbol)
            addSpace = len(signode) != numChildren

            if len(str(self.rightSpecs)) > 0:
                if addSpace:
                    signode += nodes.Text(' ')
                self.rightSpecs.describe_signature(signode)


# Declarator
################################################################################

class ASTArray(ASTBase):
    def __init__(self, size: ASTExpression):
        self.size = size

    def _stringify(self, transform: StringifyTransform) -> str:
        if self.size:
            return '[' + transform(self.size) + ']'
        else:
            return '[]'

    def get_id(self) -> str:
        if self.size:
            return 'A' + self.size.get_id() + '_'
        else:
            return 'A_'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text("["))
        if self.size:
            self.size.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text("]"))


class ASTDeclarator(ASTBase):
    @property
    def name(self) -> ASTNestedName:
        raise NotImplementedError(repr(self))

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        raise NotImplementedError(repr(self))

    @property
    def trailingReturn(self) -> "ASTType":
        raise NotImplementedError(repr(self))

    def require_space_after_declSpecs(self) -> bool:
        raise NotImplementedError(repr(self))

    def get_modifiers_id(self) -> str:
        raise NotImplementedError(repr(self))

    def get_param_id(self) -> str:
        raise NotImplementedError(repr(self))

    def get_ptr_suffix_id(self) -> str:
        raise NotImplementedError(repr(self))

    def get_type_id(self, returnTypeId: str) -> str:
        raise NotImplementedError(repr(self))

    def is_function_type(self) -> bool:
        raise NotImplementedError(repr(self))

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        raise NotImplementedError(repr(self))


class ASTDeclaratorNameParamQual(ASTDeclarator):
    def __init__(self, declId: ASTNestedName,
                 arrayOps: List[ASTArray],
                 paramQual: ASTParametersQualifiers) -> None:
        self.declId = declId
        self.arrayOps = arrayOps
        self.paramQual = paramQual

    @property
    def name(self) -> ASTNestedName:
        return self.declId

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.paramQual.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.paramQual.trailingReturn

    # only the modifiers for a method, e.g.,
    def get_modifiers_id(self) -> str:
        # cv-qualifiers
        if self.paramQual:
            return self.paramQual.get_modifiers_id()
        raise Exception("This should only be called on a method: %s" % self)

    def get_param_id(self) -> str:  # only the parameters (if any)
        if self.paramQual:
            return self.paramQual.get_param_id()
        else:
            return ''

    def get_ptr_suffix_id(self) -> str:  # only the array specifiers
        return ''.join(a.get_id() for a in self.arrayOps)

    def get_type_id(self, returnTypeId: str) -> str:
        res = []
        # TOOD: can we actually have both array ops and paramQual?
        res.append(self.get_ptr_suffix_id())
        if self.paramQual:
            res.append(self.get_modifiers_id())
            res.append('F')
            res.append(returnTypeId)
            res.append(self.get_param_id())
            res.append('E')
        else:
            res.append(returnTypeId)
        return ''.join(res)

    # ------------------------------------------------------------------------

    def require_space_after_declSpecs(self) -> bool:
        return self.declId is not None

    def is_function_type(self) -> bool:
        return self.paramQual is not None

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.declId:
            res.append(transform(self.declId))
        for op in self.arrayOps:
            res.append(transform(op))
        if self.paramQual:
            res.append(transform(self.paramQual))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.declId:
            self.declId.describe_signature(signode, mode, env, symbol)
        for op in self.arrayOps:
            op.describe_signature(signode, mode, env, symbol)
        if self.paramQual:
            self.paramQual.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorNameBitField(ASTDeclarator):
    def __init__(self, declId: ASTNestedName, size: ASTExpression):
        self.declId = declId
        self.size = size

    @property
    def name(self) -> ASTNestedName:
        return self.declId

    def get_param_id(self) -> str:  # only the parameters (if any)
        return ''

    def get_ptr_suffix_id(self) -> str:  # only the array specifiers
        return ''

    # ------------------------------------------------------------------------

    def require_space_after_declSpecs(self) -> bool:
        return self.declId is not None

    def is_function_type(self) -> bool:
        return False

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.declId:
            res.append(transform(self.declId))
        res.append(" : ")
        res.append(transform(self.size))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.declId:
            self.declId.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(' : ', ' : '))
        self.size.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorParen(ASTDeclarator):
    def __init__(self, inner: ASTDeclarator, next: ASTDeclarator) -> None:
        assert inner
        assert next
        self.inner = inner
        self.next = next
        # TODO: we assume the name, params, and qualifiers are in inner

    @property
    def name(self) -> ASTNestedName:
        return self.inner.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.inner.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.inner.trailingReturn

    def require_space_after_declSpecs(self) -> bool:
        return True

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['(']
        res.append(transform(self.inner))
        res.append(')')
        res.append(transform(self.next))
        return ''.join(res)

    def get_modifiers_id(self) -> str:
        return self.inner.get_modifiers_id()

    def get_param_id(self) -> str:  # only the parameters (if any)
        return self.inner.get_param_id()

    def get_ptr_suffix_id(self) -> str:
        return self.inner.get_ptr_suffix_id() + \
                self.next.get_ptr_suffix_id()

    def get_type_id(self, returnTypeId: str) -> str:
        # ReturnType (inner)next, so 'inner' returns everything outside
        nextId = self.next.get_type_id(returnTypeId)
        return self.inner.get_type_id(returnTypeId=nextId)

    def is_function_type(self) -> bool:
        return self.inner.is_function_type()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text('(')
        self.inner.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text(')')
        self.next.describe_signature(signode, "noneIsName", env, symbol)


# Type and initializer stuff
##############################################################################################

class ASTParenExprList(ASTBaseParenExprList):
    def __init__(self, exprs: List[Union[ASTExpression, ASTBracedInitList]]) -> None:
        self.exprs = exprs

    def get_id(self) -> str:
        return "pi%sE" % ''.join(e.get_id() for e in self.exprs)

    def _stringify(self, transform: StringifyTransform) -> str:
        exprs = [transform(e) for e in self.exprs]
        return '(%s)' % ', '.join(exprs)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text('('))
        first = True
        for e in self.exprs:
            if not first:
                signode.append(nodes.Text(', '))
            else:
                first = False
            e.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(')'))


class ASTInitializer(ASTBase):
    def __init__(self, value: Union[ASTExpression, ASTBracedInitList],
                 hasAssign: bool = True) -> None:
        self.value = value
        self.hasAssign = hasAssign

    def _stringify(self, transform: StringifyTransform) -> str:
        val = transform(self.value)
        if self.hasAssign:
            return ' = ' + val
        else:
            return val

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.hasAssign:
            signode.append(nodes.Text(' = '))
        self.value.describe_signature(signode, 'markType', env, symbol)


class ASTType(ASTBase):
    def __init__(self, declSpecs: ASTDeclSpecs, decl: ASTDeclarator) -> None:
        assert declSpecs
        assert decl
        self.declSpecs = declSpecs
        self.decl = decl

    @property
    def name(self) -> ASTNestedName:
        return self.decl.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.decl.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.decl.trailingReturn

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        res = []
        res.append(self.declSpecs.get_id())
        if objectType:  # needs the name
            if objectType in ('method', 'delegate', 'property'):  # also modifiers
                modifiers = self.decl.get_modifiers_id()
                res.append(symbol.get_full_nested_name().get_id(modifiers))
                res.append(self.decl.get_param_id())
            elif objectType == 'type':  # just the name
                res.append(symbol.get_full_nested_name().get_id())
            else:
                print(objectType)
                assert False
        else:  # only type encoding
            # the 'returnType' of a non-function type is simply just the last
            # type, i.e., for 'int*' it is 'int'
            returnTypeId = self.declSpecs.get_id()
            typeId = self.decl.get_type_id(returnTypeId)
            res.append(typeId)
        res.append('Type')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        declSpecs = transform(self.declSpecs)
        res.append(declSpecs)
        if self.decl.require_space_after_declSpecs() and len(declSpecs) > 0:
            res.append(' ')
        res.append(transform(self.decl))
        return ''.join(res)

    def get_type_declaration_prefix(self) -> str:
        if self.declSpecs.trailingTypeSpec:
            return 'typedef'
        else:
            return 'type'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.declSpecs.describe_signature(signode, 'markType', env, symbol)
        if (self.decl.require_space_after_declSpecs() and
                len(str(self.declSpecs)) > 0):
            signode += nodes.Text(' ')
        # for parameters that don't really declare new names we get 'markType',
        # this should not be propagated, but be 'noneIsName'.
        if mode == 'markType':
            mode = 'noneIsName'
        self.decl.describe_signature(signode, mode, env, symbol)


class ASTTemplateParamConstrainedTypeWithInit(ASTBase):
    def __init__(self, type: ASTType, init: ASTType) -> None:
        assert type
        self.type = type
        self.init = init

    @property
    def name(self) -> ASTNestedName:
        return self.type.name

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C#
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id()
        else:
            return self.type.get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.type)
        if self.init:
            res += " = "
            res += transform(self.init)
        return res

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.type.describe_signature(signode, mode, env, symbol)
        if self.init:
            signode += nodes.Text(" = ")
            self.init.describe_signature(signode, mode, env, symbol)


class ASTTypeWithInit(ASTBase):
    def __init__(self, type: ASTType, init: ASTInitializer) -> None:
        self.type = type
        self.init = init

    @property
    def name(self) -> ASTNestedName:
        return self.type.name

    def get_id(self, objectType: str = None,
               symbol: "Symbol" = None) -> str:
        if objectType != 'member':
            return self.type.get_id(objectType)
        return symbol.get_full_nested_name().get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.type))
        if self.init:
            res.append(transform(self.init))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.type.describe_signature(signode, mode, env, symbol)
        if self.init:
            self.init.describe_signature(signode, mode, env, symbol)


class ASTTypeUsing(ASTBase):
    def __init__(self, name: ASTNestedName, type: ASTType) -> None:
        self.name = name
        self.type = type

    def get_id(self, objectType: str = None,
               symbol: "Symbol" = None) -> str:
        return symbol.get_full_nested_name().get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if self.type:
            res.append(' = ')
            res.append(transform(self.type))
        return ''.join(res)

    def get_type_declaration_prefix(self) -> str:
        return 'using'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)
        if self.type:
            signode += nodes.Text(' = ')
            self.type.describe_signature(signode, 'markType', env, symbol=symbol)


# Other declarations
##############################################################################################

class ASTAccessor(ASTBase):
    def __init__(self, accessor: str, visibility: str) -> None:
        self.accessor = accessor
        self.visibility = visibility

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []

        if self.visibility is not None:
            res.append(self.visibility)
            res.append(' ')
        res.append(self.accessor)
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.visibility is not None:
            signode += addnodes.desc_addname(self.visibility, self.visibility)
            signode += nodes.Text(' ')
        signode += addnodes.desc_addname(self.accessor, self.accessor)


class ASTProperty(ASTBase):
    def __init__(self, declSpecs: ASTDeclSpecs, decl: ASTDeclarator, accessors: List[ASTAccessor]) -> None:
        self.type = ASTType(declSpecs, decl)
        self.accessors = accessors

    @property
    def name(self) -> ASTNestedName:
        return self.type.name

    def get_id(self, objectType: str, symbol: "Symbol") -> str:
        return self.type.get_id() + 'Property'

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.type))
        res.append(' { ')
        if len(self.accessors) > 0:
            for a in self.accessors:
                res.append(transform(a))
                res.append('; ')
        res.append('}')
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.type.describe_signature(signode, mode, env, symbol=symbol)
        signode += nodes.Text(' { ')
        if len(self.accessors) > 0:
            for a in self.accessors:
                a.describe_signature(signode, mode, env, symbol=symbol)
                signode += nodes.Text('; ')
        signode += nodes.Text('}')


class ASTBaseClass(ASTBase):
    def __init__(self, name: ASTNestedName, visibility: str,
                 virtual: bool) -> None:
        self.name = name
        self.visibility = visibility
        self.virtual = virtual

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []

        if self.virtual:
            res.append('virtual ')
        if self.visibility is not None:
            res.append(self.visibility)
            res.append(' ')
        res.append(transform(self.name))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.virtual:
            signode += addnodes.desc_sig_keyword('virtual', 'virtual')
            signode += addnodes.desc_sig_space()
        if self.visibility is not None:
            signode += addnodes.desc_sig_keyword(self.visibility,
                                                 self.visibility)
            signode += addnodes.desc_sig_space()
        self.name.describe_signature(signode, 'markType', env, symbol=symbol)


class ASTConstrainedType(ASTBase):
    def __init__(self, type_name: ASTNestedName, constaints: List[ASTType]) -> None:
        self.type_name = type_name
        self.constraints = constaints

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append('where ')
        res.append(transform(self.type_name))
        res.append(' : ')
        assert len(self.constraints) > 0
        res.append(', '.join(transform(self.constraints)))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += addnodes.desc_annotation('where', 'where')
        signode += nodes.Text(' ')
        self.type_name.describe_signature(signode, mode, env, symbol=symbol)
        signode += nodes.Text(' : ')
        first = True
        for constraint in self.constraints:
            if not first:
                signode += nodes.Text(', ')
            constraint.describe_signature(signode, 'markType', env, symbol=symbol)
            first = False


class ASTClass(ASTBase):
    def __init__(self, name: ASTNestedName, bases: List[ASTBaseClass], type_constraints: List[ASTConstrainedType]) -> None:
        self.name = name
        self.bases = bases
        self.type_constraints = type_constraints

    def get_id(self, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id() + 'Class'

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if len(self.bases) > 0:
            res.append(' : ')
            first = True
            for b in self.bases:
                if not first:
                    res.append(', ')
                first = False
                res.append(transform(b))
        if len(self.type_constraints) > 0:
            for tc in self.type_constraints:
                res.append(' ')
                res.append(transform(tc))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)
        if len(self.bases) > 0:
            signode += nodes.Text(' : ')
            for b in self.bases:
                b.describe_signature(signode, mode, env, symbol=symbol)
                signode += nodes.Text(', ')
            signode.pop()
        if len(self.type_constraints) > 0:
            for tc in self.type_constraints:
                signode += nodes.Text(' ')
                tc.describe_signature(signode, mode, env, symbol=symbol)


class ASTMethod(ASTBase):
    def __init__(self, declSpecs: ASTDeclSpecs, decl: ASTDeclarator, type_constraints: List[ASTConstrainedType]) -> None:
        self.declSpecs = declSpecs
        self.decl = decl
        self.type_constraints = type_constraints

    @property
    def name(self) -> ASTNestedName:
        return self.decl.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.decl.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.decl.trailingReturn

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        res = []
        res.append(self.declSpecs.get_id())
        modifiers = self.decl.get_modifiers_id()
        res.append(symbol.get_full_nested_name().get_id(modifiers))
        res.append(self.decl.get_param_id())
        res.append('Method')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        declSpecs = transform(self.declSpecs)
        res.append(declSpecs)
        if self.decl.require_space_after_declSpecs() and len(declSpecs) > 0:
            res.append(' ')
        res.append(transform(self.decl))
        if len(self.type_constraints) > 0:
            for tc in self.type_constraints:
                res.append(' ')
                res.append(transform(tc))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.declSpecs.describe_signature(signode, 'markType', env, symbol)
        if (self.decl.require_space_after_declSpecs() and
                len(str(self.declSpecs)) > 0):
            signode += nodes.Text(' ')
        # for parameters that don't really declare new names we get 'markType',
        # this should not be propagated, but be 'noneIsName'.
        if mode == 'markType':
            mode = 'noneIsName'
        self.decl.describe_signature(signode, mode, env, symbol)
        if len(self.type_constraints) > 0:
            for tc in self.type_constraints:
                signode += nodes.Text(' ')
                tc.describe_signature(signode, mode, env, symbol=symbol)


class ASTUnion(ASTBase):
    def __init__(self, name: ASTNestedName) -> None:
        self.name = name

    def get_id(self, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id() + 'Union'

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.name)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)


class ASTEnum(ASTBase):
    def __init__(self, name: ASTNestedName, scoped: str,
                 underlyingType: ASTType) -> None:
        self.name = name
        self.scoped = scoped
        self.underlyingType = underlyingType

    def get_id(self, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id() + 'Enum'

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.scoped:
            res.append(self.scoped)
            res.append(' ')
        res.append(transform(self.name))
        if self.underlyingType:
            res.append(' : ')
            res.append(transform(self.underlyingType))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        # self.scoped has been done by the CSharpEnumObject
        self.name.describe_signature(signode, mode, env, symbol=symbol)
        if self.underlyingType:
            signode += nodes.Text(' : ')
            self.underlyingType.describe_signature(signode, 'noneIsName',
                                                   env, symbol=symbol)


class ASTEnumerator(ASTBase):
    def __init__(self, name: ASTNestedName, init: ASTInitializer) -> None:
        self.name = name
        self.init = init

    def get_id(self, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id() + 'Enumerator'

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if self.init:
            res.append(transform(self.init))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol)
        if self.init:
            self.init.describe_signature(signode, 'markType', env, symbol)


################################################################################
# Templates
################################################################################

# Parameters
################################################################################

class ASTTemplateParam(ASTBase):
    def get_identifier(self) -> ASTIdentifier:
        raise NotImplementedError(repr(self))

    def get_id(self) -> str:
        raise NotImplementedError(repr(self))

    def describe_signature(self, parentNode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        raise NotImplementedError(repr(self))


class ASTTemplateKeyParamPackIdDefault(ASTTemplateParam):
    def __init__(self, key: str, identifier: ASTIdentifier) -> None:
        assert key
        self.key = key
        self.identifier = identifier

    def get_identifier(self) -> ASTIdentifier:
        return self.identifier

    def get_id(self) -> str:
        # this is not part of the normal name mangling in C#
        res = []
        res.append('0')  # we need to put something
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = [self.key]
        if self.identifier:
            res.append(' ')
            res.append(transform(self.identifier))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += nodes.Text(self.key)
        if self.identifier:
            signode += nodes.Text(' ')
            self.identifier.describe_signature(signode, mode, env, '', '', symbol)


class ASTTemplateParamType(ASTTemplateParam):
    def __init__(self, data: ASTTemplateKeyParamPackIdDefault) -> None:
        assert data
        self.data = data

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], rooted=False)

    def get_identifier(self) -> ASTIdentifier:
        return self.data.get_identifier()

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C#
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id()
        else:
            return self.data.get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.data)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.data.describe_signature(signode, mode, env, symbol)


class ASTTemplateParamTemplateType(ASTTemplateParam):
    def __init__(self, nestedParams: "ASTTemplateParams",
                 data: ASTTemplateKeyParamPackIdDefault) -> None:
        assert nestedParams
        assert data
        self.nestedParams = nestedParams
        self.data = data

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], rooted=False)

    def get_identifier(self) -> ASTIdentifier:
        return self.data.get_identifier()

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C#
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id()
        else:
            return self.nestedParams.get_id() + self.data.get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.nestedParams) + transform(self.data)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.nestedParams.describe_signature(signode, 'noneIsName', env, symbol)
        signode += nodes.Text(' ')
        self.data.describe_signature(signode, mode, env, symbol)


class ASTTemplateParamNonType(ASTTemplateParam):
    def __init__(self,
                 param: Union[ASTTypeWithInit,
                              ASTTemplateParamConstrainedTypeWithInit]) -> None:
        assert param
        self.param = param

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], rooted=False)

    def get_identifier(self) -> ASTIdentifier:
        name = self.param.name
        if name:
            assert len(name.names) == 1
            assert name.names[0].identOrOp
            assert not name.names[0].templateArgs
            res = name.names[0].identOrOp
            assert isinstance(res, ASTIdentifier)
            return res
        else:
            return None

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C#
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id()
        else:
            return '_' + self.param.get_id()

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.param)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.param.describe_signature(signode, mode, env, symbol)


class ASTTemplateParams(ASTBase):
    def __init__(self, params: List[ASTTemplateParam]) -> None:
        assert params is not None
        self.params = params

    def get_id(self) -> str:
        res = []
        res.append("I")
        for param in self.params:
            res.append(param.get_id())
        res.append("E")
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append("template<")
        res.append(", ".join(transform(a) for a in self.params))
        res.append("> ")
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += nodes.Text("template<")
        first = True
        for param in self.params:
            if not first:
                signode += nodes.Text(", ")
            first = False
            param.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text(">")

    def describe_signature_as_introducer(
            self, parentNode: desc_signature, mode: str, env: "BuildEnvironment",
            symbol: "Symbol", lineSpec: bool) -> None:
        def makeLine(parentNode: desc_signature) -> addnodes.desc_signature_line:
            signode = addnodes.desc_signature_line()
            parentNode += signode
            signode.sphinx_line_type = 'templateParams'
            return signode
        lineNode = makeLine(parentNode)
        lineNode += nodes.Text("template<")
        first = True
        for param in self.params:
            if not first:
                lineNode += nodes.Text(", ")
            first = False
            if lineSpec:
                lineNode = makeLine(parentNode)
            param.describe_signature(lineNode, mode, env, symbol)
        if lineSpec and not first:
            lineNode = makeLine(parentNode)
        lineNode += nodes.Text(">")


# Template introducers
################################################################################

class ASTTemplateIntroductionParameter(ASTBase):
    def __init__(self, identifier: ASTIdentifier) -> None:
        self.identifier = identifier

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], rooted=False)

    def get_identifier(self) -> ASTIdentifier:
        return self.identifier

    def get_id(self, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C#
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id()
        else:
            return '0'  # we need to put something

    def get_id_as_arg(self) -> str:
        # used for the implicit requires clause
        res = self.identifier.get_id()
        return res

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.identifier))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.identifier.describe_signature(signode, mode, env, '', '', symbol)


class ASTTemplateIntroduction(ASTBase):
    def __init__(self, concept: ASTNestedName,
                 params: List[ASTTemplateIntroductionParameter]) -> None:
        assert len(params) > 0
        self.concept = concept
        self.params = params

    def get_id(self) -> str:
        # first do the same as a normal template parameter list
        res = []
        res.append("I")
        for param in self.params:
            res.append(param.get_id())
        res.append("E")
        # let's use X expr E, which is otherwise for constant template args
        res.append("X")
        res.append(self.concept.get_id())
        res.append("I")
        for param in self.params:
            res.append(param.get_id_as_arg())
        res.append("E")
        res.append("E")
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.concept))
        res.append('{')
        res.append(', '.join(transform(param) for param in self.params))
        res.append('} ')
        return ''.join(res)

    def describe_signature_as_introducer(
            self, parentNode: desc_signature, mode: str,
            env: "BuildEnvironment", symbol: "Symbol", lineSpec: bool) -> None:
        # Note: 'lineSpec' has no effect on template introductions.
        signode = addnodes.desc_signature_line()
        parentNode += signode
        signode.sphinx_line_type = 'templateIntroduction'
        self.concept.describe_signature(signode, 'markType', env, symbol)
        signode += nodes.Text('{')
        first = True
        for param in self.params:
            if not first:
                signode += nodes.Text(', ')
            first = False
            param.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text('}')


################################################################################
################################################################################

class ASTDeclaration(ASTBase):
    def __init__(self, objectType: str, directiveType: str, visibility: str,
                 declaration: Any,  semicolon: bool = False) -> None:
        self.objectType = objectType
        self.directiveType = directiveType
        self.visibility = visibility
        self.declaration = declaration
        self.semicolon = semicolon

        self.symbol = None  # type: Symbol
        # set by CSharpObject._add_enumerator_to_parent
        self.enumeratorScopedSymbol: Symbol = None

    def clone(self) -> "ASTDeclaration":
        return ASTDeclaration(self.objectType, self.directiveType, self.visibility,
                              self.declaration.clone(), self.semicolon)

    @property
    def name(self) -> ASTNestedName:
        return self.declaration.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        if not self.objectType in ('method', 'delegate'):
            return None
        return self.declaration.function_params

    def get_id(self) -> str:
        if self.objectType == 'enumerator' and self.enumeratorScopedSymbol:
            return self.enumeratorScopedSymbol.declaration.get_id()
        res = ['_CS']
        res.append(self.declaration.get_id(self.objectType, self.symbol))
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.visibility and self.visibility != "public":
            res.append(self.visibility)
            res.append(' ')
        if self.semicolon:
            res.append(';')
        return ''.join(res)

    def describe_signature(self, signode: desc_signature, mode: str,
                           env: "BuildEnvironment", options: Dict) -> None:
        verify_description_mode(mode)
        assert self.symbol
        # The caller of the domain added a desc_signature node.
        # Always enable multiline:
        signode['is_multiline'] = True
        # Put each line in a desc_signature_line node.
        mainDeclNode = addnodes.desc_signature_line()
        mainDeclNode.sphinx_line_type = 'declarator'
        mainDeclNode['add_permalink'] = not self.symbol.isRedeclaration

        signode += mainDeclNode
        if self.visibility and self.visibility != "public":
            mainDeclNode += addnodes.desc_sig_keyword(self.visibility, self.visibility)
            mainDeclNode += addnodes.desc_sig_space()
        if self.objectType == 'type':
            prefix = self.declaration.get_type_declaration_prefix()
            mainDeclNode += addnodes.desc_sig_keyword(prefix, prefix)
            mainDeclNode += addnodes.desc_sig_space()
        elif self.objectType == 'member':
            pass
        elif self.objectType == 'method':
            if self.directiveType == 'delegate':
                prefix = 'delegate '
                mainDeclNode += addnodes.desc_annotation(prefix, prefix)
            elif self.directiveType == 'method':
                pass
            else:
                assert False  # wrong directiveType used
        elif self.objectType == 'property':
            pass
        elif self.objectType == 'class':
            assert self.directiveType in ('interface', 'class', 'struct')
            mainDeclNode += addnodes.desc_sig_keyword(self.directiveType, self.directiveType)
            mainDeclNode += addnodes.desc_sig_space()
        elif self.objectType == 'enum':
            mainDeclNode += addnodes.desc_sig_keyword('enum', 'enum')
            mainDeclNode += addnodes.desc_sig_space()
        elif self.objectType == 'enumerator':
            mainDeclNode += addnodes.desc_sig_keyword('enumerator', 'enumerator')
            mainDeclNode += addnodes.desc_sig_space()
        else:
            print(self.objectType)
            assert False
        self.declaration.describe_signature(mainDeclNode, mode, env, self.symbol)
        lastDeclNode = mainDeclNode
        if self.semicolon:
            lastDeclNode += addnodes.desc_sig_punctuation(';', ';')


class ASTNamespace(ASTBase):
    def __init__(self, nestedName: ASTNestedName) -> None:
        self.nestedName = nestedName

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.nestedName))
        return ''.join(res)


class SymbolLookupResult:
    def __init__(self, symbols: Iterator["Symbol"], parentSymbol: "Symbol",
                 identOrOp: Union[ASTIdentifier, ASTOperator], templateParams: Any,
                 templateArgs: ASTTemplateArgs) -> None:
        self.symbols = symbols
        self.parentSymbol = parentSymbol
        self.identOrOp = identOrOp
        self.templateParams = templateParams
        self.templateArgs = templateArgs


class LookupKey:
    def __init__(self, data: List[Tuple[ASTNestedNameElement,
                                        Union[ASTTemplateParams,
                                              ASTTemplateIntroduction],
                                        str]]) -> None:
        self.data = data


class Symbol:
    debug_indent = 0
    debug_indent_string = "  "
    debug_lookup = False  # overridden by the corresponding config value
    debug_show_tree = False  # overridden by the corresponding config value

    def __copy__(self):
        assert False  # shouldn't happen

    def __deepcopy__(self, memo):
        if self.parent:
            assert False  # shouldn't happen
        else:
            # the domain base class makes a copy of the initial data, which is fine
            return Symbol(None, None, None, None, None, None, None)

    @staticmethod
    def debug_print(*args: Any) -> None:
        print(Symbol.debug_indent_string * Symbol.debug_indent, end="")
        print(*args)

    def _assert_invariants(self) -> None:
        if not self.parent:
            # parent == None means global scope, so declaration means a parent
            assert not self.identOrOp
            assert not self.templateParams
            assert not self.templateArgs
            assert not self.declaration
            assert not self.docname
        else:
            if self.declaration:
                assert self.docname

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "children":
            assert False
        else:
            return super().__setattr__(key, value)

    def __init__(self, parent: "Symbol", identOrOp: Union[ASTIdentifier, ASTOperator],
                 templateParams: Union[ASTTemplateParams, ASTTemplateIntroduction],
                 templateArgs: Any, declaration: ASTDeclaration,
                 docname: str, line: int) -> None:
        self.parent = parent
        # declarations in a single directive are linked together
        self.siblingAbove = None  # type: Symbol
        self.siblingBelow = None  # type: Symbol
        self.identOrOp = identOrOp
        self.templateParams = templateParams  # template<templateParams>
        self.templateArgs = templateArgs  # identifier<templateArgs>
        self.declaration = declaration
        self.docname = docname
        self.line = line
        self.isRedeclaration = False
        self._assert_invariants()

        # Remember to modify Symbol.remove if modifications to the parent change.
        self._children = []  # type: List[Symbol]
        self._anonChildren = []  # type: List[Symbol]
        # note: _children includes _anonChildren
        if self.parent:
            self.parent._children.append(self)
        if self.declaration:
            self.declaration.symbol = self

        # Do symbol addition after self._children has been initialized.
        self._add_template_and_function_params()

    def _fill_empty(self, declaration: ASTDeclaration, docname: str, line: int) -> None:
        self._assert_invariants()
        assert self.declaration is None
        assert self.docname is None
        assert self.line is None
        assert declaration is not None
        assert docname is not None
        assert line is not None
        self.declaration = declaration
        self.declaration.symbol = self
        self.docname = docname
        self.line = line
        self._assert_invariants()
        # and symbol addition should be done as well
        self._add_template_and_function_params()

    def _add_template_and_function_params(self) -> None:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_add_template_and_function_params:")
        # Note: we may be called from _fill_empty, so the symbols we want
        #       to add may actually already be present (as empty symbols).

        # add symbols for the template params
        if self.templateParams:
            for tp in self.templateParams.params:
                if not tp.get_identifier():
                    continue
                # only add a declaration if we our self are from a declaration
                if self.declaration:
                    decl = ASTDeclaration('templateParam', None, None, tp)
                else:
                    decl = None
                nne = ASTNestedNameElement(tp.get_identifier(), None)
                nn = ASTNestedName([nne], rooted=False)
                self._add_symbols(nn, [], decl, self.docname, self.line)
        # add symbols for function parameters, if any
        if self.declaration is not None and self.declaration.function_params is not None:
            for fp in self.declaration.function_params:
                if fp.arg is None:
                    continue
                nn = fp.arg.name
                if nn is None:
                    continue
                # (comparing to the template params: we have checked that we are a declaration)
                decl = ASTDeclaration('functionParam', None, None, fp)
                assert not nn.rooted
                assert len(nn.names) == 1
                self._add_symbols(nn, [], decl, self.docname, self.line)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1

    def remove(self) -> None:
        if self.parent is None:
            return
        assert self in self.parent._children
        self.parent._children.remove(self)
        self.parent = None

    def clear_doc(self, docname: str) -> None:
        newChildren = []  # type: List[Symbol]
        for sChild in self._children:
            sChild.clear_doc(docname)
            if sChild.declaration and sChild.docname == docname:
                sChild.declaration = None
                sChild.docname = None
                sChild.line = None
                if sChild.siblingAbove is not None:
                    sChild.siblingAbove.siblingBelow = sChild.siblingBelow
                if sChild.siblingBelow is not None:
                    sChild.siblingBelow.siblingAbove = sChild.siblingAbove
                sChild.siblingAbove = None
                sChild.siblingBelow = None
            newChildren.append(sChild)
        self._children = newChildren

    def get_all_symbols(self) -> Iterator[Any]:
        yield self
        for sChild in self._children:
            yield from sChild.get_all_symbols()

    @property
    def children_recurse_anon(self) -> Generator["Symbol", None, None]:
        for c in self._children:
            yield c
            if not c.identOrOp.is_anon():
                continue

            yield from c.children_recurse_anon

    def get_lookup_key(self) -> "LookupKey":
        # The pickle files for the environment and for each document are distinct.
        # The environment has all the symbols, but the documents has xrefs that
        # must know their scope. A lookup key is essentially a specification of
        # how to find a specific symbol.
        symbols = []
        s = self
        while s.parent:
            symbols.append(s)
            s = s.parent
        symbols.reverse()
        key = []
        for s in symbols:
            nne = ASTNestedNameElement(s.identOrOp, s.templateArgs)
            if s.declaration is not None:
                key.append((nne, s.templateParams, s.declaration.get_id()))
            else:
                key.append((nne, s.templateParams, None))
        return LookupKey(key)

    def get_full_nested_name(self) -> ASTNestedName:
        symbols = []
        s = self
        while s.parent:
            symbols.append(s)
            s = s.parent
        symbols.reverse()
        names = []
        templates = []
        for s in symbols:
            names.append(ASTNestedNameElement(s.identOrOp, s.templateArgs))
            templates.append(False)
        return ASTNestedName(names, rooted=False)

    def _find_first_named_symbol(self, identOrOp: Union[ASTIdentifier, ASTOperator],
                                 templateParams: Any, templateArgs: ASTTemplateArgs,
                                 templateShorthand: bool, matchSelf: bool,
                                 recurseInAnon: bool, correctPrimaryTemplateArgs: bool) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_print("_find_first_named_symbol ->")
        res = self._find_named_symbols(identOrOp, templateParams, templateArgs,
                                       templateShorthand, matchSelf, recurseInAnon,
                                       correctPrimaryTemplateArgs,
                                       searchInSiblings=False)
        try:
            return next(res)
        except StopIteration:
            return None

    def _find_named_symbols(self, identOrOp: Union[ASTIdentifier, ASTOperator],
                            templateParams: Any, templateArgs: ASTTemplateArgs,
                            templateShorthand: bool, matchSelf: bool,
                            recurseInAnon: bool, correctPrimaryTemplateArgs: bool,
                            searchInSiblings: bool) -> Iterator["Symbol"]:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_find_named_symbols:")
            Symbol.debug_indent += 1
            Symbol.debug_print("self:")
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_print("identOrOp:                  ", identOrOp)
            Symbol.debug_print("templateParams:             ", templateParams)
            Symbol.debug_print("templateArgs:               ", templateArgs)
            Symbol.debug_print("templateShorthand:          ", templateShorthand)
            Symbol.debug_print("matchSelf:                  ", matchSelf)
            Symbol.debug_print("recurseInAnon:              ", recurseInAnon)
            Symbol.debug_print("correctPrimaryTemplateArgs: ", correctPrimaryTemplateArgs)
            Symbol.debug_print("searchInSiblings:           ", searchInSiblings)

        def isSpecialization() -> bool:
            # the names of the template parameters must be given exactly as args
            # and params that are packs must in the args be the name expanded
            if len(templateParams.params) != len(templateArgs.args):
                return True
            # having no template params and no arguments is also a specialization
            if len(templateParams.params) == 0:
                return True
            for i in range(len(templateParams.params)):
                param = templateParams.params[i]
                arg = templateArgs.args[i]
                # TODO: doing this by string manipulation is probably not the most efficient
                paramName = str(param.name)
                argTxt = str(arg)
                if paramName != argTxt:
                    return True
            return False
        if correctPrimaryTemplateArgs:
            if templateParams is not None and templateArgs is not None:
                # If both are given, but it's not a specialization, then do lookup as if
                # there is no argument list.
                # For example: template<typename T> int A<T>::var;
                if not isSpecialization():
                    templateArgs = None

        def matches(s: "Symbol") -> bool:
            if s.identOrOp != identOrOp:
                return False
            if (s.templateParams is None) != (templateParams is None):
                if templateParams is not None:
                    # we query with params, they must match params
                    return False
                if not templateShorthand:
                    # we don't query with params, and we do care about them
                    return False
            if templateParams:
                # TODO: do better comparison
                if str(s.templateParams) != str(templateParams):
                    return False
            if (s.templateArgs is None) != (templateArgs is None):
                return False
            if s.templateArgs:
                # TODO: do better comparison
                if str(s.templateArgs) != str(templateArgs):
                    return False
            return True

        def candidates() -> Generator[Symbol, None, None]:
            s = self
            if Symbol.debug_lookup:
                Symbol.debug_print("searching in self:")
                print(s.to_string(Symbol.debug_indent + 1), end="")
            while True:
                if matchSelf:
                    yield s
                if recurseInAnon:
                    yield from s.children_recurse_anon
                else:
                    yield from s._children

                if s.siblingAbove is None:
                    break
                s = s.siblingAbove
                if Symbol.debug_lookup:
                    Symbol.debug_print("searching in sibling:")
                    print(s.to_string(Symbol.debug_indent + 1), end="")

        for s in candidates():
            if Symbol.debug_lookup:
                Symbol.debug_print("candidate:")
                print(s.to_string(Symbol.debug_indent + 1), end="")
            if matches(s):
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 1
                    Symbol.debug_print("matches")
                    Symbol.debug_indent -= 3
                yield s
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 2
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2

    def _symbol_lookup(self, nestedName: ASTNestedName, templateDecls: List[Any],
                       onMissingQualifiedSymbol: Callable[["Symbol", Union[ASTIdentifier, ASTOperator], Any, ASTTemplateArgs], "Symbol"],  # NOQA
                       ancestorLookupType: str, templateShorthand: bool, matchSelf: bool,
                       recurseInAnon: bool, correctPrimaryTemplateArgs: bool,
                       searchInSiblings: bool) -> SymbolLookupResult:
        # ancestorLookupType: if not None, specifies the target type of the lookup
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_symbol_lookup:")
            Symbol.debug_indent += 1
            Symbol.debug_print("self:")
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_print("nestedName:        ", nestedName)
            Symbol.debug_print("templateDecls:     ", ",".join(str(t) for t in templateDecls))
            Symbol.debug_print("ancestorLookupType:", ancestorLookupType)
            Symbol.debug_print("templateShorthand: ", templateShorthand)
            Symbol.debug_print("matchSelf:         ", matchSelf)
            Symbol.debug_print("recurseInAnon:     ", recurseInAnon)
            Symbol.debug_print("correctPrimaryTemplateArgs: ", correctPrimaryTemplateArgs)
            Symbol.debug_print("searchInSiblings:  ", searchInSiblings)

        assert len(templateDecls) <= nestedName.num_templates() + 1

        names = nestedName.names

        # find the right starting point for lookup
        parentSymbol = self
        if nestedName.rooted:
            while parentSymbol.parent:
                parentSymbol = parentSymbol.parent
        if ancestorLookupType is not None:
            # walk up until we find the first identifier
            firstName = names[0]
            if not firstName.is_operator():
                while parentSymbol.parent:
                    if parentSymbol.find_identifier(firstName.identOrOp,
                                                    matchSelf=matchSelf,
                                                    recurseInAnon=recurseInAnon,
                                                    searchInSiblings=searchInSiblings):
                        # if we are in the scope of a constructor but wants to
                        # reference the class we need to walk one extra up
                        if (len(names) == 1 and ancestorLookupType == 'class' and matchSelf and
                                parentSymbol.parent and
                                parentSymbol.parent.identOrOp == firstName.identOrOp):
                            pass
                        else:
                            break
                    parentSymbol = parentSymbol.parent

        if Symbol.debug_lookup:
            Symbol.debug_print("starting point:")
            print(parentSymbol.to_string(Symbol.debug_indent + 1), end="")

        # and now the actual lookup
        iTemplateDecl = 0
        for name in names[:-1]:
            identOrOp = name.identOrOp
            templateArgs = name.templateArgs

            # take the next template parameter list if there is one
            # otherwise it's ok
            if templateArgs and iTemplateDecl < len(templateDecls):
                templateParams = templateDecls[iTemplateDecl]
                iTemplateDecl += 1
            else:
                templateParams = None

            symbol = parentSymbol._find_first_named_symbol(
                identOrOp,
                templateParams, templateArgs,
                templateShorthand=templateShorthand,
                matchSelf=matchSelf,
                recurseInAnon=recurseInAnon,
                correctPrimaryTemplateArgs=correctPrimaryTemplateArgs)
            if symbol is None:
                symbol = onMissingQualifiedSymbol(parentSymbol, identOrOp,
                                                  templateParams, templateArgs)
                if symbol is None:
                    if Symbol.debug_lookup:
                        Symbol.debug_indent -= 2
                    return None
            # We have now matched part of a nested name, and need to match more
            # so even if we should matchSelf before, we definitely shouldn't
            # even more. (see also issue #2666)
            matchSelf = False
            parentSymbol = symbol

        if Symbol.debug_lookup:
            Symbol.debug_print("handle last name from:")
            print(parentSymbol.to_string(Symbol.debug_indent + 1), end="")

        # handle the last name
        name = names[-1]
        identOrOp = name.identOrOp
        templateArgs = name.templateArgs
        if iTemplateDecl < len(templateDecls):
            assert iTemplateDecl + 1 == len(templateDecls)
            templateParams = templateDecls[iTemplateDecl]
        else:
            assert iTemplateDecl == len(templateDecls)
            templateParams = None

        symbols = parentSymbol._find_named_symbols(
            identOrOp, templateParams, templateArgs,
            templateShorthand=templateShorthand, matchSelf=matchSelf,
            recurseInAnon=recurseInAnon, correctPrimaryTemplateArgs=False,
            searchInSiblings=searchInSiblings)
        if Symbol.debug_lookup:
            symbols = list(symbols)  # type: ignore
            Symbol.debug_indent -= 2
        return SymbolLookupResult(symbols, parentSymbol,
                                  identOrOp, templateParams, templateArgs)

    def _add_symbols(self, nestedName: ASTNestedName, templateDecls: List[Any],
                     declaration: ASTDeclaration, docname: str, line: int) -> "Symbol":
        # Used for adding a whole path of symbols, where the last may or may not
        # be an actual declaration.

        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_add_symbols:")
            Symbol.debug_indent += 1
            Symbol.debug_print("tdecls:", ",".join(str(t) for t in templateDecls))
            Symbol.debug_print("nn:       ", nestedName)
            Symbol.debug_print("decl:     ", declaration)
            Symbol.debug_print("location: {}:{}".format(docname, line))

        def onMissingQualifiedSymbol(parentSymbol: "Symbol",
                                     identOrOp: Union[ASTIdentifier, ASTOperator],
                                     templateParams: Any, templateArgs: ASTTemplateArgs
                                     ) -> "Symbol":
            if Symbol.debug_lookup:
                Symbol.debug_indent += 1
                Symbol.debug_print("_add_symbols, onMissingQualifiedSymbol:")
                Symbol.debug_indent += 1
                Symbol.debug_print("templateParams:", templateParams)
                Symbol.debug_print("identOrOp:     ", identOrOp)
                Symbol.debug_print("templateARgs:  ", templateArgs)
                Symbol.debug_indent -= 2
            return Symbol(parent=parentSymbol, identOrOp=identOrOp,
                          templateParams=templateParams,
                          templateArgs=templateArgs, declaration=None,
                          docname=None, line=None)

        lookupResult = self._symbol_lookup(nestedName, templateDecls,
                                           onMissingQualifiedSymbol,
                                           ancestorLookupType=None,
                                           templateShorthand=False,
                                           matchSelf=False,
                                           recurseInAnon=False,
                                           correctPrimaryTemplateArgs=True,
                                           searchInSiblings=False)
        assert lookupResult is not None  # we create symbols all the way, so that can't happen
        symbols = list(lookupResult.symbols)
        if len(symbols) == 0:
            if Symbol.debug_lookup:
                Symbol.debug_print("_add_symbols, result, no symbol:")
                Symbol.debug_indent += 1
                Symbol.debug_print("templateParams:", lookupResult.templateParams)
                Symbol.debug_print("identOrOp:     ", lookupResult.identOrOp)
                Symbol.debug_print("templateArgs:  ", lookupResult.templateArgs)
                Symbol.debug_print("declaration:   ", declaration)
                Symbol.debug_print("location:      {}:{}".format(docname, line))
                Symbol.debug_indent -= 1
            symbol = Symbol(parent=lookupResult.parentSymbol,
                            identOrOp=lookupResult.identOrOp,
                            templateParams=lookupResult.templateParams,
                            templateArgs=lookupResult.templateArgs,
                            declaration=declaration,
                            docname=docname, line=line)
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 2
            return symbol

        if Symbol.debug_lookup:
            Symbol.debug_print("_add_symbols, result, symbols:")
            Symbol.debug_indent += 1
            Symbol.debug_print("number symbols:", len(symbols))
            Symbol.debug_indent -= 1

        if not declaration:
            if Symbol.debug_lookup:
                Symbol.debug_print("no declaration")
                Symbol.debug_indent -= 2
            # good, just a scope creation
            # TODO: what if we have more than one symbol?
            return symbols[0]

        noDecl = []
        withDecl = []
        dupDecl = []
        for s in symbols:
            if s.declaration is None:
                noDecl.append(s)
            elif s.isRedeclaration:
                dupDecl.append(s)
            else:
                withDecl.append(s)
        if Symbol.debug_lookup:
            Symbol.debug_print("#noDecl:  ", len(noDecl))
            Symbol.debug_print("#withDecl:", len(withDecl))
            Symbol.debug_print("#dupDecl: ", len(dupDecl))
        # With partial builds we may start with a large symbol tree stripped of declarations.
        # Essentially any combination of noDecl, withDecl, and dupDecls seems possible.
        # TODO: make partial builds fully work. What should happen when the primary symbol gets
        #  deleted, and other duplicates exist? The full document should probably be rebuild.

        # First check if one of those with a declaration matches.
        # If it's a function, we need to compare IDs,
        # otherwise there should be only one symbol with a declaration.
        def makeCandSymbol() -> "Symbol":
            if Symbol.debug_lookup:
                Symbol.debug_print("begin: creating candidate symbol")
            symbol = Symbol(parent=lookupResult.parentSymbol,
                            identOrOp=lookupResult.identOrOp,
                            templateParams=lookupResult.templateParams,
                            templateArgs=lookupResult.templateArgs,
                            declaration=declaration,
                            docname=docname, line=line)
            if Symbol.debug_lookup:
                Symbol.debug_print("end:   creating candidate symbol")
            return symbol
        if len(withDecl) == 0:
            candSymbol = None
        else:
            candSymbol = makeCandSymbol()

            def handleDuplicateDeclaration(symbol: "Symbol", candSymbol: "Symbol") -> None:
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 1
                    Symbol.debug_print("redeclaration")
                    Symbol.debug_indent -= 1
                    Symbol.debug_indent -= 2
                # Redeclaration of the same symbol.
                # Let the new one be there, but raise an error to the client
                # so it can use the real symbol as subscope.
                # This will probably result in a duplicate id warning.
                candSymbol.isRedeclaration = True
                raise _DuplicateSymbolError(symbol, declaration)

            # a function, so compare IDs
            candId = declaration.get_id()
            if Symbol.debug_lookup:
                Symbol.debug_print("candId:", candId)
            for symbol in withDecl:
                oldId = symbol.declaration.get_id()
                if Symbol.debug_lookup:
                    Symbol.debug_print("oldId: ", oldId)
                if candId == oldId:
                    handleDuplicateDeclaration(symbol, candSymbol)
                    # (not reachable)
            # no candidate symbol found with matching ID
        # if there is an empty symbol, fill that one
        if len(noDecl) == 0:
            if Symbol.debug_lookup:
                Symbol.debug_print("no match, no empty")
                if candSymbol is not None:
                    Symbol.debug_print("result is already created candSymbol")
                else:
                    Symbol.debug_print("result is makeCandSymbol()")
                Symbol.debug_indent -= 2
            if candSymbol is not None:
                return candSymbol
            else:
                return makeCandSymbol()
        else:
            if Symbol.debug_lookup:
                Symbol.debug_print("no match, but fill an empty declaration, candSybmol is not None?:", candSymbol is not None)  # NOQA
                Symbol.debug_indent -= 2
            if candSymbol is not None:
                candSymbol.remove()
            # assert len(noDecl) == 1
            # TODO: enable assertion when we at some point find out how to do cleanup
            # for now, just take the first one, it should work fine ... right?
            symbol = noDecl[0]
            # If someone first opened the scope, and then later
            # declares it, e.g,
            # .. namespace:: Test
            # .. namespace:: nullptr
            # .. class:: Test
            symbol._fill_empty(declaration, docname, line)
            return symbol

    def merge_with(self, other: "Symbol", docnames: List[str],
                   env: "BuildEnvironment") -> None:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("merge_with:")
        assert other is not None

        def unconditionalAdd(self, otherChild):
            # TODO: hmm, should we prune by docnames?
            self._children.append(otherChild)
            otherChild.parent = self
            otherChild._assert_invariants()

        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
        for otherChild in other._children:
            if Symbol.debug_lookup:
                Symbol.debug_print("otherChild:\n", otherChild.to_string(Symbol.debug_indent))
                Symbol.debug_indent += 1
            if otherChild.isRedeclaration:
                unconditionalAdd(self, otherChild)
                if Symbol.debug_lookup:
                    Symbol.debug_print("isRedeclaration")
                    Symbol.debug_indent -= 1
                continue
            candiateIter = self._find_named_symbols(
                identOrOp=otherChild.identOrOp,
                templateParams=otherChild.templateParams,
                templateArgs=otherChild.templateArgs,
                templateShorthand=False, matchSelf=False,
                recurseInAnon=False, correctPrimaryTemplateArgs=False,
                searchInSiblings=False)
            candidates = list(candiateIter)

            if Symbol.debug_lookup:
                Symbol.debug_print("raw candidate symbols:", len(candidates))
            symbols = [s for s in candidates if not s.isRedeclaration]
            if Symbol.debug_lookup:
                Symbol.debug_print("non-duplicate candidate symbols:", len(symbols))

            if len(symbols) == 0:
                unconditionalAdd(self, otherChild)
                if Symbol.debug_lookup:
                    Symbol.debug_indent -= 1
                continue

            ourChild = None
            if otherChild.declaration is None:
                if Symbol.debug_lookup:
                    Symbol.debug_print("no declaration in other child")
                ourChild = symbols[0]
            else:
                queryId = otherChild.declaration.get_id()
                if Symbol.debug_lookup:
                    Symbol.debug_print("queryId:  ", queryId)
                for symbol in symbols:
                    if symbol.declaration is None:
                        if Symbol.debug_lookup:
                            Symbol.debug_print("empty candidate")
                        # if in the end we have non matching, but have an empty one,
                        # then just continue with that
                        ourChild = symbol
                        continue
                    candId = symbol.declaration.get_id()
                    if Symbol.debug_lookup:
                        Symbol.debug_print("candidate:", candId)
                    if candId == queryId:
                        ourChild = symbol
                        break
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 1
            if ourChild is None:
                unconditionalAdd(self, otherChild)
                continue
            if otherChild.declaration and otherChild.docname in docnames:
                if not ourChild.declaration:
                    ourChild._fill_empty(otherChild.declaration,
                                         otherChild.docname, otherChild.line)
                elif ourChild.docname != otherChild.docname:
                    name = str(ourChild.declaration)
                    msg = __("Duplicate C# declaration, also defined at %s:%s.\n"
                             "Declaration is '.. csharp:%s:: %s'.")
                    msg = msg % (ourChild.docname, ourChild.line,
                                 ourChild.declaration.directiveType, name)
                    logger.warning(msg, location=(otherChild.docname, otherChild.line))
                else:
                    # Both have declarations, and in the same docname.
                    # This can apparently happen, it should be safe to
                    # just ignore it, right?
                    # Hmm, only on duplicate declarations, right?
                    msg = "Internal C# domain error during symbol merging.\n"
                    msg += "ourChild:\n" + ourChild.to_string(1)
                    msg += "\notherChild:\n" + otherChild.to_string(1)
                    logger.warning(msg, location=otherChild.docname)
            ourChild.merge_with(otherChild, docnames, env)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2

    def add_name(self, nestedName: ASTNestedName) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("add_name:")
        else:
            templateDecls = []
        res = self._add_symbols(nestedName, templateDecls,
                                declaration=None, docname=None, line=None)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1
        return res

    def add_declaration(self, declaration: ASTDeclaration,
                        docname: str, line: int) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("add_declaration:")
        assert declaration is not None
        assert docname is not None
        assert line is not None
        nestedName = declaration.name
        templateDecls = []
        res = self._add_symbols(nestedName, templateDecls, declaration, docname, line)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1
        return res

    def find_identifier(self, identOrOp: Union[ASTIdentifier, ASTOperator],
                        matchSelf: bool, recurseInAnon: bool, searchInSiblings: bool
                        ) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("find_identifier:")
            Symbol.debug_indent += 1
            Symbol.debug_print("identOrOp:       ", identOrOp)
            Symbol.debug_print("matchSelf:       ", matchSelf)
            Symbol.debug_print("recurseInAnon:   ", recurseInAnon)
            Symbol.debug_print("searchInSiblings:", searchInSiblings)
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_indent -= 2
        current = self
        while current is not None:
            if Symbol.debug_lookup:
                Symbol.debug_indent += 2
                Symbol.debug_print("trying:")
                print(current.to_string(Symbol.debug_indent + 1), end="")
                Symbol.debug_indent -= 2
            if matchSelf and current.identOrOp == identOrOp:
                return current
            children = current.children_recurse_anon if recurseInAnon else current._children
            for s in children:
                if s.identOrOp == identOrOp:
                    return s
            if not searchInSiblings:
                break
            current = current.siblingAbove
        return None

    def direct_lookup(self, key: "LookupKey") -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("direct_lookup:")
            Symbol.debug_indent += 1
        s = self
        for name, templateParams, id_ in key.data:
            if id_ is not None:
                res = None
                for cand in s._children:
                    if cand.declaration is None:
                        continue
                    if cand.declaration.get_id() == id_:
                        res = cand
                        break
                s = res
            else:
                identOrOp = name.identOrOp
                templateArgs = name.templateArgs
                s = s._find_first_named_symbol(identOrOp,
                                               templateParams, templateArgs,
                                               templateShorthand=False,
                                               matchSelf=False,
                                               recurseInAnon=False,
                                               correctPrimaryTemplateArgs=False)
            if Symbol.debug_lookup:
                Symbol.debug_print("name:          ", name)
                Symbol.debug_print("templateParams:", templateParams)
                Symbol.debug_print("id:            ", id_)
                if s is not None:
                    print(s.to_string(Symbol.debug_indent + 1), end="")
                else:
                    Symbol.debug_print("not found")
            if s is None:
                if Symbol.debug_lookup:
                    Symbol.debug_indent -= 2
                return None
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
        return s

    def find_name(self, nestedName: ASTNestedName, templateDecls: List[Any],
                  typ: str, templateShorthand: bool, matchSelf: bool,
                  recurseInAnon: bool, searchInSiblings: bool) -> Tuple[List["Symbol"], str]:
        # templateShorthand: missing template parameter lists for templates is ok
        # If the first component is None,
        # then the second component _may_ be a string explaining why.
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("find_name:")
            Symbol.debug_indent += 1
            Symbol.debug_print("self:")
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_print("nestedName:       ", nestedName)
            Symbol.debug_print("templateDecls:    ", templateDecls)
            Symbol.debug_print("typ:              ", typ)
            Symbol.debug_print("templateShorthand:", templateShorthand)
            Symbol.debug_print("matchSelf:        ", matchSelf)
            Symbol.debug_print("recurseInAnon:    ", recurseInAnon)
            Symbol.debug_print("searchInSiblings: ", searchInSiblings)

        class QualifiedSymbolIsTemplateParam(Exception):
            pass

        def onMissingQualifiedSymbol(parentSymbol: "Symbol",
                                     identOrOp: Union[ASTIdentifier, ASTOperator],
                                     templateParams: Any,
                                     templateArgs: ASTTemplateArgs) -> "Symbol":
            # TODO: Maybe search without template args?
            #       Though, the correctPrimaryTemplateArgs does
            #       that for primary templates.
            #       Is there another case where it would be good?
            if parentSymbol.declaration is not None:
                if parentSymbol.declaration.objectType == 'templateParam':
                    raise QualifiedSymbolIsTemplateParam()
            return None

        try:
            lookupResult = self._symbol_lookup(nestedName, templateDecls,
                                               onMissingQualifiedSymbol,
                                               ancestorLookupType=typ,
                                               templateShorthand=templateShorthand,
                                               matchSelf=matchSelf,
                                               recurseInAnon=recurseInAnon,
                                               correctPrimaryTemplateArgs=False,
                                               searchInSiblings=searchInSiblings)
        except QualifiedSymbolIsTemplateParam:
            return None, "templateParamInQualified"

        if lookupResult is None:
            # if it was a part of the qualification that could not be found
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 2
            return None, None

        res = list(lookupResult.symbols)
        if len(res) != 0:
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 2
            return res, None

        if lookupResult.parentSymbol.declaration is not None:
            if lookupResult.parentSymbol.declaration.objectType == 'templateParam':
                return None, "templateParamInQualified"

        # try without template params and args
        symbol = lookupResult.parentSymbol._find_first_named_symbol(
            lookupResult.identOrOp, None, None,
            templateShorthand=templateShorthand, matchSelf=matchSelf,
            recurseInAnon=recurseInAnon, correctPrimaryTemplateArgs=False)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
        if symbol is not None:
            return [symbol], None
        else:
            return None, None

    def find_declaration(self, declaration: ASTDeclaration, typ: str, templateShorthand: bool,
                         matchSelf: bool, recurseInAnon: bool) -> "Symbol":
        # templateShorthand: missing template parameter lists for templates is ok
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("find_declaration:")
        nestedName = declaration.name
        templateDecls = []

        def onMissingQualifiedSymbol(parentSymbol: "Symbol",
                                     identOrOp: Union[ASTIdentifier, ASTOperator],
                                     templateParams: Any,
                                     templateArgs: ASTTemplateArgs) -> "Symbol":
            return None

        lookupResult = self._symbol_lookup(nestedName, templateDecls,
                                           onMissingQualifiedSymbol,
                                           ancestorLookupType=typ,
                                           templateShorthand=templateShorthand,
                                           matchSelf=matchSelf,
                                           recurseInAnon=recurseInAnon,
                                           correctPrimaryTemplateArgs=False,
                                           searchInSiblings=False)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1
        if lookupResult is None:
            return None

        symbols = list(lookupResult.symbols)
        if len(symbols) == 0:
            return None

        querySymbol = Symbol(parent=lookupResult.parentSymbol,
                             identOrOp=lookupResult.identOrOp,
                             templateParams=lookupResult.templateParams,
                             templateArgs=lookupResult.templateArgs,
                             declaration=declaration,
                             docname='fakeDocnameForQuery',
                             line=42)
        queryId = declaration.get_id()
        for symbol in symbols:
            if symbol.declaration is None:
                continue
            candId = symbol.declaration.get_id()
            if candId == queryId:
                querySymbol.remove()
                return symbol
        querySymbol.remove()
        return None

    def to_string(self, indent: int) -> str:
        res = [Symbol.debug_indent_string * indent]
        if not self.parent:
            res.append('.')
        else:
            if self.templateParams:
                res.append(str(self.templateParams))
                res.append('\n')
                res.append(Symbol.debug_indent_string * indent)
            if self.identOrOp:
                res.append(str(self.identOrOp))
            else:
                res.append(str(self.declaration))
            if self.templateArgs:
                res.append(str(self.templateArgs))
            if self.declaration:
                res.append(": ")
                if self.isRedeclaration:
                    res.append('!!duplicate!! ')
                res.append(str(self.declaration))
        if self.docname:
            res.append('\t(')
            res.append(self.docname)
            res.append(')')
        res.append('\n')
        return ''.join(res)

    def dump(self, indent: int) -> str:
        res = [self.to_string(indent)]
        for c in self._children:
            res.append(c.dump(indent + 1))
        return ''.join(res)


class DefinitionParser(BaseParser):
    # those without signedness and size modifiers
    # see https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/builtin-types/built-in-types
    _simple_fundamental_types = (
        'void', 'bool', 'char', 'byte', 'sbyte', 'short', 'ushort', 'int', 'uint', 'nint', 'nuint', 'long', 'ulong', 'float', 'double', 'decimal', 'var'
    )

    _prefix_keys = ('interface', 'class', 'struct', 'enum')

    @property
    def language(self) -> str:
        return 'C#'

    @property
    def id_attributes(self):
        return self.config.csharp_id_attributes

    @property
    def paren_attributes(self):
        return self.config.csharp_paren_attributes

    def _parse_string(self) -> str:
        if self.current_char != '"':
            return None
        startPos = self.pos
        self.pos += 1
        escape = False
        while True:
            if self.eof:
                self.fail("Unexpected end during inside string.")
            elif self.current_char == '"' and not escape:
                self.pos += 1
                break
            elif self.current_char == '\\':
                escape = True
            else:
                escape = False
            self.pos += 1
        return self.definition[startPos:self.pos]

    def _parse_literal(self) -> ASTLiteral:
        # -> integer-literal
        #  | character-literal
        #  | floating-literal
        #  | string-literal
        #  | simple-literal -> "null" | "false" | "true" | "default"
        #  | user-defined-literal

        def _udl(literal: ASTLiteral) -> ASTLiteral:
            if not self.match(udl_identifier_re):
                return literal
            # hmm, should we care if it's a keyword?
            # it looks like GCC does not disallow keywords
            ident = ASTIdentifier(self.matched_text)
            return ASTUserDefinedLiteral(literal, ident)

        self.skip_ws()
        for literal, id in _literal_keywords.items():
            if self.skip_word(literal):
                return ASTSimpleLiteral(literal, id)
        pos = self.pos
        if self.match(float_literal_re):
            hasSuffix = self.match(float_literal_suffix_re)
            floatLit = ASTNumberLiteral(self.definition[pos:self.pos])
            if hasSuffix:
                return floatLit
            else:
                return _udl(floatLit)
        for regex in [binary_literal_re, hex_literal_re,
                      integer_literal_re, octal_literal_re]:
            if self.match(regex):
                hasSuffix = self.match(integers_literal_suffix_re)
                intLit = ASTNumberLiteral(self.definition[pos:self.pos])
                if hasSuffix:
                    return intLit
                else:
                    return _udl(intLit)

        string = self._parse_string()
        if string is not None:
            return _udl(ASTStringLiteral(string))

        # character-literal
        if self.match(char_literal_re):
            prefix = self.last_match.group(1)  # may be None when no prefix
            data = self.last_match.group(2)
            try:
                charLit = ASTCharLiteral(prefix, data)
            except UnicodeDecodeError as e:
                self.fail("Can not handle character literal. Internal error was: %s" % e)
            except UnsupportedMultiCharacterCharLiteral:
                self.fail("Can not handle character literal"
                          " resulting in multiple decoded characters.")
            return _udl(charLit)
        return None

    def _parse_attribute(self) -> Optional[ASTAttribute]:
        self.skip_ws()
        if self.skip_string_and_ws('['):
            arg = self._parse_balanced_token_seq(end=[']'])
            if not self.skip_string_and_ws(']'):
                self.fail("Expected ']' in end of attribute.")
            return ASTCSharpAttribute(arg)

        # try the simple id attributes defined by the user
        for id in self.id_attributes:
            if self.skip_word_and_ws(id):
                return ASTIdAttribute(id)

        # try the paren attributes defined by the user
        for id in self.paren_attributes:
            if not self.skip_string_and_ws(id):
                continue
            if not self.skip_string('('):
                self.fail("Expected '(' after user-defined paren-attribute.")
            arg = self._parse_balanced_token_seq(end=[')'])
            if not self.skip_string(')'):
                self.fail("Expected ')' to end user-defined paren-attribute.")
            return ASTParenAttribute(id, arg)

        if self.skip_string_and_ws('?'):
            return ASTNullableAttribute()

        return None

    def _parse_paren_expression(self) -> ASTExpression:
        # "(" expression ")"
        if self.current_char != '(':
            return None
        self.pos += 1
        self.skip_ws()
        try:
            res = self._parse_expression()
            self.skip_ws()
            if not self.skip_string(')'):
                self.fail("Expected ')' in end of parenthesized expression.")
        except DefinitionError as eExpr:
            raise Exception('Error in parenthesized expression.')
        return ASTParenExpr(res)

    def _parse_primary_expression(self) -> ASTExpression:
        # literal
        # lambda-expression
        # "(" expression ")"
        # fold-expression
        # id-expression -> we parse this with _parse_nested_name
        self.skip_ws()
        res = self._parse_literal()  # type: ASTExpression
        if res is not None:
            return res
        self.skip_ws()
        # TODO: try lambda expression
        res = self._parse_paren_expression()
        if res is not None:
            return res
        nn = self._parse_nested_name()
        if nn is not None:
            return ASTIdExpression(nn)
        return None

    def _parse_initializer_list(self, name: str, open: str, close: str
                                ) -> Tuple[List[Union[ASTExpression,
                                                      ASTBracedInitList]],
                                           bool]:
        self.skip_ws()
        if not self.skip_string_and_ws(open):
            return None, None
        if self.skip_string(close):
            return [], False

        exprs = []  # type: List[Union[ASTExpression, ASTBracedInitList]]
        trailingComma = False
        while True:
            self.skip_ws()
            expr = self._parse_initializer_clause()
            self.skip_ws()
            exprs.append(expr)
            self.skip_ws()
            if self.skip_string(close):
                break
            if not self.skip_string_and_ws(','):
                self.fail("Error in %s, expected ',' or '%s'." % (name, close))
            if self.current_char == close and close == '}':
                self.pos += 1
                trailingComma = True
                break
        return exprs, trailingComma

    def _parse_paren_expression_list(self) -> ASTParenExprList:
        # -> '(' expression-list ')'
        # though, we relax it to also allow empty parens
        # as it's needed in some cases
        #
        # expression-list
        # -> initializer-list
        exprs, trailingComma = self._parse_initializer_list("parenthesized expression-list",
                                                            '(', ')')
        if exprs is None:
            return None
        return ASTParenExprList(exprs)

    def _parse_initializer_clause(self) -> Union[ASTExpression, ASTBracedInitList]:
        bracedInitList = self._parse_braced_init_list()
        if bracedInitList is not None:
            return bracedInitList
        return self._parse_assignment_expression(inTemplate=False)

    def _parse_braced_init_list(self) -> ASTBracedInitList:
        # -> '{' initializer-list ','[opt] '}'
        #  | '{' '}'
        exprs, trailingComma = self._parse_initializer_list("braced-init-list", '{', '}')
        if exprs is None:
            return None
        return ASTBracedInitList(exprs, trailingComma)

    def _parse_expression_list_or_braced_init_list(
        self
    ) -> Union[ASTParenExprList, ASTBracedInitList]:
        paren = self._parse_paren_expression_list()
        if paren is not None:
            return paren
        return self._parse_braced_init_list()

    def _parse_postfix_expression(self) -> ASTPostfixExpr:
        # -> primary
        #  | postfix "[" expression "]"
        #  | postfix "[" braced-init-list [opt] "]"
        #  | postfix "(" expression-list [opt] ")"
        #  | postfix "." "template" [opt] id-expression
        #  | postfix "->" "template" [opt] id-expression
        #  | postfix "." pseudo-destructor-name
        #  | postfix "->" pseudo-destructor-name
        #  | postfix "++"
        #  | postfix "--"
        #  | simple-type-specifier "(" expression-list [opt] ")"
        #  | simple-type-specifier braced-init-list
        #  | typename-specifier "(" expression-list [opt] ")"
        #  | typename-specifier braced-init-list

        prefix = None  # type: Any
        self.skip_ws()

        pos = self.pos
        try:
            prefix = self._parse_primary_expression()
        except DefinitionError as eOuter:
            self.pos = pos
            try:
                # we are potentially casting, so save parens for us
                # TODO: hmm, would we need to try both with operatorCast and with None?
                prefix = self._parse_type(False, 'operatorCast')
                #  | simple-type-specifier "(" expression-list [opt] ")"
                #  | simple-type-specifier braced-init-list
                #  | typename-specifier "(" expression-list [opt] ")"
                #  | typename-specifier braced-init-list
                self.skip_ws()
                if self.current_char != '(' and self.current_char != '{':
                    self.fail("Expecting '(' or '{' after type in cast expression.")
            except DefinitionError as eInner:
                self.pos = pos
                header = "Error in postfix expression,"
                header += " expected primary expression or type."
                errors = []
                errors.append((eOuter, "If primary expression"))
                errors.append((eInner, "If type"))
                raise self._make_multi_error(errors, header) from eInner

        # and now parse postfixes
        postFixes = []  # type: List[ASTPostfixOp]
        while True:
            self.skip_ws()
            lst = self._parse_expression_list_or_braced_init_list()
            if lst is not None:
                postFixes.append(ASTPostfixCallExpr(lst))
                continue
            break
        return ASTPostfixExpr(prefix, postFixes)

    def _parse_unary_expression(self) -> ASTExpression:
        # -> postfix
        #  | "++" cast
        #  | "--" cast
        #  | unary-operator cast -> (* | & | + | - | ! | ~) cast
        # The rest:
        #  | "sizeof" unary
        #  | "sizeof" "(" type-id ")"
        #  | new-expression
        self.skip_ws()
        for op in _expression_unary_ops:
            res = self.skip_string(op)
            if res:
                expr = self._parse_cast_expression()
                return ASTUnaryOpExpr(op, expr)
        if self.skip_word_and_ws('sizeof'):
            if self.skip_string_and_ws('('):
                typ = self._parse_type(named=False)
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail("Expecting ')' to end 'sizeof'.")
                return ASTSizeofType(typ)
            expr = self._parse_unary_expression()
            return ASTSizeofExpr(expr)
        # new-expression
        pos = self.pos
        rooted = self.skip_string('.')
        self.skip_ws()
        if not self.skip_word_and_ws('new'):
            self.pos = pos
        else:
            # new-placement[opt] new-type-id new-initializer[opt]
            # new-placement[opt] ( type-id ) new-initializer[opt]
            isNewTypeId = True
            if self.skip_string_and_ws('('):
                # either this is a new-placement or it's the second production
                # without placement, and it's actually the ( type-id ) part
                self.fail("Sorry, neither new-placement nor parenthesised type-id "
                          "in new-epression is supported yet.")
                # set isNewTypeId = False if it's (type-id)
            if isNewTypeId:
                declSpecs = self._parse_decl_specs(outer=None)
                decl = self._parse_declarator(named=False, paramMode="new")
            else:
                self.fail("Sorry, parenthesised type-id in new expression not yet supported.")
            lst = self._parse_expression_list_or_braced_init_list()
            return ASTNewExpr(rooted, isNewTypeId, ASTType(declSpecs, decl), lst)
        # delete-expression
        pos = self.pos
        rooted = self.skip_string('.')
        self.skip_ws()
        self.pos = pos
        return self._parse_postfix_expression()

    def _parse_cast_expression(self) -> ASTExpression:
        pos = self.pos
        self.skip_ws()
        if self.skip_string('('):
            try:
                typ = self._parse_type(False)
                if not self.skip_string(')'):
                    self.fail("Expected ')' in cast expression.")
                expr = self._parse_cast_expression()
                return ASTCastExpr(typ, expr)
            except DefinitionError as exCast:
                self.pos = pos
                try:
                    return self._parse_unary_expression()
                except DefinitionError as exUnary:
                    errs = []
                    errs.append((exCast, "If type cast expression"))
                    errs.append((exUnary, "If unary expression"))
                    raise self._make_multi_error(errs,
                                                 "Error in cast expression.") from exUnary
        else:
            return self._parse_unary_expression()

    def _parse_logical_or_expression(self, inTemplate: bool) -> ASTExpression:
        def _parse_bin_op_expr(self: DefinitionParser,
                               opId: int, inTemplate: bool) -> ASTExpression:
            if opId + 1 == len(_expression_bin_ops):
                def parser(inTemplate: bool) -> ASTExpression:
                    return self._parse_cast_expression()
            else:
                def parser(inTemplate: bool) -> ASTExpression:
                    return _parse_bin_op_expr(self, opId + 1, inTemplate=inTemplate)
            exprs = []
            ops = []
            exprs.append(parser(inTemplate=inTemplate))
            while True:
                self.skip_ws()
                if inTemplate and self.current_char == '>':
                    break
                pos = self.pos
                oneMore = False
                for op in _expression_bin_ops[opId]:
                    if not self.skip_string(op):
                        continue
                    if op == '&' and self.current_char == '&' or \
                       op == '|' and self.current_char == '|' or \
                       op == '<' and self.current_char == '<' or \
                       op == '>' and self.current_char == '>':
                        # don't split the &&, ||, << or >> tokens
                        self.pos -= 1
                        break
                    try:
                        expr = parser(inTemplate=inTemplate)
                        exprs.append(expr)
                        ops.append(op)
                        oneMore = True
                        break
                    except DefinitionError:
                        self.pos = pos
                if not oneMore:
                    break
            return ASTBinOpExpr(exprs, ops)
        return _parse_bin_op_expr(self, 0, inTemplate=inTemplate)

    def _parse_conditional_expression_tail(self, orExprHead: Any) -> None:
        # -> "?" expression ":" assignment-expression
        return None

    def _parse_assignment_expression(self, inTemplate: bool) -> ASTExpression:
        # -> conditional-expression
        #  | logical-or-expression assignment-operator initializer-clause
        #  | throw-expression
        # TODO: parse throw-expression: "throw" assignment-expression [opt]
        # if not a throw expression, then:
        # -> conditional-expression ->
        #     logical-or-expression
        #   | logical-or-expression "?" expression ":" assignment-expression
        #   | logical-or-expression assignment-operator initializer-clause
        exprs = []  # type: List[Union[ASTExpression, ASTBracedInitList]]
        ops = []
        orExpr = self._parse_logical_or_expression(inTemplate=inTemplate)
        exprs.append(orExpr)
        # TODO: handle ternary with _parse_conditional_expression_tail
        while True:
            oneMore = False
            self.skip_ws()
            for op in _expression_assignment_ops:
                if not self.skip_string(op):
                    continue
                expr = self._parse_initializer_clause()
                exprs.append(expr)
                ops.append(op)
                oneMore = True
            if not oneMore:
                break
        if len(ops) == 0:
            return orExpr
        else:
            return ASTAssignmentExpr(exprs, ops)

    def _parse_constant_expression(self, inTemplate: bool) -> ASTExpression:
        # -> conditional-expression
        orExpr = self._parse_logical_or_expression(inTemplate=inTemplate)
        # TODO: use _parse_conditional_expression_tail
        return orExpr

    def _parse_expression(self) -> ASTExpression:
        # -> assignment-expression
        #  | expression "," assignment-expresion
        exprs = [self._parse_assignment_expression(inTemplate=False)]
        while True:
            self.skip_ws()
            if not self.skip_string(','):
                break
            exprs.append(self._parse_assignment_expression(inTemplate=False))
        if len(exprs) == 1:
            return exprs[0]
        else:
            return ASTCommaExpr(exprs)

    def _parse_expression_fallback(self, end: List[str],
                                   parser: Callable[[], ASTExpression],
                                   allow: bool = True) -> ASTExpression:
        # Stupidly "parse" an expression.
        # 'end' should be a list of characters which ends the expression.

        # first try to use the provided parser
        prevPos = self.pos
        try:
            return parser()
        except DefinitionError as e:
            # some places (e.g., template parameters) we really don't want to use fallback,
            # and for testing we may want to globally disable it
            if not allow or not self.allowFallbackExpressionParsing:
                raise
            self.warn("Parsing of expression failed. Using fallback parser."
                      " Error was:\n%s" % e)
            self.pos = prevPos
        # and then the fallback scanning
        assert end is not None
        self.skip_ws()
        startPos = self.pos
        if self.match(_string_re):
            value = self.matched_text
        else:
            # TODO: add handling of more bracket-like things, and quote handling
            brackets = {'(': ')', '{': '}', '[': ']', '<': '>'}
            symbols = []  # type: List[str]
            while not self.eof:
                if (len(symbols) == 0 and self.current_char in end):
                    break
                if self.current_char in brackets.keys():
                    symbols.append(brackets[self.current_char])
                elif len(symbols) > 0 and self.current_char == symbols[-1]:
                    symbols.pop()
                self.pos += 1
            if len(end) > 0 and self.eof:
                self.fail("Could not find end of expression starting at %d."
                          % startPos)
            value = self.definition[startPos:self.pos].strip()
        return ASTFallbackExpr(value.strip())

    # ==========================================================================

    def _parse_property_accessor(self) -> str:
        if not self.match(_property_accessor_re):
            self.fail("Expected property accessor.")
        return self.matched_text

# ==========================================================================

    def _parse_operator(self) -> ASTOperator:
        self.skip_ws()

        if self.skip_word('operator'):
            self.skip_ws()
            for op in _id_builtin_operator.keys():
                if self.skip_string(op):
                    return ASTOperatorBuiltIn(op)

            # user-defined literal?
            if not self.match(identifier_re):
                self.fail("Expected user-defined literal suffix.")
            identifier = ASTIdentifier(self.matched_text)
            return ASTOperatorLiteral(identifier)
        else:
            for operator in _operators:
                if self.skip_string(operator):
                    return ASTOperator(operator)

        self.fail("Expected operator definition.")

    def _parse_template_argument_list(self) -> ASTTemplateArgs:
        # template-argument-list: (but we include the < and > here
        #    template-argument ...[opt]
        #    template-argument-list, template-argument ...[opt]
        # template-argument:
        #    constant-expression
        #    type-id
        #    id-expression
        self.skip_ws()
        if not self.skip_string_and_ws('<'):
            return None
        if self.skip_string('>'):
            return ASTTemplateArgs([], False)
        prevErrors = []
        templateArgs = []  # type: List[Union[ASTType, ASTTemplateArgConstant]]
        while 1:
            pos = self.pos
            parsedComma = False
            parsedEnd = False
            try:
                type = self._parse_type(named=False)
                self.skip_ws()
                if self.skip_string('>'):
                    parsedEnd = True
                elif self.skip_string(','):
                    parsedComma = True
                else:
                    self.fail('Expected ">" or "," in template argument list.')
                templateArgs.append(type)
            except DefinitionError as e:
                prevErrors.append((e, "If type argument"))
                self.pos = pos
                try:
                    value = self._parse_constant_expression(inTemplate=True)
                    self.skip_ws()
                    if self.skip_string('>'):
                        parsedEnd = True
                    elif self.skip_string(','):
                        parsedComma = True
                    else:
                        self.fail('Expected ">" or "," in template argument list.')
                    templateArgs.append(ASTTemplateArgConstant(value))
                except DefinitionError as e:
                    self.pos = pos
                    prevErrors.append((e, "If non-type argument"))
                    header = "Error in parsing template argument list."
                    raise self._make_multi_error(prevErrors, header) from e
            if parsedEnd:
                assert not parsedComma
                break
        return ASTTemplateArgs(templateArgs)

    def _parse_nested_name(self, memberPointer: bool = False) -> ASTNestedName:
        names = []  # type: List[ASTNestedNameElement]

        self.skip_ws()
        rooted = False
        if self.skip_string('.'):
            rooted = True
        while 1:
            self.skip_ws()
            identOrOp = None  # type: Union[ASTIdentifier, ASTOperator]
            pos = self.pos
            for operator in _operators:
                if self.skip_word(operator):
                    self.pos = pos
                    identOrOp = self._parse_operator()
                    continue

            if not identOrOp:
                if not self.match(identifier_re):
                    if memberPointer and len(names) > 0:
                        break
                    self.fail("Expected identifier in nested name.")
                identifier = self.matched_text
                if self.match(_array_indexer_re):
                    identifier += self.matched_text
                identOrOp = ASTIdentifier(identifier)
            # try greedily to get template arguments,
            # but otherwise a < might be because we are in an expression
            pos = self.pos
            try:
                templateArgs = self._parse_template_argument_list()
            except DefinitionError as ex:
                self.pos = pos
                templateArgs = None
                self.otherErrors.append(ex)
            names.append(ASTNestedNameElement(identOrOp, templateArgs))

            self.skip_ws()
            if not self.skip_string('.'):
                if memberPointer:
                    self.fail("Expected '.' in pointer to member (function).")
                break
        return ASTNestedName(names, rooted)

    # ==========================================================================

    def _parse_trailing_type_spec(self) -> ASTTrailingTypeSpec:
        # fundamental types
        self.skip_ws()
        for t in self._simple_fundamental_types:
            if self.skip_word(t):
                return ASTTrailingTypeSpecFundamental(t)

        # prefixed
        prefix = None
        self.skip_ws()
        for k in self._prefix_keys:
            if self.skip_word_and_ws(k):
                prefix = k
                break
        nestedName = self._parse_nested_name()
        return ASTTrailingTypeSpecName(prefix, nestedName)

    def _parse_parameters_and_qualifiers(self, paramMode: str) -> ASTParametersQualifiers:
        if paramMode == 'new':
            return None
        self.skip_ws()
        if paramMode in ('method', 'delegate'):
            if not self.skip_string('('):
                self.fail('Expecting "(" in parameters-and-qualifiers.')
            else:
                args = []
                self.skip_ws()
                if not self.skip_string(')'):
                    while 1:
                        self.skip_ws()
                        # note: it seems that function arguments can always be named,
                        # even in function pointers and similar.
                        arg = self._parse_type_with_init(outer=None, named='single')
                        # TODO: parse default parameters # TODO: didn't we just do that?
                        args.append(ASTFunctionParameter(arg))

                        self.skip_ws()
                        if self.skip_string(','):
                            continue
                        elif self.skip_string(')'):
                            break
                        else:
                            self.fail(
                                'Expecting "," or ")" in parameters-and-qualifiers, '
                                'got "%s".' % self.current_char)
        else:
            return None

        return ASTParametersQualifiers(paramMode, args)

    def _parse_decl_specs_simple(self, outer: str, typed: bool) -> ASTDeclSpecsSimple:
        """Just parse the simple ones."""
        modifiers = []
        while 1:  # accept any permutation of a subset of some decl-specs
            self.skip_ws()
            found_modifier = False
            for modifier in _modifiers:
                if self.skip_word(modifier):
                    modifiers.append(ASTModifier(modifier))
                    found_modifier = True
                    break
            if found_modifier:
                continue
            attr = self._parse_attribute()
            if attr:
                modifiers.append(attr)
                continue
            break
        return ASTDeclSpecsSimple(modifiers)

    def _parse_decl_specs(self, outer: str, typed: bool = True) -> ASTDeclSpecs:
        if outer:
            if outer not in ('type', 'member', 'method', 'delegate', 'property', 'templateParam'):
                raise Exception('Internal error, unknown outer "%s".' % outer)

        leftSpecs = self._parse_decl_specs_simple(outer, typed)
        rightSpecs = None

        if typed:
            trailing = self._parse_trailing_type_spec()
            rightSpecs = self._parse_decl_specs_simple(outer, typed)
        else:
            trailing = None
        return ASTDeclSpecs(outer, leftSpecs, rightSpecs, trailing)

    def _parse_declarator_name_suffix(
        self, named: Union[bool, str], paramMode: str, typed: bool
    ) -> Union[ASTDeclaratorNameParamQual, ASTDeclaratorNameBitField]:
        # now we should parse the name, and then suffixes
        if named == 'maybe':
            pos = self.pos
            try:
                declId = self._parse_nested_name()
            except DefinitionError:
                self.pos = pos
                declId = None
        elif named == 'single':
            if self.match(identifier_re):
                matched_text = self.matched_text
                if self.match(_array_indexer_re):
                    matched_text += self.matched_text
                identifier = ASTIdentifier(matched_text)
                nne = ASTNestedNameElement(identifier, None)
                declId = ASTNestedName([nne], rooted=False)
                # if it's a member pointer, we may have '::', which should be an error
                self.skip_ws()
                if self.current_char == ':':
                    self.fail("Unexpected ':' after identifier.")
            else:
                declId = None
        elif named:
            declId = self._parse_nested_name()
        else:
            declId = None
        arrayOps = []
        while 1:
            self.skip_ws()
            if typed and self.skip_string('['):
                self.skip_ws()
                if self.skip_string(']'):
                    arrayOps.append(ASTArray(None))
                    continue

                def parser() -> ASTExpression:
                    return self._parse_expression()
                value = self._parse_expression_fallback([']'], parser)
                if not self.skip_string(']'):
                    self.fail("Expected ']' in end of array operator.")
                arrayOps.append(ASTArray(value))
                continue
            else:
                break
        paramQual = self._parse_parameters_and_qualifiers(paramMode)
        if paramQual is None and len(arrayOps) == 0:
            # perhaps a bit-field
            if named and paramMode == 'type' and typed:
                self.skip_ws()
                if self.skip_string(':'):
                    size = self._parse_constant_expression(inTemplate=False)
                    return ASTDeclaratorNameBitField(declId=declId, size=size)
        return ASTDeclaratorNameParamQual(declId=declId, arrayOps=arrayOps,
                                          paramQual=paramQual)

    def _parse_declarator(self, named: Union[bool, str], paramMode: str,
                          typed: bool = True
                          ) -> ASTDeclarator:
        # 'typed' here means 'parse return type stuff'
        if paramMode not in ('type', 'method', 'delegate', 'property', 'operatorCast', 'new'):
            raise Exception(
                "Internal error, unknown paramMode '%s'." % paramMode)
        prevErrors = []
        self.skip_ws()
        if typed and self.current_char == '(':  # note: peeking, not skipping
            if paramMode == "operatorCast":
                # TODO: we should be able to parse cast operators which return
                # function pointers. For now, just hax it and ignore.
                return ASTDeclaratorNameParamQual(declId=None, arrayOps=[],
                                                  paramQual=None)
            # maybe this is the beginning of params and quals,try that first,
            # otherwise assume it's noptr->declarator > ( ptr-declarator )
            pos = self.pos
            try:
                # assume this is params and quals
                res = self._parse_declarator_name_suffix(named, paramMode,
                                                         typed)
                return res
            except DefinitionError as exParamQual:
                prevErrors.append((exParamQual,
                                   "If declarator-id with parameters-and-qualifiers"))
                self.pos = pos
                try:
                    assert self.current_char == '('
                    self.skip_string('(')
                    # TODO: hmm, if there is a name, it must be in inner, right?
                    # TODO: hmm, if there must be parameters, they must be
                    #       inside, right?
                    inner = self._parse_declarator(named, paramMode, typed)
                    if not self.skip_string(')'):
                        self.fail("Expected ')' in \"( ptr-declarator )\"")
                    next = self._parse_declarator(named=False,
                                                  paramMode="type",
                                                  typed=typed)
                    paren = ASTDeclaratorParen(inner=inner, next=next)
                    return paren
                except DefinitionError as exNoPtrParen:
                    self.pos = pos
                    prevErrors.append((exNoPtrParen, "If parenthesis in noptr-declarator"))
                    header = "Error in declarator"
                    raise self._make_multi_error(prevErrors, header) from exNoPtrParen
        pos = self.pos
        try:
            res = self._parse_declarator_name_suffix(named, paramMode, typed)
            # this is a heuristic for error messages, for when there is a < after a
            # nested name, but it was not a successful template argument list
            if self.current_char == '<':
                self.otherErrors.append(self._make_multi_error(prevErrors, ""))
            return res
        except DefinitionError as e:
            self.pos = pos
            prevErrors.append((e, "If declarator-id"))
            header = "Error in declarator or parameters-and-qualifiers"
            raise self._make_multi_error(prevErrors, header) from e

    def _parse_initializer(self, outer: str = None, allowFallback: bool = True
                           ) -> ASTInitializer:
        self.skip_ws()
        if outer == 'member':
            bracedInit = self._parse_braced_init_list()
            if bracedInit is not None:
                return ASTInitializer(bracedInit, hasAssign=False)

        if not self.skip_string('='):
            return None

        bracedInit = self._parse_braced_init_list()
        if bracedInit is not None:
            return ASTInitializer(bracedInit)

        if outer == 'member':
            fallbackEnd = []  # type: List[str]
        elif outer == 'templateParam':
            fallbackEnd = [',', '>']
        elif outer is None:  # function parameter
            fallbackEnd = [',', ')']
        else:
            self.fail("Internal error, initializer for outer '%s' not "
                      "implemented." % outer)

        inTemplate = outer == 'templateParam'

        def parser() -> ASTExpression:
            return self._parse_assignment_expression(inTemplate=inTemplate)
        value = self._parse_expression_fallback(fallbackEnd, parser, allow=allowFallback)
        return ASTInitializer(value)

    def _parse_type(self, named: Union[bool, str], outer: str = None) -> ASTType:
        """
        named=False|'maybe'|True: 'maybe' is e.g., for function objects which
        doesn't need to name the arguments

        outer == operatorCast: annoying case, we should not take the params
        """
        if outer:  # always named
            if outer not in ('type', 'member', 'method', 'delegate',
                             'operatorCast', 'templateParam'):
                raise Exception('Internal error, unknown outer "%s".' % outer)
            if outer != 'operatorCast':
                assert named
        if outer in ('type', 'method', 'delegate'):
            # We allow type objects to just be a name.
            # Some functions don't have normal return types: constructors,
            # destrutors, cast operators
            prevErrors = []
            startPos = self.pos
            # first try without the type
            try:
                declSpecs = self._parse_decl_specs(outer=outer, typed=False)
                decl = self._parse_declarator(named=True, paramMode=outer,
                                              typed=False)
                self.assert_end(allowSemicolon=True)
            except DefinitionError as exUntyped:
                if outer == 'type':
                    desc = "If just a name"
                elif outer in ('method', 'delegate'):
                    desc = "If the function has no return type"
                else:
                    assert False
                prevErrors.append((exUntyped, desc))
                self.pos = startPos
                try:
                    declSpecs = self._parse_decl_specs(outer=outer)
                    decl = self._parse_declarator(named=True, paramMode=outer)
                except DefinitionError as exTyped:
                    self.pos = startPos
                    if outer == 'type':
                        desc = "If typedef-like declaration"
                    elif outer == 'method':
                        desc = "If the function has a return type"
                    else:
                        assert False
                    prevErrors.append((exTyped, desc))
                    # Retain the else branch for easier debugging.
                    # TODO: it would be nice to save the previous stacktrace
                    #       and output it here.
                    if True:
                        if outer == 'type':
                            header = "Type must be either just a name or a "
                            header += "typedef-like declaration."
                        elif outer == 'method':
                            header = "Error when parsing function declaration."
                        else:
                            assert False
                        raise self._make_multi_error(prevErrors, header) from exTyped
                    else:
                        # For testing purposes.
                        # do it again to get the proper traceback (how do you
                        # reliably save a traceback when an exception is
                        # constructed?)
                        self.pos = startPos
                        typed = True
                        declSpecs = self._parse_decl_specs(outer=outer, typed=typed)
                        decl = self._parse_declarator(named=True, paramMode=outer,
                                                      typed=typed)
        else:
            paramMode = 'type'
            if outer == 'member':  # i.e., member
                named = True
            elif outer == 'operatorCast':
                paramMode = 'operatorCast'
                outer = None
            elif outer == 'templateParam':
                named = 'single'
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=named, paramMode=paramMode)
        self.skip_ws()
        return ASTType(declSpecs, decl)

    def _parse_type_with_init(
            self, named: Union[bool, str],
            outer: str) -> Union[ASTTypeWithInit, ASTTemplateParamConstrainedTypeWithInit]:
        if outer:
            assert outer in ('type', 'member', 'method', 'templateParam')
        type = self._parse_type(outer=outer, named=named)
        if outer != 'templateParam':
            init = self._parse_initializer(outer=outer)
            return ASTTypeWithInit(type, init)
        # it could also be a constrained type parameter, e.g., C T = int&
        pos = self.pos
        eExpr = None
        try:
            init = self._parse_initializer(outer=outer, allowFallback=False)
            # note: init may be None if there is no =
            if init is None:
                return ASTTypeWithInit(type, None)
            # we parsed an expression, so we must have a , or a >,
            # otherwise the expression didn't get everything
            self.skip_ws()
            if self.current_char != ',' and self.current_char != '>':
                # pretend it didn't happen
                self.pos = pos
                init = None
            else:
                # we assume that it was indeed an expression
                return ASTTypeWithInit(type, init)
        except DefinitionError as e:
            self.pos = pos
            eExpr = e
        if not self.skip_string("="):
            return ASTTypeWithInit(type, None)
        try:
            typeInit = self._parse_type(named=False, outer=None)
            return ASTTemplateParamConstrainedTypeWithInit(type, typeInit)
        except DefinitionError as eType:
            if eExpr is None:
                raise eType
            errs = []
            errs.append((eExpr, "If default template argument is an expression"))
            errs.append((eType, "If default template argument is a type"))
            msg = "Error in non-type template parameter"
            msg += " or constrained template parameter."
            raise self._make_multi_error(errs, msg) from eType

    def _parse_type_using(self) -> ASTTypeUsing:
        name = self._parse_nested_name()
        self.skip_ws()
        if not self.skip_string('='):
            return ASTTypeUsing(name, None)
        type = self._parse_type(False, None)
        return ASTTypeUsing(name, type)

    def _parse_constraints(self) -> List[ASTConstrainedType]:
        type_constraints = []
        self.skip_ws()
        while self.skip_string('where'):
            self.skip_ws()
            type_name = self._parse_nested_name()
            self.skip_ws()
            if not self.skip_string(':'):
                self.fail('Expected ":" for definition of type constraints')
            self.skip_ws()
            constraints = [self._parse_type(False, None)]
            self.skip_ws()
            while self.skip_string(','):
                self.skip_ws()
                constraints.append(self._parse_type())
                self.skip_ws()
            type_constraints.append(ASTConstrainedType(type_name, constraints))
            self.skip_ws()
        return type_constraints

    def _parse_method(self) -> ASTMethod:
        type_declaration = self._parse_type(named=True, outer='method')
        type_constraints = self._parse_constraints()
        return ASTMethod(type_declaration.declSpecs, type_declaration.decl, type_constraints)

    def _parse_delegate(self) -> ASTMethod:
        type_declaration = self._parse_type(named=True, outer='delegate')
        type_constraints = self._parse_constraints()
        return ASTMethod(type_declaration.declSpecs, type_declaration.decl, type_constraints)

    def _parse_property(self) -> ASTProperty:
        declSpecs = self._parse_decl_specs(outer='property')
        decl = self._parse_declarator(named='True', paramMode='property')
        self.skip_ws()
        accessors = []
        self.skip_ws()
        if self.skip_string('{'):
            while 1:
                self.skip_ws()
                visibility = None
                if self.match(_visibility_re):
                    visibility = self.matched_text
                    self.skip_ws()
                accessorIdent = self._parse_property_accessor()
                self.skip_ws()
                accessors.append(ASTAccessor(accessorIdent, visibility))
                self.skip_ws()
                if not self.skip_string(';'):
                    self.fail('Expected semicolon (;)')
                self.skip_ws()
                pos = self.pos
                if self.skip_string('}'):
                    break
                else:
                    self.pos = pos
        return ASTProperty(declSpecs, decl, accessors)

    def _parse_class(self) -> ASTClass:
        name = self._parse_nested_name()
        self.skip_ws()
        bases = []
        self.skip_ws()
        if self.skip_string(':'):
            while 1:
                self.skip_ws()
                visibility = None
                virtual = False
                if self.skip_word_and_ws('virtual'):
                    virtual = True
                if self.match(_visibility_re):
                    visibility = self.matched_text
                    self.skip_ws()
                if not virtual and self.skip_word_and_ws('virtual'):
                    virtual = True
                baseName = self._parse_nested_name()
                self.skip_ws()
                bases.append(ASTBaseClass(baseName, visibility, virtual))
                self.skip_ws()
                if self.skip_string(','):
                    continue
                else:
                    break
        self.skip_ws()
        type_constraints = self._parse_constraints()
        return ASTClass(name, bases, type_constraints)

    def _parse_enum(self) -> ASTEnum:
        scoped = None  # is set by CSharpEnumObject
        self.skip_ws()
        name = self._parse_nested_name()
        self.skip_ws()
        underlyingType = None
        if self.skip_string(':'):
            underlyingType = self._parse_type(named=False)
        return ASTEnum(name, scoped, underlyingType)

    def _parse_enumerator(self) -> ASTEnumerator:
        name = self._parse_nested_name()
        self.skip_ws()
        init = None
        if self.skip_string('='):
            self.skip_ws()

            def parser() -> ASTExpression:
                return self._parse_constant_expression(inTemplate=False)
            initVal = self._parse_expression_fallback([], parser)
            init = ASTInitializer(initVal)
        return ASTEnumerator(name, init)

    # ==========================================================================

    def _parse_template_parameter(self) -> ASTTemplateParam:
        self.skip_ws()
        if self.skip_word('template'):
            # declare a tenplate template parameter
            nestedParams = self._parse_template_parameter_list()
        else:
            nestedParams = None

        pos = self.pos
        try:
            # Unconstrained type parameter or template type parameter
            key = None
            self.skip_ws()
            if self.skip_word_and_ws('interface'):
                key = 'interface'
            elif self.skip_word_and_ws('class'):
                key = 'class'
            elif nestedParams:
                self.fail("Expected 'interface' or 'class' after "
                          "template template parameter list.")
            else:
                self.fail("Expected 'interface' or 'class' in tbe "
                          "beginning of template type parameter.")
            self.skip_ws()
            if self.match(identifier_re):
                matched_text = self.matched_text
                if self.match(_array_indexer_re):
                    matched_text += self.matched_text
                identifier = ASTIdentifier(matched_text)
            else:
                identifier = None
            self.skip_ws()
            if self.current_char not in ',>':
                self.fail('Expected "," or ">" after (template) type parameter.')
            data = ASTTemplateKeyParamPackIdDefault(key, identifier)
            if nestedParams:
                return ASTTemplateParamTemplateType(nestedParams, data)
            else:
                return ASTTemplateParamType(data)
        except DefinitionError as eType:
            if nestedParams:
                raise
            try:
                # non-type parameter or constrained type parameter
                self.pos = pos
                param = self._parse_type_with_init('maybe', 'templateParam')
                return ASTTemplateParamNonType(param)
            except DefinitionError as eNonType:
                self.pos = pos
                header = "Error when parsing template parameter."
                errs = []
                errs.append(
                    (eType, "If unconstrained type parameter or template type parameter"))
                errs.append(
                    (eNonType, "If constrained type parameter or non-type parameter"))
                raise self._make_multi_error(errs, header)

    def _parse_template_parameter_list(self) -> ASTTemplateParams:
        # only: '<' parameter-list '>'
        # we assume that 'template' has just been parsed
        templateParams = []  # type: List[ASTTemplateParam]
        self.skip_ws()
        if not self.skip_string("<"):
            self.fail("Expected '<' after 'template'")
        while 1:
            pos = self.pos
            err = None
            try:
                param = self._parse_template_parameter()
                templateParams.append(param)
            except DefinitionError as eParam:
                self.pos = pos
                err = eParam
            self.skip_ws()
            if self.skip_string('>'):
                return ASTTemplateParams(templateParams)
            elif self.skip_string(','):
                continue
            else:
                header = "Error in template parameter list."
                errs = []
                if err:
                    errs.append((err, "If parameter"))
                try:
                    self.fail('Expected "," or ">".')
                except DefinitionError as e:
                    errs.append((e, "If no parameter"))
                print(errs)
                raise self._make_multi_error(errs, header)

    def _parse_template_introduction(self) -> ASTTemplateIntroduction:
        pos = self.pos
        try:
            concept = self._parse_nested_name()
        except Exception:
            self.pos = pos
            return None
        self.skip_ws()
        if not self.skip_string('{'):
            self.pos = pos
            return None

        # for sure it must be a template introduction now
        params = []
        while 1:
            self.skip_ws()
            self.skip_ws()
            if not self.match(identifier_re):
                self.fail("Expected identifier in template introduction list.")
            txt_identifier = self.matched_text
            # make sure there isn't a keyword
            if txt_identifier in _keywords:
                self.fail("Expected identifier in template introduction list, "
                          "got keyword: %s" % txt_identifier)
            if self.match(_array_indexer_re):
                txt_identifier += self.matched_text
            identifier = ASTIdentifier(txt_identifier)
            params.append(ASTTemplateIntroductionParameter(identifier))

            self.skip_ws()
            if self.skip_string('}'):
                break
            elif self.skip_string(','):
                continue
            else:
                self.fail("Error in template introduction list. "
                          'Expected ",", or "}".')
        return ASTTemplateIntroduction(concept, params)

    def parse_declaration(self, objectType: str, directiveType: str) -> ASTDeclaration:
        if objectType not in ('interface', 'class', 'method', 'delegate', 'property', 'member', 'type', 'enum', 'enumerator'):
            raise Exception('Internal error, unknown objectType "%s".' % objectType)
        if directiveType not in ('interface', 'class', 'struct', 'method', 'delegate', 'property', 'member', 'var', 'type', 'enum', 'enumerator'):
            raise Exception('Internal error, unknown directiveType "%s".' % directiveType)
        visibility = None
        declaration = None  # type: Any

        self.skip_ws()
        if self.match(_visibility_re):
            visibility = self.matched_text

        if objectType == 'type':
            prevErrors = []
            pos = self.pos
            try:
                declaration = self._parse_type(named=True, outer='type')
            except DefinitionError as e:
                prevErrors.append((e, "If typedef-like declaration"))
                self.pos = pos
            pos = self.pos
            try:
                if not declaration:
                    declaration = self._parse_type_using()
            except DefinitionError as e:
                self.pos = pos
                prevErrors.append((e, "If type alias or template alias"))
                header = "Error in type declaration."
                raise self._make_multi_error(prevErrors, header) from e
        elif objectType == 'member':
            declaration = self._parse_type_with_init(named=True, outer='member')
        elif objectType == 'method':
            declaration = self._parse_method()
        elif objectType == 'delegate':
            declaration = self._parse_delegate()
        elif objectType == 'property':
            declaration = self._parse_property()
        elif objectType == 'interface':
            declaration = self._parse_class()
        elif objectType == 'class':
            declaration = self._parse_class()
        elif objectType == 'enum':
            declaration = self._parse_enum()
        elif objectType == 'enumerator':
            declaration = self._parse_enumerator()
        else:
            assert False

        self.skip_ws()
        semicolon = self.skip_string(';')
        return ASTDeclaration(objectType, directiveType, visibility,
                              declaration, semicolon)

    def parse_namespace_object(self) -> ASTNamespace:
        name = self._parse_nested_name()
        res = ASTNamespace(name)
        res.objectType = 'namespace'  # type: ignore
        return res

    def parse_xref_object(self) -> Tuple[Union[ASTNamespace, ASTDeclaration], bool]:
        pos = self.pos
        try:
            name = self._parse_nested_name()
            # if there are '()' left, just skip them
            self.skip_ws()
            self.skip_string('()')
            self.assert_end()
            res1 = ASTNamespace(name)
            res1.objectType = 'xref'  # type: ignore
            return res1, True
        except DefinitionError as e1:
            try:
                self.pos = pos
                res2 = self.parse_declaration('method', 'method')
                self.assert_end()
                return res2, False
            except DefinitionError as e2:
                try:
                    self.pos = pos
                    res3 = self.parse_declaration('property', 'property')
                    self.assert_end()
                    return res3, False
                except DefinitionError as e3:
                    errs = []
                    errs.append((e1, "If shorthand ref"))
                    errs.append((e2, "If full function ref"))
                    errs.append((e3, "If full property ref"))
                    msg = "Error in cross-reference."
                    raise self._make_multi_error(errs, msg) from e3

    def parse_expression(self) -> Union[ASTExpression, ASTType]:
        pos = self.pos
        try:
            expr = self._parse_expression()
            self.skip_ws()
            self.assert_end()
            return expr
        except DefinitionError as exExpr:
            self.pos = pos
            try:
                typ = self._parse_type(False)
                self.skip_ws()
                self.assert_end()
                return typ
            except DefinitionError as exType:
                header = "Error when parsing (type) expression."
                errs = []
                errs.append((exExpr, "If expression"))
                errs.append((exType, "If type"))
                raise self._make_multi_error(errs, header) from exType


def _make_phony_error_name() -> ASTNestedName:
    nne = ASTNestedNameElement(ASTIdentifier("PhonyNameDueToError"), None)
    return ASTNestedName([nne], rooted=False)


class CSharpObject(ObjectDescription[ASTDeclaration]):
    """Description of a C# language object."""

    doc_field_types = [
        GroupedField('parameter', label=_('Parameters'),
                     names=('param', 'parameter', 'arg', 'argument'),
                     can_collapse=True),
        GroupedField('template parameter', label=_('Template Parameters'),
                     names=('tparam', 'template parameter'),
                     can_collapse=True),
        GroupedField('exceptions', label=_('Throws'), rolename='class',
                     names=('throws', 'throw', 'exception'),
                     can_collapse=True),
        GroupedField('remarks', label=_('Remarks'),
                     names=('remark', 'remarks'),
                     can_collapse=True)
    ]

    option_spec = {
        'noindexentry': directives.flag,
        'tparam-line-spec': directives.flag,
    }

    def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
        assert ast.objectType == 'enumerator'
        # find the parent, if it exists && is an enum
        #                     && it's unscoped,
        #                  then add the name to the parent scope
        symbol = ast.symbol
        assert symbol
        assert symbol.identOrOp is not None
        assert symbol.templateParams is None
        assert symbol.templateArgs is None
        parentSymbol = symbol.parent
        assert parentSymbol
        if parentSymbol.parent is None:
            # TODO: we could warn, but it is somewhat equivalent to unscoped
            # enums, without the enum
            return  # no parent
        parentDecl = parentSymbol.declaration
        if parentDecl is None:
            # the parent is not explicitly declared
            # TODO: we could warn, but it could be a style to just assume
            # enumerator parents to be scoped
            return
        if parentDecl.objectType != 'enum':
            # TODO: maybe issue a warning, enumerators in non-enums is weird,
            # but it is somewhat equivalent to unscoped enums, without the enum
            return
        if parentDecl.directiveType != 'enum':
            return

        targetSymbol = parentSymbol.parent
        s = targetSymbol.find_identifier(symbol.identOrOp, matchSelf=False, recurseInAnon=True,
                                         searchInSiblings=False)
        if s is not None:
            # something is already declared with that name
            return
        declClone = symbol.declaration.clone()
        declClone.enumeratorScopedSymbol = symbol
        Symbol(parent=targetSymbol, identOrOp=symbol.identOrOp,
               templateParams=None, templateArgs=None,
               declaration=declClone,
               docname=self.env.docname, line=self.get_source_info()[1])

    def add_target_and_index(self, ast: ASTDeclaration, sig: str,
                             signode: TextElement) -> None:
        # general note: name must be lstrip(':')'ed, to remove "::"
        id = ast.get_id()
        assert id  # shouldn't be None
        if not re.compile(r'^[a-zA-Z0-9_]*$').match(id):
            logger.warning('Index id generation for C# object "%s" failed, please '
                           'report as bug (id=%s).', ast, id,
                           location=self.get_source_info())
        name = ast.symbol.get_full_nested_name().get_display_string().lstrip(':')
        # Add index entry, but not if it's a declaration inside a concept
        s = ast.symbol.parent
        while s is not None:
            decl = s.declaration
            s = s.parent
            if decl is None:
                continue
        if 'noindexentry' not in self.options:
            strippedName = name
            indexText = self.get_index_text(strippedName)
            self.indexnode['entries'].append(('single', indexText, id, '', None))

        if id not in self.state.document.ids:
            # if the name is not unique, the first one will win
            names = self.env.domaindata['csharp']['names']
            if name not in names:
                names[name] = ast.symbol.docname
            # always add the newest id
            assert id
            signode['ids'].append(id)
            self.state.document.note_explicit_target(signode)

    @property
    def object_type(self) -> str:
        raise NotImplementedError()

    @property
    def display_object_type(self) -> str:
        return self.object_type

    def get_index_text(self, name: str) -> str:
        return _('%s (C# %s)') % (name, self.display_object_type)

    def parse_definition(self, parser: DefinitionParser) -> ASTDeclaration:
        return parser.parse_declaration(self.object_type, self.objtype)

    def describe_signature(self, signode: desc_signature,
                           ast: ASTDeclaration, options: Dict) -> None:
        ast.describe_signature(signode, 'lastIsName', self.env, options)

    def run(self) -> List[Node]:
        env = self.state.document.settings.env  # from ObjectDescription.run
        if 'csharp:parent_symbol' not in env.temp_data:
            root = env.domaindata['csharp']['root_symbol']
            env.temp_data['csharp:parent_symbol'] = root
            env.ref_context['csharp:parent_key'] = root.get_lookup_key()

        # The lookup keys assume that no nested scopes exists inside overloaded functions.
        # (see also #5191)
        # Example:
        # .. csharp:function:: void f(int)
        # .. csharp:function:: void f(double)
        #
        #    .. csharp:function:: void g()
        #
        #       :csharp:any:`boom`
        #
        # So we disallow any signatures inside functions.
        parentSymbol = env.temp_data['csharp:parent_symbol']
        parentDecl = parentSymbol.declaration
        if parentDecl is not None and parentDecl.objectType == 'method':
            msg = "C# declarations inside methods are not supported." \
                  " Parent function: {}\nDirective name: {}\nDirective arg: {}"
            logger.warning(msg.format(
                str(parentSymbol.get_full_nested_name()),
                self.name, self.arguments[0]
            ), location=self.get_source_info())
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            env.temp_data['csharp:last_symbol'] = symbol
            return []
        # When multiple declarations are made in the same directive
        # they need to know about each other to provide symbol lookup for function parameters.
        # We use last_symbol to store the latest added declaration in a directive.
        env.temp_data['csharp:last_symbol'] = None
        return super().run()

    def handle_signature(self, sig: str, signode: desc_signature) -> ASTDeclaration:
        parentSymbol = self.env.temp_data['csharp:parent_symbol']  # type: Symbol

        parser = DefinitionParser(sig, location=signode, config=self.env.config)
        try:
            ast = self.parse_definition(parser)
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=signode)
            # It is easier to assume some phony name than handling the error in
            # the possibly inner declarations.
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            self.env.temp_data['csharp:last_symbol'] = symbol
            raise ValueError from e

        try:
            symbol = parentSymbol.add_declaration(
                ast, docname=self.env.docname, line=self.get_source_info()[1])
            # append the new declaration to the sibling list
            assert symbol.siblingAbove is None
            assert symbol.siblingBelow is None
            symbol.siblingAbove = self.env.temp_data['csharp:last_symbol']
            if symbol.siblingAbove is not None:
                assert symbol.siblingAbove.siblingBelow is None
                symbol.siblingAbove.siblingBelow = symbol
            self.env.temp_data['csharp:last_symbol'] = symbol
        except _DuplicateSymbolError as e:
            # Assume we are actually in the old symbol,
            # instead of the newly created duplicate.
            self.env.temp_data['csharp:last_symbol'] = e.symbol
            msg = __("Duplicate C# declaration, also defined at %s:%s.\n"
                     "Declaration is '.. csharp:%s:: %s'.")
            msg = msg % (e.symbol.docname, e.symbol.line,
                         self.display_object_type, sig)
            logger.warning(msg, location=signode)

        if ast.objectType == 'enumerator':
            self._add_enumerator_to_parent(ast)

        # note: handle_signature may be called multiple time per directive,
        # if it has multiple signatures, so don't mess with the original options.
        options = dict(self.options)
        options['tparam-line-spec'] = 'tparam-line-spec' in self.options
        self.describe_signature(signode, ast, options)
        return ast

    def before_content(self) -> None:
        lastSymbol = self.env.temp_data['csharp:last_symbol']  # type: Symbol
        assert lastSymbol
        self.oldParentSymbol = self.env.temp_data['csharp:parent_symbol']
        self.oldParentKey = self.env.ref_context['csharp:parent_key']  # type: LookupKey
        self.env.temp_data['csharp:parent_symbol'] = lastSymbol
        self.env.ref_context['csharp:parent_key'] = lastSymbol.get_lookup_key()

    def after_content(self) -> None:
        self.env.temp_data['csharp:parent_symbol'] = self.oldParentSymbol
        self.env.ref_context['csharp:parent_key'] = self.oldParentKey


class CSharpTypeObject(CSharpObject):
    object_type = 'type'


class CSharpMemberObject(CSharpObject):
    object_type = 'member'


class CSharpMethodObject(CSharpObject):
    object_type = 'method'
    doc_field_types = CSharpObject.doc_field_types + [
        GroupedField('returnvalue', label=_('Returns'),
                     names=('returns', 'return'),
                     can_collapse=True)
    ]

    @property
    def display_object_type(self) -> str:
        # the distinction between method and delegate is only cosmetic
        assert self.objtype in ('method', 'delegate')
        return self.objtype

class CSharpPropertyObject(CSharpObject):
    object_type = 'property'
    doc_field_types = CSharpObject.doc_field_types + [
        GroupedField('value', label=_('Value'),
                     names=('value', 'val'),
                     can_collapse=True)
    ]


class CSharpClassObject(CSharpObject):
    object_type = 'class'

    @property
    def display_object_type(self) -> str:
        # the distinction between interface, class and struct is only cosmetic
        assert self.objtype in ('interface', 'class', 'struct')
        return self.objtype


class CSharpEnumObject(CSharpObject):
    object_type = 'enum'


class CSharpEnumeratorObject(CSharpObject):
    object_type = 'enumerator'


class CSharpNamespaceObject(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting stuff in
    namespace foo.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        rootSymbol = self.env.domaindata['csharp']['root_symbol']
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            symbol = rootSymbol
            stack = []  # type: List[Symbol]
        else:
            parser = DefinitionParser(self.arguments[0],
                                      location=self.get_source_info(),
                                      config=self.config)
            try:
                ast = parser.parse_namespace_object()
                parser.assert_end()
            except DefinitionError as e:
                logger.warning(e, location=self.get_source_info())
                name = _make_phony_error_name()
                ast = ASTNamespace(name, None)
            symbol = rootSymbol.add_name(ast.nestedName)
            stack = [symbol]
        self.env.temp_data['csharp:parent_symbol'] = symbol
        self.env.temp_data['csharp:namespace_stack'] = stack
        self.env.ref_context['csharp:parent_key'] = symbol.get_lookup_key()
        return []


class CSharpNamespacePushObject(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            return []
        parser = DefinitionParser(self.arguments[0],
                                  location=self.get_source_info(),
                                  config=self.config)
        try:
            ast = parser.parse_namespace_object()
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=self.get_source_info())
            name = _make_phony_error_name()
            ast = ASTNamespace(name, None)
        oldParent = self.env.temp_data.get('csharp:parent_symbol', None)
        if not oldParent:
            oldParent = self.env.domaindata['csharp']['root_symbol']
        symbol = oldParent.add_name(ast.nestedName)
        stack = self.env.temp_data.get('csharp:namespace_stack', [])
        stack.append(symbol)
        self.env.temp_data['csharp:parent_symbol'] = symbol
        self.env.temp_data['csharp:namespace_stack'] = stack
        self.env.ref_context['csharp:parent_key'] = symbol.get_lookup_key()
        return []


class CSharpNamespacePopObject(SphinxDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        stack = self.env.temp_data.get('csharp:namespace_stack', None)
        if not stack or len(stack) == 0:
            logger.warning("C# namespace pop on empty stack. Defaulting to gobal scope.",
                           location=self.get_source_info())
            stack = []
        else:
            stack.pop()
        if len(stack) > 0:
            symbol = stack[-1]
        else:
            symbol = self.env.domaindata['csharp']['root_symbol']
        self.env.temp_data['csharp:parent_symbol'] = symbol
        self.env.temp_data['csharp:namespace_stack'] = stack
        self.env.ref_context['csharp:parent_key'] = symbol.get_lookup_key()
        return []


class AliasNode(nodes.Element):
    def __init__(self, sig: str, aliasOptions: dict,
                 env: "BuildEnvironment" = None,
                 parentKey: LookupKey = None) -> None:
        super().__init__()
        self.sig = sig
        self.aliasOptions = aliasOptions
        if env is not None:
            if 'csharp:parent_symbol' not in env.temp_data:
                root = env.domaindata['csharp']['root_symbol']
                env.temp_data['csharp:parent_symbol'] = root
            self.parentKey = env.temp_data['csharp:parent_symbol'].get_lookup_key()
        else:
            assert parentKey is not None
            self.parentKey = parentKey

    def copy(self) -> 'AliasNode':
        return self.__class__(self.sig, self.aliasOptions,
                              env=None, parentKey=self.parentKey)


class AliasTransform(SphinxTransform):
    default_priority = ReferencesResolver.default_priority - 1

    def _render_symbol(self, s: Symbol, maxdepth: int, skipThis: bool,
                       aliasOptions: dict, renderOptions: dict,
                       document: Any) -> List[Node]:
        if maxdepth == 0:
            recurse = True
        elif maxdepth == 1:
            recurse = False
        else:
            maxdepth -= 1
            recurse = True

        nodes = []  # type: List[Node]
        if not skipThis:
            signode = addnodes.desc_signature('', '')
            nodes.append(signode)
            s.declaration.describe_signature(signode, 'markName', self.env, renderOptions)

        if recurse:
            if skipThis:
                childContainer = nodes  # type: Union[List[Node], addnodes.desc]
            else:
                content = addnodes.desc_content()
                desc = addnodes.desc()
                content.append(desc)
                desc.document = document
                desc['domain'] = 'csharp'
                # 'desctype' is a backwards compatible attribute
                desc['objtype'] = desc['desctype'] = 'alias'
                desc['noindex'] = True
                childContainer = desc

            for sChild in s._children:
                if sChild.declaration is None:
                    continue
                if sChild.declaration.objectType in ("templateParam", "functionParam"):
                    continue
                childNodes = self._render_symbol(
                    sChild, maxdepth=maxdepth, skipThis=False,
                    aliasOptions=aliasOptions, renderOptions=renderOptions,
                    document=document)
                childContainer.extend(childNodes)

            if not skipThis and len(desc.children) != 0:
                nodes.append(content)
        return nodes

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.traverse(AliasNode):
            node = cast(AliasNode, node)
            sig = node.sig
            parentKey = node.parentKey
            try:
                parser = DefinitionParser(sig, location=node,
                                          config=self.env.config)
                ast, isShorthand = parser.parse_xref_object()
                parser.assert_end()
            except DefinitionError as e:
                logger.warning(e, location=node)
                ast, isShorthand = None, None

            if ast is None:
                # could not be parsed, so stop here
                signode = addnodes.desc_signature(sig, '')
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                node.replace_self(signode)
                continue

            rootSymbol = self.env.domains['csharp'].data['root_symbol']  # type: Symbol
            parentSymbol = rootSymbol.direct_lookup(parentKey)  # type: Symbol
            if not parentSymbol:
                print("Target: ", sig)
                print("ParentKey: ", parentKey)
                print(rootSymbol.dump(1))
            assert parentSymbol  # should be there

            symbols = []  # type: List[Symbol]
            if isShorthand:
                assert isinstance(ast, ASTNamespace)
                ns = ast
                name = ns.nestedName
                templateDecls = []
                symbols, failReason = parentSymbol.find_name(
                    nestedName=name,
                    templateDecls=templateDecls,
                    typ='any',
                    templateShorthand=True,
                    matchSelf=True, recurseInAnon=True,
                    searchInSiblings=False)
                if symbols is None:
                    symbols = []
            else:
                assert isinstance(ast, ASTDeclaration)
                decl = ast
                name = decl.name
                s = parentSymbol.find_declaration(decl, 'any',
                                                  templateShorthand=True,
                                                  matchSelf=True, recurseInAnon=True)
                if s is not None:
                    symbols.append(s)

            symbols = [s for s in symbols if s.declaration is not None]

            if len(symbols) == 0:
                signode = addnodes.desc_signature(sig, '')
                node.append(signode)
                signode.clear()
                signode += addnodes.desc_name(sig, sig)

                logger.warning("Can not find C# declaration for alias '%s'." % ast,
                               location=node)
                node.replace_self(signode)
            else:
                nodes = []
                renderOptions = {
                    'tparam-line-spec': False,
                }
                for s in symbols:
                    assert s.declaration is not None
                    res = self._render_symbol(
                        s, maxdepth=node.aliasOptions['maxdepth'],
                        skipThis=node.aliasOptions['noroot'],
                        aliasOptions=node.aliasOptions,
                        renderOptions=renderOptions,
                        document=node.document)
                    nodes.extend(res)
                node.replace_self(nodes)


class CSharpAliasObject(ObjectDescription):
    option_spec = {
        'maxdepth': directives.nonnegative_int,
        'noroot': directives.flag,
    }  # type: Dict

    def run(self) -> List[Node]:
        """
        On purpose this doesn't call the ObjectDescription version, but is based on it.
        Each alias signature may expand into multiple real signatures (an overload set).
        The code is therefore based on the ObjectDescription version.
        """
        if ':' in self.name:
            self.domain, self.objtype = self.name.split(':', 1)
        else:
            self.domain, self.objtype = '', self.name

        node = addnodes.desc()
        node.document = self.state.document
        node['domain'] = self.domain
        # 'desctype' is a backwards compatible attribute
        node['objtype'] = node['desctype'] = self.objtype

        self.names = []  # type: List[str]
        aliasOptions = {
            'maxdepth': self.options.get('maxdepth', 1),
            'noroot': 'noroot' in self.options,
        }
        if aliasOptions['noroot'] and aliasOptions['maxdepth'] == 1:
            logger.warning("Error in C# alias declaration."
                           " Requested 'noroot' but 'maxdepth' 1."
                           " When skipping the root declaration,"
                           " need 'maxdepth' 0 for infinite or at least 2.",
                           location=self.get_source_info())
        signatures = self.get_signatures()
        for i, sig in enumerate(signatures):
            node.append(AliasNode(sig, aliasOptions, env=self.env))

        contentnode = addnodes.desc_content()
        node.append(contentnode)
        self.before_content()
        self.state.nested_parse(self.content, self.content_offset, contentnode)
        self.env.temp_data['object'] = None
        self.after_content()
        return [node]


class CSharpXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element, has_explicit_title: bool,
                     title: str, target: str) -> Tuple[str, str]:
        refnode.attributes.update(env.ref_context)

        if not has_explicit_title:
            # major hax: replace anon names via simple string manipulation.
            # Can this actually fail?
            title = anon_identifier_re.sub("[anonymous]", str(title))

        if refnode['reftype'] == 'any':
            # Assume the removal part of fix_parens for :any: refs.
            # The addition part is done with the reference is resolved.
            if not has_explicit_title and title.endswith('()'):
                title = title[:-2]
            if target.endswith('()'):
                target = target[:-2]
        # TODO: should this really be here?
        if not has_explicit_title:
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[:1] == '~':
                title = title[1:]
                dcolon = title.rfind('.')
                if dcolon != -1:
                    title = title[dcolon + 1:]
        return title, target


class CSharpDomain(Domain):
    """C# language domain.

    There are two 'object type' attributes being used::

    - Each object created from directives gets an assigned .objtype from ObjectDescription.run.
      This is simply the directive name.
    - Each declaration (see the distinction in the directives dict below) has a nested .ast of
      type ASTDeclaration. That object has .objectType which corresponds to the keys in the
      object_types dict below. They are the core different types of declarations in C# that
      one can document.
    """
    name = 'csharp'
    label = 'C#'
    object_types = {
        'class':      ObjType(_('class'),      'class', 'struct', 'intf', 'identifier', 'type'),
        'method':     ObjType(_('method'),     'meth', 'delegate', 'identifier', 'type'),
        'property':   ObjType(_('property'),   'prop',            'identifier', 'type'),
        'member':     ObjType(_('member'),     'member', 'var',   'identifier'),
        'type':       ObjType(_('type'),                          'identifier', 'type'),
        'enum':       ObjType(_('enum'),       'enum',            'identifier', 'type'),
        'enumerator': ObjType(_('enumerator'), 'enumerator',      'identifier', 'member', 'type'),
        # generated object types
        'functionParam': ObjType(_('function parameter'),         'identifier', 'member', 'var'),  # noqa
        'templateParam': ObjType(_('template parameter'), 'identifier', 'class', 'struct', 'member', 'var', 'type')  # noqa
    }

    directives = {
        # declarations
        'interface': CSharpClassObject,
        'class': CSharpClassObject,
        'struct': CSharpClassObject,
        'method': CSharpMethodObject,
        'delegate': CSharpMethodObject,
		'property': CSharpPropertyObject,
        'member': CSharpMemberObject,
        'var': CSharpMemberObject,
        'type': CSharpTypeObject,
        'enum': CSharpEnumObject,
        'enumerator': CSharpEnumeratorObject,
        # scope control
        'namespace': CSharpNamespaceObject,
        'namespace-push': CSharpNamespacePushObject,
        'namespace-pop': CSharpNamespacePopObject,
        # other
        'alias': CSharpAliasObject
    }
    roles = {
        'any': CSharpXRefRole(),
        'intf': CSharpXRefRole(),
        'class': CSharpXRefRole(),
        'struct': CSharpXRefRole(),
        'meth': CSharpXRefRole(fix_parens=True),
        'delegate': CSharpXRefRole(fix_parens=True),
        'prop': CSharpXRefRole(),
        'member': CSharpXRefRole(),
        'var': CSharpXRefRole(),
        'type': CSharpXRefRole(),
        'enum': CSharpXRefRole(),
        'enumerator': CSharpXRefRole()
    }
    initial_data = {
        'root_symbol': Symbol(None, None, None, None, None, None, None),
        'names': {}  # full name for indexing -> docname
    }

    def clear_doc(self, docname: str) -> None:
        if Symbol.debug_show_tree:
            print("clear_doc:", docname)
            print("\tbefore:")
            print(self.data['root_symbol'].dump(1))
            print("\tbefore end")

        rootSymbol = self.data['root_symbol']
        rootSymbol.clear_doc(docname)

        if Symbol.debug_show_tree:
            print("\tafter:")
            print(self.data['root_symbol'].dump(1))
            print("\tafter end")
            print("clear_doc end:", docname)
        for name, nDocname in list(self.data['names'].items()):
            if nDocname == docname:
                del self.data['names'][name]

    def process_doc(self, env: BuildEnvironment, docname: str,
                    document: nodes.document) -> None:
        if Symbol.debug_show_tree:
            print("process_doc:", docname)
            print(self.data['root_symbol'].dump(0))
            print("process_doc end:", docname)

    def process_field_xref(self, pnode: pending_xref) -> None:
        pnode.attributes.update(self.env.ref_context)

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        if Symbol.debug_show_tree:
            print("merge_domaindata:")
            print("\tself:")
            print(self.data['root_symbol'].dump(1))
            print("\tself end")
            print("\tother:")
            print(otherdata['root_symbol'].dump(1))
            print("\tother end")

        self.data['root_symbol'].merge_with(otherdata['root_symbol'],
                                            docnames, self.env)
        ourNames = self.data['names']
        for name, docname in otherdata['names'].items():
            if docname in docnames:
                if name not in ourNames:
                    ourNames[name] = docname
                # no need to warn on duplicates, the symbol merge already does that
        if Symbol.debug_show_tree:
            print("\tresult:")
            print(self.data['root_symbol'].dump(1))
            print("\tresult end")
            print("merge_domaindata end")

    def _resolve_xref_inner(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                            typ: str, target: str, node: pending_xref,
                            contnode: Element) -> Tuple[Element, str]:
        # add parens again for those that could be functions
        if target.startswith('msdn'):
            target = target[4:]
            contnode = nodes.Text(target.split('.')[-1])
            node = get_msdn_ref(target, contnode)
            if node:
                return node, typ
            else:
                raise Exception('No MSDN reference found for {}'.format(target))

        parensAdded = False
        if typ == 'any' or typ == 'meth' or typ == 'delegate':
            if not '(' in target:
                target += '()'
                parensAdded = True
        parser = DefinitionParser(target, location=node, config=env.config)
        try:
            ast, isShorthand = parser.parse_xref_object()
        except DefinitionError as e:
            # as arg to stop flake8 from complaining
            def findWarning(e: Exception) -> Tuple[str, Exception]:
                if not parensAdded:
                    return target, e
                # hax on top of the paren hax to try to get correct errors
                parser2 = DefinitionParser(target[:-2],
                                           location=node,
                                           config=env.config)
                try:
                    parser2.parse_xref_object()
                except DefinitionError as e2:
                    return target[:-2], e2
                # strange, that we don't get the error now, use the original
                return target, e
            t, ex = findWarning(e)
            logger.warning('Unparseable C# cross-reference: %r\n%s', t, ex,
                           location=node)
            if not typ.endswith('identifier'):
                raise Exception("Reference not resolved for target '{}'".format(target))

            return None, None

        if parensAdded:
            target = target[:-2]

        rootSymbol = self.data['root_symbol']
        if isShorthand:
            assert isinstance(ast, ASTNamespace)
            ns = ast
            name = ns.nestedName
            templateDecls = []
            # let's be conservative with the sibling lookup for now
            searchInSiblings = (not name.rooted) and len(name.names) == 1
            symbols, failReason = rootSymbol.find_name(
                name, templateDecls, typ,
                templateShorthand=True,
                matchSelf=True, recurseInAnon=True,
                searchInSiblings=searchInSiblings)
            if symbols is None:
                if typ == 'identifier':
                    if failReason == 'templateParamInQualified':
                        # this is an xref we created as part of a signature,
                        # so don't warn for names nested in template parameters
                        raise NoUri(str(name), typ)
                s = None
            else:
                if len(symbols) > 1:
                    for symbol in symbols:
                        symbolFound = False
                        if symbol.declaration and symbol.declaration.objectType in typ:
                            s = symbol
                            symbolFound = True
                            break
                    if not symbolFound:
                        # Cannot determine which symbol should be chosen, so choose None
                        s = None
                else:
                    s = symbols[0]
        else:
            assert isinstance(ast, ASTDeclaration)
            decl = ast
            name = decl.name
            s = rootSymbol.find_declaration(decl, typ,
                                              templateShorthand=True,
                                              matchSelf=True, recurseInAnon=True)
        if s is None or s.declaration is None:
            if target.startswith('System.'):
                node = get_msdn_ref(target, contnode)
                if node:
                    return node, typ

            if not env.config.csharp_ignore_unresolved_xrefs and not typ.endswith('identifier'):
                raise Exception("Reference not resolved for target '{}'".format(target))

            return None, None

        if typ.startswith('csharp:'):
            typ = typ[7:]
        declTyp = s.declaration.objectType

        def checkType() -> bool:
            if typ == 'any':
                return True
            objtypes = self.objtypes_for_role(typ)
            if objtypes:
                return declTyp in objtypes
            print("Type is %s, declaration type is %s" % (typ, declTyp))
            assert False
        if not checkType():
            logger.warning("csharp:%s targets a %s (%s).",
                           typ, s.declaration.objectType,
                           s.get_full_nested_name(),
                           location=node)

        declaration = s.declaration
        if isShorthand:
            fullNestedName = s.get_full_nested_name()
            displayName = fullNestedName.get_display_string().lstrip(':')
        else:
            displayName = decl.get_display_string()
        docname = s.docname
        assert docname

        # the non-identifier refs are cross-references, which should be processed:
        # - fix parenthesis due to operator() and add_function_parentheses
        if typ != "identifier":
            title = contnode.pop(0).astext()
            # If it's operator(), we need to add '()' if explicit function parens
            # are requested. Then the Sphinx machinery will add another pair.
            # Also, if it's an 'any' ref that resolves to a function, we need to add
            # parens as well.
            # However, if it's a non-shorthand function ref, for a function that
            # takes no arguments, then we may need to add parens again as well.
            addParen = 0
            if not node.get('refexplicit', False) and declaration.objectType == 'method':
                if isShorthand:
                    # this is just the normal haxing for 'any' roles
                    if env.config.add_function_parentheses and typ == 'any':
                        addParen += 1
                else:
                    # our job here is to essentially nullify add_function_parentheses
                    if env.config.add_function_parentheses:
                        if typ == 'any' and displayName.endswith('()'):
                            addParen += 1
                        elif typ == 'meth' or typ == 'delegate':
                            if title.endswith('()') and not displayName.endswith('()'):
                                title = title[:-2]
                    else:
                        if displayName.endswith('()') and not title.endswith('()'):
                            addParen += 1
            if addParen > 0:
                title += '()' * addParen

            if (typ == 'meth' or typ == 'delegate') and not '(' in title:
                title += '()'
            # and reconstruct the title again
            contnode += nodes.Text(title)
        return make_refnode(builder, fromdocname, docname,
                            declaration.get_id(), contnode, displayName
                            ), declaration.objectType

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                     typ: str, target: str, node: pending_xref, contnode: Element
                     ) -> Element:
        return self._resolve_xref_inner(env, fromdocname, builder, typ,
                                        target, node, contnode)[0]

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                         target: str, node: pending_xref, contnode: Element
                         ) -> List[Tuple[str, Element]]:
        return []

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        rootSymbol = self.data['root_symbol']
        for symbol in rootSymbol.get_all_symbols():
            if symbol.declaration is None:
                continue
            assert symbol.docname
            fullNestedName = symbol.get_full_nested_name()
            name = str(fullNestedName).lstrip(':')
            dispname = fullNestedName.get_display_string().lstrip(':')
            objectType = symbol.declaration.objectType
            docname = symbol.docname
            newestId = symbol.declaration.get_id()
            yield (name, dispname, objectType, docname, newestId, 1)

    def get_full_qualified_name(self, node: Element) -> str:
        target = node.get('reftarget', None)
        if target is None:
            return None
        parentKey = node.get("csharp:parent_key", None)  # type: LookupKey
        if parentKey is None or len(parentKey.data) <= 0:
            return None

        rootSymbol = self.data['root_symbol']
        parentSymbol = rootSymbol.direct_lookup(parentKey)
        parentName = parentSymbol.get_full_nested_name()
        return '.'.join([str(parentName), target])


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_domain(CSharpDomain)
    app.add_config_value("csharp_id_attributes", [], 'env')
    app.add_config_value("csharp_paren_attributes", [], 'env')
    app.add_post_transform(AliasTransform)

    app.add_config_value("csharp_ignore_unresolved_xrefs", False, '')

    # debug stuff
    app.add_config_value("csharp_debug_lookup", False, '')
    app.add_config_value("csharp_debug_show_tree", False, '')

    def setDebugFlags(app):
        Symbol.debug_lookup = app.config.csharp_debug_lookup
        Symbol.debug_show_tree = app.config.csharp_debug_show_tree
    app.connect("builder-inited", setDebugFlags)

    return {
        'version': 'builtin',
        'env_version': 4,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
