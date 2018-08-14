import sys
import weakref
from enum import Enum
from typing import Callable, List, Dict, Union, Iterable, Optional, Any, Tuple, TypeVar
from dataclasses import dataclass

from cephlib.units import b2ssize, b2ssize_10


assert sys.version_info >= (3, 6), "This python module must run on 3.6, it requires buidin dict ordering"


class RTag:
    def __getattr__(self, name: str) -> Callable:
        def closure(text: str = "", **attrs: str) -> str:
            name2 = name.replace("_", '-')

            if '_class' in attrs:
                attrs['class'] = attrs.pop('_class')

            if len(attrs) == 0:
                sattrs = ""
            else:
                sattrs = " " + " ".join(f'{name2}="{val}"' for name2, val in attrs.items())

            if name2 == 'br':
                assert text == ""
                assert attrs == {}
                return "<br>"
            elif text == "" and name2 not in ('script', 'link'):
                return f"<{name2}{sattrs} />"
            elif name2 == 'link':
                assert text == ''
                return f"<link{sattrs}>"
            else:
                return f"<{name2}{sattrs}>{text}</{name2}>"
        return closure


rtag = RTag()


class TagProxy:
    def __init__(self, doc: 'Doc', name :str) -> None:
        self.__doc = doc
        self.__name = name
        self.__text = ""
        self.__attrs: Dict[str, str] = {}
        self.__childs: List[str] = []

    def __call__(self, text: str = "", **attrs) -> 'TagProxy':
        self.__childs.append(text)
        self.__attrs.update(attrs)
        return self

    def __getattr__(self, name: str) -> 'TagProxy':
        tagp = TagProxy(self.__doc, name)
        self.__childs.append(tagp)
        return tagp

    def __enter__(self) -> 'TagProxy':
        self.__doc += self
        return self

    def __exit__(self, x, y, z):
        self.__doc -= self

    def __str__(self) -> str:
        inner = "".join(map(str, self.__childs))
        return getattr(rtag, self.__name)(inner, **self.__attrs)


class Doc:
    def __init__(self) -> None:
        self.__stack: List[TagProxy] = []
        self.__childs: List[TagProxy] = []

    def __getattr__(self, name):
        if len(self.__stack) == 0:
            tagp = TagProxy(self, name)
            self.__childs.append(tagp)
        else:
            tagp = getattr(self.__stack[-1], name)
        return tagp

    def _enter(self, name, text="", **attrs):
        self += getattr(self, name)
        self(text, **attrs)

    def _exit(self):
        self -= self.__stack[-1]

    def __str__(self):
        assert self.__stack == []
        return "".join(map(str, self.__childs))

    def __iadd__(self, tag: TagProxy) -> 'Doc':
        self.__stack.append(tag)
        return self

    def __isub__(self, tag: TagProxy) -> 'Doc':
        assert self.__stack.pop() is tag
        return self

    def __call__(self, text: str = "", **attrs: str):
        assert self.__stack != []
        return self.__stack[-1](text, **attrs)



def html_ok(text: str) -> TagProxy:
    return rtag.font(text, color="green")


def html_fail(text: str) -> TagProxy:
    return rtag.font(text, color="red")


class TableAlign(Enum):
    center = 0
    left_right = 1
    center_right = 2
    left_center = 3


class HTMLTable:
    default_classes = {'table-bordered', 'sortable', 'zebra-table'}

    def __init__(self,
                 id: str = None,
                 headers: Iterable[str] = None,
                 table_attrs: Dict[str, str] = None,
                 zebra: bool = True,
                 header_attrs: Dict[str, str] = None,
                 extra_cls: Iterable[str] = None,
                 sortable: bool = True,
                 align: TableAlign = TableAlign.center) -> None:

        assert not isinstance(extra_cls, str)
        self.table_attrs = table_attrs.copy() if table_attrs is not None else {}
        classes = self.default_classes.copy()

        if extra_cls:
            classes.update(extra_cls)

        if not zebra:
            classes.remove('zebra-table')

        if not sortable:
            classes.remove('sortable')

        if align == TableAlign.center:
            classes.add('table_c')
        elif align == TableAlign.left_right:
            classes.add('table_lr')
        elif align == TableAlign.center_right:
            classes.add('table_cr')
        elif align == TableAlign.left_center:
            classes.add('table_lc')
        else:
            raise ValueError(f"Unknown align type: {align}")

        if id is not None:
            self.table_attrs['id'] = id

        self.table_attrs['class'] = " ".join(classes)

        if header_attrs is None:
            header_attrs = {}

        if headers is not None:
            self.headers = [(header, header_attrs) for header in headers]
        else:
            self.headers = None
        self.cells: List[List] = [[]]

    def add_header(self, text: str, attrs: Dict[str, str] = None):
        self.headers.append((text, attrs))

    def add_cell(self, data: Union[str, TagProxy], **attrs: str):
        self.cells[-1].append((data, attrs))

    def add_cell_b2ssize(self, data: Union[int, float], **attrs: str):
        assert 'sorttable_customkey' not in attrs
        self.add_cell(b2ssize(data), sorttable_customkey=str(data), **attrs)

    def add_cell_b2ssize_10(self, data: Union[int, float], **attrs: str):
        assert 'sorttable_customkey' not in attrs
        self.add_cell(b2ssize_10(data), sorttable_customkey=str(data), **attrs)

    def add_cells(self, *cells: Union[str, TagProxy], **attrs: str):
        self.add_row(cells, **attrs)

    def add_row(self, data: Iterable[Union[str, TagProxy]], **attrs: str):
        for val in data:
            self.add_cell(val, **attrs)
        self.next_row()

    def next_row(self):
        self.cells.append([])

    def __str__(self):
        t = Doc()

        with t.table('', **self.table_attrs):
            with t.thead.tr:
                if self.headers:
                    for header, attrs in self.headers:
                        t.th(header, **attrs)

            with t.tbody:
                for line in self.cells:
                    if line == [] and line is self.cells[-1]:
                        continue
                    with t.tr:
                        for cell, attrs in line:
                            t.td(cell, **attrs)

        return str(t)


@dataclass
class Field:
    tp: Any
    name: Optional[str] = None
    converter: Callable[[Any], Union[str, TagProxy]] = lambda x: str(x)
    custom_sort: Optional[Callable[[Any], str]] = lambda x: str(x)
    allow_none: bool = True
    skip_if_no_data: bool = True
    null_sort_key: str = ''
    null_value: str = ''
    dont_sort: bool = False

    def __post_init__(self):
        assert self.tp is not tuple


@dataclass
class ExtraColumns:
    names: Dict[str, str]
    base_type: Field

    def __getattr__(self, name: str) -> Any:
        if name == 'name':
            raise AttributeError(f"Instance of class {self.__class__} has no attribute 'name'")
        return getattr(self.base_type, name)


T = TypeVar('T')


def partition(items: Iterable[T], size: int) -> Iterable[List[T]]:
    curr = []
    for idx, val in enumerate(items):
        curr.append(val)
        if (idx + 1) % size == 0:
            yield curr
            curr = []

    if curr:
        yield curr


def partition_by_len(items: Iterable[Union[T, Tuple[T, int]]], chars_per_line: int, delimiter_len: int) -> Iterable[List[T]]:
    curr: List[T] = []
    curr_len = 0
    for el_r in items:
        if isinstance(el_r, tuple):
            el, el_len = el_r
        else:
            el = el_r
            el_len = len(str(el))
        if curr_len + delimiter_len + el_len <= chars_per_line:
            curr.append(el)
            curr_len += delimiter_len + el_len
        else:
            yield curr
            curr = [el]
            curr_len = el_len
    if curr:
        yield curr


def count(name: str = None) -> Field:
    return Field(int, name, b2ssize_10, null_sort_key='-1')


def exact_count(name: str = None) -> Field:
    return Field(int, name, null_sort_key='-1')


def bytes_sz(name: str = None) -> Field:
    return Field(int, name, b2ssize, null_sort_key='-1')


def ident(name: str = None, **kwargs) -> Field:
    return Field(str, name, **kwargs)


def to_str(name: str = None) -> Field:
    return Field(None, name)


def float_vl(name: str = None) -> Field:
    return Field((int, float), name, converter=lambda x: f"{x:.2f}")


def seconds(name: str = None) -> Field:
    return count(name)


def idents_list(name: str = None, delim: str = "<br>", chars_per_line: int = None,
                partition_size: int = 1, part_delim: str = ', ') -> Field:
    def converter(vals: List[Any]) -> str:
        if chars_per_line is not None:
            assert partition_size == 1
            data = [((el_r[0], len(el_r[1])) if isinstance(el_r, tuple) else (el_r, len(str(el_r)))) for el_r in vals]
            res = []
            for line in partition_by_len(data, chars_per_line, len(part_delim) if part_delim != ', ' else 1):
                res.append(part_delim.join(map(str, line)))
            return delim.join(res)
        elif partition != 1:
            assert all(not isinstance(vl, tuple) for vl in vals)
            return delim.join(part_delim.join(part) for part in partition(map(str, vals), partition_size))
        else:
            assert all(not isinstance(vl, tuple) for vl in vals)
            return delim.join(map(str, vals))
    return Field(list, name, converter=converter, dont_sort=True)


def ok_or_fail(name: str = None) -> Field:
    return Field(bool, name, converter=lambda ok: html_ok('ok') if ok else html_fail('fail'))


def yes_or_no(name: str = None, true_fine: bool = True) -> Field:
    def converter(val: bool) -> str:
        if true_fine:
            return html_ok('yes') if val else html_fail('no')
        else:
            return html_fail('yes') if val else html_ok('no')
    return Field(bool, name, converter=converter)


def extra_columns(tp: Field, **names: str) -> ExtraColumns:
    return ExtraColumns(names, tp)


class WithAttrs:
    def __init__(self, val: Any, **attrs: str) -> None:
        self.val = val
        self.attrs = attrs


def prepare_field(table: 'Table', name: str, val: Any, must_fld: bool = True) -> Any:
    fld: Union[Field, ExtraColumns] = getattr(table, name)

    if must_fld:
        assert isinstance(fld, Field)

    if isinstance(fld, ExtraColumns):
        fld = fld.base_type

    if isinstance(val, tuple):
        val, sort_by = val
        rval = WithAttrs(val, sorttable_customkey=str(sort_by))
    else:
        rval = val

    if fld.tp:
        assert isinstance(val, fld.tp) or (fld.allow_none and val is None)

    return rval


class Row:
    def __init__(self, table: 'Table', target: Dict[str, Any]) -> None:
        self.__dict__['_target__'] = target
        self.__dict__['_table__'] = weakref.ref(table)

    def __setattr__(self, name: str, val: Any) -> None:
        table = self._table__()
        assert table
        self._target__[name] = prepare_field(table, name, val)

    def __getattr__(self, name) -> Any:
        table = self._table__()
        assert table
        fld: Union[Field, ExtraColumns] = getattr(table, name)
        assert isinstance(fld, ExtraColumns)

        class Extra:
            def __init__(self, table: 'Table', target: Dict[str, Any]) -> None:
                self.table = weakref.ref(table)
                self.target = target

            def __setitem__(self, key: str, val: Any):
                table = self.table()
                assert table
                self.target[key] = prepare_field(table, name, val, must_fld=False)

        return Extra(table, self._target__)


class Table:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.all_names = [name for name, _, _ in self.all_fields()]

    @classmethod
    def all_fields(cls) -> Iterable[Tuple[str, Field, str]]:
        for key, val in cls.__dict__.items():
            if isinstance(val, Field):
                yield (key, val, val.name)
            elif isinstance(val, ExtraColumns):
                yield from ((sname, val.base_type, pname) for sname, pname in val.names.items())

    def __init_subclass__(cls, **kwargs):
        for key, val, ext_name in cls.all_fields():
            if isinstance(val, Field) and ext_name is None:
                val.name = key.replace("_", " ").capitalize()

    def next_row(self) -> Row:
        self.rows.append({})
        return Row(self, self.rows[-1])

    def all_headers(self) -> Tuple[List[str], List[str], Dict[str, Field]]:
        items = list(self.all_fields())
        names = []
        printable_names = {}
        types = {}

        for key, fld, pname in items:
            names.append(key)
            printable_names[key] = pname
            types[key] = fld

        names_set = set(names)

        # find all used keys
        all_keys = set()
        for row in self.rows:
            if not set(row.keys()).issubset(names_set):
                x = 1
            assert set(row.keys()).issubset(names_set), f"{row.keys()} {names_set}"
            all_keys.update(row.keys())

        headers = [name for name, val, _ in items if name in all_keys or not val.skip_if_no_data]
        headers.sort(key=names.index)
        header_names = [printable_names[name] for name in headers]
        return headers, header_names, types

    def add_row(self, *vals: Any):
        self.rows.append(dict(zip(self.all_names, vals)))

    def html(self, id, **kwargs) -> HTMLTable:
        headers, header_names, types = self.all_headers()

        table = HTMLTable(id, headers=header_names, **kwargs)

        for row in self.rows:
            table.next_row()
            for attr_name in headers:
                val: Any = row.get(attr_name)
                field = types[attr_name]

                if isinstance(val, WithAttrs):
                    attrs = val.attrs
                    val = val.val
                else:
                    attrs = {}

                if val is None:
                    table.add_cell(field.null_value, sorttable_customkey=field.null_sort_key, **attrs)
                else:
                    if field.custom_sort and 'sorttable_customkey' not in attrs and not field.dont_sort:
                        table.add_cell(field.converter(val), sorttable_customkey=field.custom_sort(val), **attrs)
                    else:
                        table.add_cell(field.converter(val), **attrs)

        return table
