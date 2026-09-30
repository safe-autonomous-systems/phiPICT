# Copyright 2026 Jannis Becktepe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A parser for OpenFOAM dictionary files.

The file is turned into nested Python values: a dictionary becomes a
:class:`FoamDict` (entries in file order), a list ``( ... )`` a Python list, and
each token a ``str`` or a number. The items of an entry up to its ``;`` form a
list, e.g. ``convertToMeters 0.1;`` gives ``[0.1]``; a sub-dictionary entry
``name { ... }`` gives the dictionary itself.

``$name`` macros are expanded from the entries read so far (also ``$a.b`` for an
entry of a sub-dictionary). Code and include directives (``#calc``,
``#codeStream``, ``#include``, ...) are not evaluated: they raise
:class:`FoamParseError`.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["FoamDict", "FoamParseError", "parse_foam", "tokenize"]


class FoamParseError(ValueError):
    """The file cannot be parsed or uses an unsupported feature."""


class FoamDict(dict[str, Any]):
    """An OpenFOAM dictionary; entries keep their file order."""


_TOKEN = re.compile(
    r"""
    (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<punct>[(){};\[\]])
  | (?P<word>[^\s(){};"\[\]]+)
    """,
    re.VERBOSE,
)


def _strip_comments(text: str) -> str:
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise FoamParseError("Unterminated /* comment.")
            out.append(" ")
            i = j + 2
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _number(tok: str) -> Any:
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def tokenize(text: str) -> list[Any]:
    """Tokens of a dictionary text, comments removed and numbers converted."""
    tokens: list[Any] = []
    for m in _TOKEN.finditer(_strip_comments(text)):
        if m.group("string") is not None:
            tokens.append(m.group("string"))
        elif m.group("punct") is not None:
            tokens.append(m.group("punct"))
        else:
            tokens.append(_number(m.group("word")))
    return tokens


_DIRECTIVES = {
    "#calc",
    "#codeStream",
    "#include",
    "#includeEtc",
    "#includeFunc",
    "#eval",
    "#includeIfPresent",
    "#neg",
}


class _Parser:
    def __init__(self, tokens: list[Any]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.scopes: list[FoamDict] = []

    def peek(self) -> Any:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> Any:
        if self.pos >= len(self.tokens):
            raise FoamParseError("Unexpected end of file.")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tok: str) -> None:
        got = self.next()
        if got != tok:
            raise FoamParseError(f"Expected {tok!r}, got {got!r}.")

    def lookup(self, name: str) -> Any:
        path = name.lstrip(":").split(".")
        for scope in reversed(self.scopes):
            value: Any = scope
            for part in path:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    break
            else:
                return value
        raise FoamParseError(f"Unknown macro ${name}.")

    def item(self) -> list[Any]:
        """One value item; macros may expand to several."""
        tok = self.next()
        if tok == "(":
            return [self.list_items()]
        if tok == "{":
            return [self.dictionary(closing="}")]
        if isinstance(tok, str) and tok.startswith("#"):
            raise FoamParseError(
                f"The directive {tok} is not supported; expand it in the file first."
            )
        if isinstance(tok, str) and tok.startswith("$"):
            name = tok[1:].strip("{}")
            value = self.lookup(name)
            if isinstance(value, list) and not isinstance(value, FoamDict):
                return list(value)
            return [value]
        if isinstance(tok, str) and tok.startswith("-$"):
            value = self.lookup(tok[2:].strip("{}"))
            if (
                isinstance(value, list)
                and len(value) == 1
                and isinstance(value[0], int | float)
            ):
                return [-value[0]]
            raise FoamParseError(f"Cannot negate the macro {tok}.")
        if tok in (")", "}", ";"):
            raise FoamParseError(f"Unexpected {tok!r}.")
        return [tok]

    def list_items(self) -> list[Any]:
        items: list[Any] = []
        while self.peek() != ")":
            if self.peek() is None:
                raise FoamParseError("Unterminated list.")
            items += self.item()
        self.next()
        return items

    def dictionary(self, closing: str | None) -> FoamDict:
        d = FoamDict()
        self.scopes.append(d)
        while True:
            tok = self.peek()
            if tok is None:
                if closing is not None:
                    raise FoamParseError("Unterminated dictionary.")
                break
            if tok == closing:
                self.next()
                break
            key = self.next()
            if isinstance(key, str) and key in _DIRECTIVES:
                raise FoamParseError(
                    f"The directive {key} is not supported; expand it in the file "
                    "first."
                )
            if not isinstance(key, str):
                raise FoamParseError(f"Expected a keyword, got {key!r}.")
            if key.startswith("$"):  # dictionary merge "$name;"
                value = self.lookup(key[1:])
                if isinstance(value, dict):
                    d.update(value)
                if self.peek() == ";":
                    self.next()
                continue
            if self.peek() == "{":
                self.next()
                d[key] = self.dictionary(closing="}")
                continue
            values: list[Any] = []
            while self.peek() != ";":
                if self.peek() is None or self.peek() == closing:
                    raise FoamParseError(f"Entry {key!r} is not terminated by ';'.")
                values += self.item()
            self.next()
            d[key] = values
        self.scopes.pop()
        return d


def parse_foam(text: str) -> FoamDict:
    """Parse the text of an OpenFOAM dictionary file.

    Parameters
    ----------
    text : str
        Contents of the file.

    Returns
    -------
    FoamDict
        The top-level dictionary.

    Raises
    ------
    FoamParseError
        If the text is malformed or uses directives.
    """
    return _Parser(tokenize(text)).dictionary(closing=None)
