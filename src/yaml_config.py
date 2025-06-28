import ast
from typing import Any, Iterable

def safe_load(stream: Iterable[str] | str) -> Any:
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = str(stream)

    # remove comments and empty lines
    lines = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if line:
            lines.append(line)

    result: dict[str, Any] = {}
    stack = [result]
    indents = [0]
    for raw in lines:
        indent = len(raw) - len(raw.lstrip())
        key, _, val = raw.partition(":")
        key = key.strip()
        val = val.strip()

        while indent < indents[-1]:
            stack.pop()
            indents.pop()

        if not val:
            new_dict: dict[str, Any] = {}
            stack[-1][key] = new_dict
            stack.append(new_dict)
            indents.append(indent + 2)
        else:
            try:
                value = ast.literal_eval(val)
            except Exception:
                value = val.strip('"\'')
            stack[-1][key] = value
    return result
