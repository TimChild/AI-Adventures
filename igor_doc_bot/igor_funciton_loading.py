from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import json
from IPython.display import Markdown, display
from langchain.schema import Document


def extract_functions(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    functions = []
    function_data = None
    docstring = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.startswith('function'):
            function_data, docstring = start_new_function(i, line, stripped_line=stripped)
        elif function_data and stripped.startswith('//'):
            docstring = accumulate_docstring(stripped, docstring)
            function_data.code.append(line)
        elif stripped == 'end':
            functions.append(end_current_function(function_data, docstring, i, filename))
            function_data, docstring = None, None
        elif function_data:  # We are inside a function block
            function_data.code.append(line)
        else:
            print(f'Ignoring: {line}')

    return functions


def start_new_function(line_number, line, stripped_line):
    function_name = stripped_line.split(' ')[1].split('(')[0]
    function_data = Function(name=function_name, start_line=line_number, code=[line], docstring=[], end_line=None, filename=None)
    return function_data, function_data.docstring


def accumulate_docstring(stripped_line, docstring):
    docstring.append(stripped_line[2:].strip())
    return docstring


def end_current_function(function_data, docstring, end_line, filename):
    function_data.end_line = end_line
    function_data.filename = filename
    function_data.docstring = ' '.join(docstring)
    function_data.code = ''.join(function_data.code)
    return function_data

@dataclass
class IgorObject:
    name: str
    declaration: str
    start_line: int
    end_line: int
    filename: str
    docstring: str
    code: str

    @property
    def metadata(self):
        data = asdict(self)
        data.pop('code')
        return data

    def to_markdown(self, include_code: bool = True) -> str:
        summary = f'## Summary\n\n'
        for key, value in self.metadata.items():
            if key != 'docstring':
                summary += f'- **{key.capitalize()}**: {value}\n'
        
        docstring = f'## Docstring\n\n```\n{self.docstring}\n```\n'
        
        if include_code:
            code = f'## Code\n\n```igor\n{self.code}\n```\n'
            md = summary + docstring + code
        else:
            md = summary + docstring
        return md

    def to_document(self) -> Document:
        return Document(page_content=self.code, metadata=self.metadata)

    @classmethod
    def from_document(cls, document: Document) -> IgorObject:
        igor_object = cls(code=document.page_content, **document.metadata)
        return igor_object
    
    def to_json(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_json(cls, json_dict: dict | str) -> IgorObject:
        if isinstance(json_dict, str):
            with open(json_dict, 'r') as f:
                json_dict = json.load(f)
        return cls(**json_dict)


Function = Structure = Macro = IgorObject


@dataclass
class IgorFile:
    filename: str
    preamble: str
    functions: List[Function]
    structures: List[Structure]
    macros: List[Macro]
        
    def to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        # Convert dicts back to dataclass instances
        data['functions'] = [Function.from_json(func) for func in data['functions']]
        data['structures'] = [Structure.from_json(struct) for struct in data['structures']]
        data['macros'] = [Macro.from_json(macro) for macro in data['macros']]
        return cls(**data)


def parse_igor_file(filename, verbose=False):
    with open(filename, 'r') as f:
        lines = f.readlines()

    preamble = []
    functions = []
    structures = []
    macros = []
    igor_object_data = None
    igor_object_list = None
    docstring = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.lower().startswith(('function', 'structure', 'window')):
            igor_object_data, object_type = start_new_igor_object(i, line, stripped)
            if verbose:
                print(f'Started adding {igor_object_data.name}')
            if object_type == 'function':
                igor_object_list = functions
            elif object_type == 'structure':
                igor_object_list = structures
            elif object_type == 'macro':
                igor_object_list = macros
        elif igor_object_data and stripped.startswith('//') and (len(igor_object_data.code) == len(igor_object_data.docstring)):
            accumulate_docstring(igor_object_data, line)
        elif stripped.lower() in ('end', 'endstructure', 'endmacro'):
            if verbose:
                print(f'Finished adding {igor_object_data.name}')
            igor_object_list.append(end_current_igor_object(igor_object_data, i, filename))
            igor_object_data, igor_object_list, docstring =  None, None, None
        elif igor_object_data:  # We are inside an Igor object block
            igor_object_data.code.append(line)
        elif all([len(l) == 0 for l in [functions, structures, macros]]):
            preamble.append(line)
        else:
            if verbose:
                if line:
                    print(f'Ignoring line: {line}')

    return IgorFile(filename=filename, preamble=''.join(preamble), functions=functions, structures=structures, macros=macros)


def start_new_igor_object(line_number, line, stripped_line):
    name = stripped_line.split(' ')[1].split('(')[0]
    declaration = ''.join(stripped_line.split(' ')[1:])
    stripped_line = stripped_line.lower()
    if stripped_line.startswith('function'):
        NewObject = Function
        obj_type = 'function'
    elif stripped_line.startswith('structure'):
        NewObject = Structure
        obj_type = 'structure'
    elif stripped_line.startswith('window'):
        NewObject = Macro
        obj_type = 'macro'
    else:
        raise RuntimeError(f"Don't know how to handle: {stripped_line}")

    igor_object_data = NewObject(name=name, declaration=declaration, start_line=line_number, code=[], docstring=[], end_line=None, filename=None)
    return igor_object_data, obj_type


def accumulate_docstring(igor_object, line):
    igor_object.docstring.append(line.strip()[2:].strip())
    igor_object.code.append(line)


def end_current_igor_object(igor_object_data, end_line, filename):
    igor_object_data.end_line = end_line
    igor_object_data.filename = filename
    if igor_object_data.docstring:
        igor_object_data.docstring = '\n'.join(igor_object_data.docstring)
    if igor_object_data.code:
        igor_object_data.code = ''.join(igor_object_data.code)
    return igor_object_data