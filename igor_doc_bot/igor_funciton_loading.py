from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import json
from tqdm.auto import tqdm
from IPython.display import Markdown, display
from langchain.schema import Document


# def extract_functions(filename: str) -> List[Function]:
#     """
#     Extracts function data from a file with given filename.

#     Args:
#         filename (str): The name of the file to extract functions from.

#     Returns:
#         list[Function]: List of `Function` dataclass instances representing functions in the file.
#     """
#     with open(filename, 'r') as f:
#         lines = f.readlines()

#     functions = []
#     function_data = None
#     docstring = None

#     for i, line in enumerate(lines, start=1):
#         stripped = line.strip()

#         if stripped.startswith('function'):
#             function_data, docstring = start_new_function(i, line, stripped_line=stripped)
#         elif function_data and stripped.startswith('//'):
#             docstring = accumulate_docstring(stripped, docstring)
#             function_data.code.append(line)
#         elif stripped == 'end':
#             functions.append(end_current_function(function_data, docstring, i, filename))
#             function_data, docstring = None, None
#         elif function_data:  # We are inside a function block
#             function_data.code.append(line)
#         else:
#             print(f'Ignoring: {line}')

#     return functions


# def start_new_function(line_number: int, line: str, stripped_line: str) -> Tuple[Function, List[str]]:
#     """
#     Starts processing a new function from the file.

#     Args:
#         line_number (int): The line number in the file where the function starts.
#         line (str): The line of text in the file where the function starts.
#         stripped_line (str): The line of text with leading/trailing whitespace removed.

#     Returns:
#         tuple: A tuple containing a `Function` dataclass instance representing the new function,
#                and a list to accumulate the function's docstring.
#     """
#     return function_data, function_data.docstring


# def accumulate_docstring(stripped_line: str, docstring: List[str]) -> List[str]:
#     """
#     Adds a line of text to a function's docstring.

#     Args:
#         stripped_line (str): The line of text to add to the docstring, with leading/trailing whitespace removed.
#         docstring (List[str]): The list accumulating the function's docstring.

#     Returns:
#         List[str]: The updated docstring list.
#     """
#     docstring.append(stripped_line[2:].strip())
#     return docstring

# def end_current_function(function_data: Function, docstring: List[str], end_line: int, filename: str) -> Function:
#     """
#     Finalizes processing of a function.

#     Args:
#         function_data (Function): The `Function` dataclass instance representing the current function.
#         docstring (List[str]): The list containing the function's docstring.
#         end_line (int): The line number in the file where the function ends.
#         filename (str): The name of the file.

#     Returns:
#         Function: The updated `Function` dataclass instance.
#     """
#     function_data.end_line = end_line
#     function_data.filename = filename
#     function_data.docstring = ' '.join(docstring)
#     function_data.code = ''.join(function_data.code)
#     return function_data


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
        data.pop("code")
        return data

    def to_markdown(self, include_code: bool = True) -> str:
        """
        Converts the IgorObject to a markdown format.

        Args:
            include_code (bool, optional): If True, include the object's code in the output. Defaults to True.

        Returns:
            str: The markdown string representation of the IgorObject.
        """
        summary = f"## Summary\n\n"
        for key, value in self.metadata.items():
            if key != "docstring":
                summary += f"- **{key.capitalize()}**: {value}\n"

        docstring = f"## Docstring\n\n```\n{self.docstring}\n```\n"

        if include_code:
            code = f"## Code\n\n```igor\n{self.code}\n```\n"
            md = summary + docstring + code
        else:
            md = summary + docstring
        return md

    def to_document(self) -> Document:
        """
        Converts the IgorObject to a Document.

        Returns:
            Document: The `Document` dataclass instance representing the IgorObject.
        """
        return Document(page_content=self.code, metadata=self.metadata)

    @classmethod
    def from_document(cls, document: Document) -> IgorObject:
        """
        Constructs an IgorObject from a Document.

        Args:
            document (Document): The `Document` dataclass instance to convert.

        Returns:
            IgorObject: The `IgorObject` dataclass instance representing the document.
        """
        igor_object = cls(code=document.page_content, **document.metadata)
        return igor_object

    def to_json(self) -> dict:
        """
        Converts the IgorObject to a JSON-compatible dictionary.

        Returns:
            dict: The dictionary representing the IgorObject.
        """
        return asdict(self)

    @classmethod
    def from_json(cls, json_dict: dict | str) -> IgorObject:
        """
        Constructs an IgorObject from a JSON-compatible dictionary or a JSON file.

        Args:
            json_dict (dict | str): The dictionary or filename of JSON file to convert.

        Returns:
            IgorObject: The `IgorObject` dataclass instance representing the dictionary or file.
        """
        if isinstance(json_dict, str):
            with open(json_dict, "r") as f:
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

    def to_file(self, filename: str) -> None:
        """
        Writes the IgorFile to a JSON file.

        Args:
            filename (str): The name of the file to write to.
        """
        with open(filename, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_file(cls, filename: str) -> IgorFile:
        """
        Constructs an IgorFile from a JSON file.

        Args:
            filename (str): The name of the file to convert.

        Returns:
            IgorFile: The `IgorFile` dataclass instance representing the file.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        # Convert dicts back to dataclass instances
        data["functions"] = [Function.from_json(func) for func in data["functions"]]
        data["structures"] = [
            Structure.from_json(struct) for struct in data["structures"]
        ]
        data["macros"] = [Macro.from_json(macro) for macro in data["macros"]]
        return cls(**data)


def start_new_igor_object(
    line_number: int, line: str, stripped_line: str
) -> tuple[IgorObject, str]:
    """
    Starts processing a new Igor object from the file.

    Args:
        line_number (int): The line number in the file where the object starts.
        line (str): The line of text in the file where the object starts.
        stripped_line (str): The line of text with leading/trailing whitespace removed.

    Returns:
        tuple: A tuple containing an `IgorObject` dataclass instance representing the new object,
               and a string indicating the type of the object (either 'function', 'structure', or 'macro').
    """
    name = stripped_line.split(" ")[1].split("(")[0]
    declaration = "".join(stripped_line.split(" ")[1:])
    stripped_line = stripped_line.lower()
    if stripped_line.startswith("function"):
        NewObject = Function
        obj_type = "function"
    elif stripped_line.startswith("structure"):
        NewObject = Structure
        obj_type = "structure"
    elif stripped_line.startswith("window"):
        NewObject = Macro
        obj_type = "macro"
    else:
        raise RuntimeError(f"Don't know how to handle: {stripped_line}")

    igor_object_data = NewObject(
        name=name,
        declaration=declaration,
        start_line=line_number,
        code=[],
        docstring=[],
        end_line=None,
        filename=None,
    )
    return igor_object_data, obj_type


def accumulate_docstring(igor_object: IgorObject, line: str) -> None:
    """
    Adds a line of text to an Igor object's docstring.

    Args:
        igor_object (IgorObject): The `IgorObject` dataclass instance representing the current object.
        line (str): The line of text to add to the docstring.
    """
    igor_object.docstring.append(line.strip()[2:].strip())
    igor_object.code.append(line)


def end_current_igor_object(
    igor_object_data: IgorObject, end_line: int, filename: str
) -> IgorObject:
    """
    Finalizes processing of an Igor object.

    Args:
        igor_object_data (IgorObject): The `IgorObject` dataclass instance representing the current object.
        end_line (int): The line number in the file where the object ends.
        filename (str): The name of the file.

    Returns:
        IgorObject: The updated `IgorObject` dataclass instance.
    """
    igor_object_data.end_line = end_line
    igor_object_data.filename = filename
    if igor_object_data.docstring:
        igor_object_data.docstring = "\n".join(igor_object_data.docstring)
    if igor_object_data.code:
        igor_object_data.code = "".join(igor_object_data.code)
    return igor_object_data


def parse_igor_file(filename: str, verbose: bool = False) -> IgorFile:
    """
    Parses an Igor file to extract its preamble, functions, structures, and macros.

    Args:
        filename (str): The name of the file to parse.
        verbose (bool, optional): If True, print out additional information during the parsing process.
                                   Defaults to False.

    Returns:
        IgorFile: An `IgorFile` dataclass instance representing the parsed file.
    """
    with open(filename, "r") as f:
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

        # In case there is a "threadsafe", remove from the stripped string that is used
        # for identification
        if stripped.lower().startswith("threadsafe"):
            stripped = stripped[10:].strip()

        if stripped.lower().startswith(("function", "structure", "window")):
            igor_object_data, object_type = start_new_igor_object(i, line, stripped)
            if verbose:
                print(f"Started adding {igor_object_data.name}")
            if object_type == "function":
                igor_object_list = functions
            elif object_type == "structure":
                igor_object_list = structures
            elif object_type == "macro":
                igor_object_list = macros
        elif (
            igor_object_data
            and stripped.startswith("//")
            and (len(igor_object_data.code) - 1 == len(igor_object_data.docstring))
        ):
            accumulate_docstring(igor_object_data, line)
        elif stripped.lower() in ("end", "endstructure", "endmacro"):
            if verbose:
                print(f"Finished adding {igor_object_data.name}")
            igor_object_list.append(
                end_current_igor_object(igor_object_data, i, filename)
            )
            igor_object_data, igor_object_list, docstring = None, None, None
        elif igor_object_data:  # We are inside an Igor object block
            igor_object_data.code.append(line)
        elif all([len(l) == 0 for l in [functions, structures, macros]]):
            preamble.append(line)
        else:
            if verbose:
                if line:
                    print(f"Ignoring line: {line}")

    return IgorFile(
        filename=filename,
        preamble="".join(preamble),
        functions=functions,
        structures=structures,
        macros=macros,
    )


def parse_all_ipf_files(
