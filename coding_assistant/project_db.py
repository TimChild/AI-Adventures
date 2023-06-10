"""
Functions related to:
 - making a vector database that contains context about the current project (i.e. Code and descriptions).
 - modifying the vector database
 - querying the vector database
"""

import os
import ast
from typing import List

def extract_python_functions(filename: str) -> list[str]:
    """
    Extract Python function definitions from a source file.

    This function uses Python's Abstract Syntax Trees (ast) module to parse
    the source code and extract all function definitions.

    Args:
        filename (str): The path to the Python source file from which to
            extract function definitions.

    Returns:
        list[str]: A list of strings, where each string is the source code
            of a function definition.

    Examples:
        >>> extract_python_functions("sample.py")
        ["def func1():\n    pass", "def func2(arg):\n    return arg * 2"]
    """
    with open(filename, "r") as source:
        tree = ast.parse(source.read())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    function_source = [ast.unparse(func) for func in functions]
    return function_source


def generate_function_descriptions(functions: List[str]) -> List[str]:
    """
    Generate a basic description of each function in a list of Python function strings.

    Each description includes the function name and its arguments.

    Args:
        functions (List[str]): A list of strings, where each string is a source code of a function.

    Returns:
        List[str]: A list of strings, where each string is a description of a function.

    Examples:
        >>> generate_function_descriptions(["def func1():\n    pass", "def func2(arg):\n    return arg * 2"])
        ['Function name: func1, Arguments: []', 'Function name: func2, Arguments: ['arg']']
    """
    descriptions = []
    for function in functions:
        tree = ast.parse(function)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                args = [arg.arg for arg in node.args.args]
                description = f"Function name: {func_name}, Arguments: {args}"
                descriptions.append(description)
    return descriptions


def save_functions_and_descriptions(functions: List[str], code_dir: str, desc_dir: str) -> None:
    """
    Save Python function strings and their descriptions in structured directories.

    Each function is saved in a .py file in the 'code_dir' directory, and its description is saved
    in a .txt file in the 'desc_dir' directory.

    Args:
        functions (List[str]): A list of strings, where each string is a source code of a function.
        code_dir (str): The directory in which to save the .py files.
        desc_dir (str): The directory in which to save the .txt files.

    Returns:
        None

    Examples:
        >>> save_functions_and_descriptions(["def func1():\n    pass", "def func2(arg):\n    return arg * 2"], 'code_dir', 'desc_dir')
    """
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
    if not os.path.exists(desc_dir):
        os.makedirs(desc_dir)

    for function in functions:
        tree = ast.parse(function)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_dir = os.path.join(code_dir, node.name)
                class_desc_dir = os.path.join(desc_dir, node.name)
                os.makedirs(class_dir, exist_ok=True)
                os.makedirs(class_desc_dir, exist_ok=True)
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        method_name = sub_node.name
                        args = [arg.arg for arg in sub_node.args.args]
                        description = f"Function name: {method_name}, Arguments: {args}"

                        with open(os.path.join(class_dir, f"{method_name}.py"), 'w') as f:
                            f.write(ast.unparse(sub_node))
                        with open(os.path.join(class_desc_dir, f"{method_name}.txt"), 'w') as f:
                            f.write(description)

            elif isinstance(node, ast.FunctionDef):
                func_name = node.name
                args = [arg.arg for arg in node.args.args]
                description = f"Function name: {func_name}, Arguments: {args}"

                with open(os.path.join(code_dir, f"{func_name}.py"), 'w') as f:
                    f.write(ast.unparse(node))
                with open(os.path.join(desc_dir, f"{func_name}.txt"), 'w') as f:
                    f.write(description)

