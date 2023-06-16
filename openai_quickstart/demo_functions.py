"""All the functions used in the openai_functions demo"""
import inspect
import json
import numpy
from typing import Callable





def get_json_representation_using_code(func: Callable) -> dict:
    """
    Returns a JSON representation of the given function.

    Args:
        func (Callable): The function to represent.

    Returns:
        str: The JSON representation of the function.
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)

    # Parse the docstring
    docstring_lines = docstring.split('\n')
    description = docstring_lines[0]
    parameter_descriptions = {}
    args_index = -1
    for line in docstring_lines[1:]:
        if line.strip().startswith("Args:"):
            args_index = docstring_lines.index(line)
            break
    if args_index != -1:
        for line in docstring_lines[args_index+1:]:
            if line.strip().startswith(tuple([f"{param}" for param in signature.parameters.keys()])):
                param, desc = line.strip().split(":", 1)
                param = param.split(" (")[0]  # Remove type hinting if present
                parameter_descriptions[param] = desc.strip()

    # Map Python types to JSON types
    type_mapping = {str: 'string', bool: 'boolean'}

    # Build the JSON representation
    json_representation = {
        'name': func.__name__,
        'description': description,
        'parameters': {
            'type': 'object',
            'properties': {
                name: {
                    'type': type_mapping.get(param.annotation, 'any') if not numpy.issubdtype(param.annotation, numpy.number) else 'number',
                    'description': parameter_descriptions.get(name, '')
                }
                for name, param in signature.parameters.items()
            },
            'required': [name for name, param in signature.parameters.items() if param.default is param.empty]
        }
    }

    return json_representation


