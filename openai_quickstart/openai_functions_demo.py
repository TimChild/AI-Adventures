# Main imports
import openai
import json  # Converting between string and dict

# Other imports that will be explained as they come up
import os
import inspect
import importlib.util
from typing import Callable


FUNCTIONS_FOLDER = "functions"
os.makedirs(FUNCTIONS_FOLDER, exist_ok=True)


def sanitize_python_code(code: str) -> str:
    """If the code is surrounded by markdown triple backticks, they are removed."""
    lines = code.split("\n")
    filtered_lines = [line for line in lines if not line.startswith("```")]
    filtered_code = "\n".join(filtered_lines)
    return filtered_code


def filepath(folder: str, function_name: str) -> str:
    return os.path.join(folder, f"{function_name}.py")


def write_to_py_file(folder: str, function_name: str, file_contents: str):
    """Write code to .py file"""
    # Save function to .py file
    file_contents = sanitize_python_code(file_contents)
    with open(filepath(folder, function_name), "w") as f:
        f.write(file_contents)


def load_function_from_file(folder: str, function_name: str) -> Callable:
    """
    Loads a function from a .py file.
    """
    # Load the spec of the module
    spec = importlib.util.spec_from_file_location(
        "temporary", filepath(folder, function_name)
    )

    # Create a module from the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module to get the function
    spec.loader.exec_module(module)

    # Get the function from the module
    function = getattr(module, function_name)

    return function


SYSTEM_PROMPT_JSON_REP = '''Your job is to convert a python function into a json representation with a specific form.
For example, given this function:
```
def get_weather_report(day_of_week: int, weather_type: str, temperature: float = 10.0) -> str:
    """
    Converts information about weather into a string representation.

    Args:
        day_of_week (int): The day of the week from 0 to 6.
        weather_type (str): The type of weather, can be "sunny", "rainy", or "windy".
        temperature (float, optional): Temperature in Celsius. Defaults to 10.0.

    Returns:
        str: A string representation of the weather report.
    """
    return f'For the {day_of_week}th day of the week, the weather is predicted to be {weather_type} with a max temperature of {temperature}'
```

You should return:
```
{
    "name": "get_weather_report",
    "description": "Converts information about weather into a string representation",
    "parameters": {
        "type": "object",
        "properties": {
            "day_of_week": {
                "type": "number",
                "description": "The day of the week from 0 to 6",
            },
            "weather_type": {"type": "string", "enum": ["sunny", "rainy", "windy"]},
            "temperature": {"type": "number", "description": "Temperature in Celsius. Defaults to 10.0."},
        },
        "required": ["day_of_week", "weather_type"],
    },
}
```
Return the JSON ONLY.'''


def get_json_representation(func: Callable) -> dict:
    """
    Uses the openai.ChatCompletion.create endpoint to return the JSON representation of a function (for GPT to use)

    Args:
        function_str (str): The function to generate a JSON representation for

    Returns:
        dict: The JSON representation of func
    """
    # Get the code of the given func
    function_code = inspect.getsource(func)

    # Ask OpenAI to make the conversion to JSON format
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.0,  # Deterministic output (most probable next word every time)
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_JSON_REP},
            {"role": "user", "content": function_code},  # Then give the func code
        ],
    )

    # Extract the assistant's response
    assistant_response = response["choices"][0]["message"]["content"]

    # convert to dict (JSON)
    description_json = json.loads(assistant_response)
    return description_json


SYSTEM_PROMPT_MAKE_NEW_FUNCTION = """You are an expert python coder that will be tasked with generating the code to go in a .py file for a single function given some specific information from the user.
You will be provided:
    - function_name: The name to give the new function
    - arg_descriptions: Descriptions of all the arguments the function should take (if their types are missing, try to infer them)
    - description: What the function should do with the given arguments

When generating the new function you should follow these rules:
    - Include ONLY the text that will be in the python file (e.g. starting with `import ...` unless no imports are necessary in which case, starting with `def ...`)
    - Do NOT include any plain text explanation at the end of the written code
    - Use the latest python programming techniques and best practices
    - Use the latest/best python libraries when appropriate (e.g. if plotting, use `plotly` instead of `matplotlib` because plotly is better library even though matplotlib is better known)
    - Always include a google style docstring
    - Include type hints for the inputs and output
    - Include all necessary imports (including `typing` ones e.g. List)
"""

def make_new_function(
    function_name: str, arg_descriptions: str, description: str
) -> Callable:
    """
    Use this if an existing function doesn't exist, and it would be helpful to have a new function to complete a task.
    The new function will be made to carry out the task described in the `description` given the arguments described by `arg_descriptions`

    Args:
        function_name: Name to give the new function (should follow python naming conventions)
        arg_descriptions: A description of any arguments that the function should take (including type, and default value if appropriate)
        description: A description of what the function should do (including the what it should output)
    """

    # Format the message we'll send to GPT in the form we have described in the system prompt
    formatted_input = f"function_name: {function_name}\narg_descriptions: {arg_descriptions}\ndescription: {description}"

    # Ask OpenAI to make the new function
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MAKE_NEW_FUNCTION},
            {"role": "user", "content": formatted_input},
        ],
    )

    # Extract the assistant's response
    function_code = response["choices"][0]["message"]["content"]

    # Tidy it, and convert to an actual function
    function_code = sanitize_python_code(function_code)
    write_to_py_file(FUNCTIONS_FOLDER, function_name, function_code)
    func = load_function_from_file(FUNCTIONS_FOLDER, function_name)
    return func

SYSTEM_PROMPT_GENERAL = """You are a helpful AI assistant. 
When responding you should follow these rules:
 - You should always use functions as an intermediate step to respond to the user when appropriate (e.g. when being asked to do math, use a function to do the calculation)
 - You should ONLY consider using the functions provided (e.g. do not assume you can use `python`, that does NOT exist)
 - If there is a missing function that would be useful, make a call to `make_new_function` to create it BEFORE responding to the user
"""


def print_colored_text(text, color):
    """
    Print colored text in JupyterLab output.
    Available colors: 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
    """
    color_code = {
        "red": "\x1b[31m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "white": "\x1b[37m",
    }
    reset_code = "\x1b[0m"

    if color not in color_code:
        raise ValueError(
            f"Invalid color. Available colors: {', '.join(color_code.keys())}"
        )

    colored_text = f"{color_code[color]}{text}{reset_code}"
    print(colored_text)


def to_descriptions_list(functions_dict: dict) -> list[dict]:
    """List out the 'description' part of the available functions dict for GPT to use"""
    return [entry["description"] for entry in functions_dict.values()]

    
global_available_functions = {}
MAKE_NEW_FUNCTION_DESCRIPTION = get_json_representation(make_new_function)
def init_available_functions(clean_slate=False):
    global global_available_functions
    if clean_slate is False and global_available_functions:
        return global_available_functions
    else:
        global_available_functions = {
            "make_new_function": {
                "func": make_new_function,
                "description": MAKE_NEW_FUNCTION_DESCRIPTION,
            }
        }
        return global_available_functions


def get_response(messages: list[dict], functions: list[dict] = None) -> dict:
    function_call = "auto" if functions is not None else "none"
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.0,
        messages=messages,
        functions=to_descriptions_list(functions),
        function_call=function_call,
    )


def is_function_call(response: dict) -> bool:
    return response["choices"][0]["finish_reason"] == "function_call"


def get_message(response: dict) -> dict:
    return response["choices"][0]["message"]


def ask_autonomous_gpt(
    question: str, clean_slate=True, verbose=True, max_steps=5
) -> str:
    """
    Ask chatGPT a question, and it will decide to make new functions when necessary to answer your question.
    Args:
        question: The overall question to answer
        clean_slate: Should it remember previously written functions, or start from a clean slate?
        verbose: Have it print messages as it goes (otherwise only returns the final answer)
        max_steps: Max number of times to loop (to prevent getting stuck in a loop and using all your credit!)
    """
    available_functions = init_available_functions(clean_slate=clean_slate)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERAL},
        {"role": "user", "content": question},
    ]

    if verbose:
        print_colored_text(f"I've been asked to answer:", "green")
        print_colored_text(question, "white")

    step = 0
    while step <= max_steps:
        step += 1
        response = get_response(messages, available_functions)
        message = get_message(response)
        if is_function_call(response):
            func_call = message["function_call"]
            func_name = func_call["name"]
            func_args = json.loads(func_call["arguments"])
            if func_name == "make_new_function":
                new_func_name = func_args["function_name"]
                if verbose:
                    print_colored_text(
                        f"Hmm, I think I'll first make a function called `{new_func_name}` to help out with this",
                        "green",
                    )
                new_func = make_new_function(**func_args)
                if verbose:
                    print_colored_text(
                        f"I made this function:",
                        "green",
                    )
                    print_colored_text(
                        inspect.getsource(new_func),
                        "magenta",
                    )

                available_functions[new_func_name] = {
                    "func": new_func,
                    "description": get_json_representation(new_func),
                }
                # Note: don't add anything to messages on purpose here (we just pretend the function already existed next time around)
                continue
            else:
                if verbose:
                    print_colored_text(
                        f"I think I should use the `{new_func_name}` function with these arguments:\n{func_args}",
                        "green",
                    )
                function_response = available_functions[func_name]["func"](**func_args)
                if verbose:
                    print_colored_text(
                        f"{func_name} returned: {str(function_response)}", "blue"
                    )
                    print_colored_text(
                        f"I'll keep that in mind as I move forward", "green"
                    )
                function_response_message = {
                    "role": "function",
                    "name": func_name,
                    "content": str(function_response),
                }
                messages.append(message)
                messages.append(function_response_message)
                continue
        else:
            answer = message["content"]
            if verbose:
                if step <= 1:
                    print_colored_text(
                        f"I think I can just answer this question directly:", "green"
                    )
                else:
                    print_colored_text(
                        f"I think I am ready to answer, here goes:", "green"
                    )
                print_colored_text(answer, "cyan")
            return answer
    if verbose:
        print_colored_text(
            f"I have failed to get to the answer soon enough :( You can try increasing max_steps from {max_steps} to a higher number, but the problem may just be too hard for me...",
            "green",
        )
    raise RuntimeError(f"Failed to reach a final answer within {max_steps} steps")