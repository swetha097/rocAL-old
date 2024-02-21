import inspect
import textwrap
import re

def generate_import_statements(function_globals):
    import_statements = []

    for alias, module in function_globals.items():
        if hasattr(module, '__file__'):
            # Check if the module has a file attribute (i.e., it's not a built-in module)
            module_path = module.__file__
            import_statements.append(f"import {module.__name__} as {alias}  # {module_path}")
    return import_statements

def write_import_statements_to_file(import_statements, output_file):
    with open(output_file, 'w') as file:
        for import_statement in import_statements:
            file.write(import_statement + '\n')

def dump_python_source_print_outputs_to_console(func, size):
    closure_vars = inspect.getclosurevars(func)
    function_globals = dict(closure_vars.globals)
    import_statements = generate_import_statements(function_globals)
    source_code = inspect.getsource(func)
    source_code = textwrap.dedent(source_code)
    source_code = re.sub(r'\bself\b,?', '', source_code)
    write_import_statements_to_file(import_statements, "temp_eso_script.py" )
    with open('temp_eso_script.py', 'a') as file:
        file.write(source_code)

    # Add a __main__ function to the file
    with open('temp_eso_script.py', 'a') as file:
        # file.write(f'\n\nif __name__ == "__main__":\n')
        file.write(f'\n\nresult = {func.__name__}({size})')
        file.write(f'\nfor item in result:')
        file.write(f'\n    print(item)')
