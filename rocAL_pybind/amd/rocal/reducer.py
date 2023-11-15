import inspect
import marshal
import os
import pickle
import types

def set_funcion_state(fun, state):
    fun.__globals__.update(state['global_refs'])
    fun.__defaults__ = state['defaults']
    fun.__kwdefaults__ = state['kwdefaults']


def function_unpickle(name, qualname, code, closure):
    code = marshal.loads(code)
    global_scope = {'__builtins__': __builtins__}
    fun = types.FunctionType(code, global_scope, name, closure=closure)
    fun.__qualname__ = qualname
    return fun

def get_global_references_from_nested_code(code, global_scope, global_refs):
    for constant in code.co_consts:
        if inspect.iscode(constant):
            closure = tuple(types.CellType(None) for _ in range(len(constant.co_freevars)))
            dummy_function = types.FunctionType(constant, global_scope, 'dummy_function',
                                                closure=closure)
            global_refs.update(inspect.getclosurevars(dummy_function).globals)
            get_global_references_from_nested_code(constant, global_scope, global_refs)

def function_by_value_reducer(fun):
    cl_vars = inspect.getclosurevars(fun)
    code = marshal.dumps(fun.__code__)
    basic_def = (fun.__name__, fun.__qualname__, code, fun.__closure__)
    global_refs = dict(cl_vars.globals)
    get_global_references_from_nested_code(fun.__code__, fun.__globals__, global_refs)
    fun_context = {
        'global_refs': global_refs,
        'defaults': fun.__defaults__,
        'kwdefaults': fun.__kwdefaults__
    }
    return function_unpickle, basic_def, fun_context, None, None, set_funcion_state