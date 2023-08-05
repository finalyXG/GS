
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pathlib
import pandas as pd
import functools
import click
# import pyinputplus as pyip
from pathlib import Path
from IPython.display import display
import inspect
import decorator
from networkx.drawing.nx_pydot import graphviz_layout

CACHED_PATH = ''
EXP_PATH = ''

def set_global_path(cached_path='../staging/tmp_t/cached_img', exp_path='../staging/stg_t/'):
    global CACHED_PATH
    global EXP_PATH
    CACHED_PATH = cached_path
    EXP_PATH = exp_path


def enable_cache_image(func):
    pathlib.Path(CACHED_PATH).mkdir(parents=True, exist_ok=True) 
    def decide_cached_run(*args, **kwargs):
        _use_last_result = False
        if '_use_last_result' in kwargs:
            _use_last_result = kwargs['_use_last_result']
            del kwargs['_use_last_result']

        cache_file = f'{CACHED_PATH}/{func.__name__}_cached_result.png'

        if _use_last_result:
            print(f"using cached image:")
            img = mpimg.imread(cache_file)
            fig, ax = plt.subplots(figsize=(15,3))
            ax.imshow(img)
            ax.axis('off')
            # func(*args, **kwargs)
        else:
            fig = func(*args, **kwargs)
            fig.savefig(f'{CACHED_PATH}/{func.__name__}_cached_result.png', bbox_inches='tight')
    return decide_cached_run


def enable_cache_dataframe(func):
    def decide_cached_run(*args, **kwargs):
        _use_last_result = False
        if '_use_last_result' in kwargs:
            _use_last_result = kwargs['_use_last_result']
            del kwargs['_use_last_result']

        _is_cached = True
        if '_is_cached' in kwargs:
            _is_cached = kwargs['_is_cached']
            del kwargs['_is_cached']


        cache_file = f'{CACHED_PATH}/{func.__name__}_cached_result.pkl'

        if _use_last_result:
            print(f"using cached dataframe:")
            _df = pd.read_pickle(cache_file)
            return _df
            # func(*args, **kwargs)
        else:
            _df = func(*args, **kwargs)
            if _is_cached:
                _df.to_pickle(f'{CACHED_PATH}/{func.__name__}_cached_result.pkl')
            return _df
    return decide_cached_run

import networkx as nx
def create_dep_graph(cls):
    G = nx.DiGraph()
    G.add_nodes_from([(node, {'alias': attr}) for (node, attr) in cls._alias_dict.items() ])
    for node, neighs in cls._require_dict.items():
        if type(neighs) == str:
            neighs = [neighs]
        for n2_alias in neighs:
            if n2_alias in cls._alias2method_name_dict:
                from_node = cls._alias2method_name_dict[n2_alias]
            else:
                from_node = n2_alias
                assert "@" in from_node, f"Currently only support missing node formatted with A@B, now get {from_node}"
                assert len(n2_alias.split("@")) == 2
                _alias = from_node.split("@")[0]
                _method = from_node.split("@")[1]
                if '.' not in _method:
                    _method = f"{_method}.{_method}"
                G.add_node(_method, alias=_alias, from_other_graph=_method)
                from_node = _method
                
            G.add_edge(from_node, node)
    return G


def class_register(cls):
    cls._alias_dict = {}
    cls._alias2method_dict = {}
    cls._alias2method_name_dict = {}
    cls._require_dict = {}
    for methodname in dir(cls):
        method = getattr(cls, methodname)
        if hasattr(method, '_alias'):
            cls._alias_dict.update(
                {cls.__name__ + '.' + methodname: method._alias})
            cls._alias2method_dict.update(
                {method._alias: method})
            cls._alias2method_name_dict.update(
                {method._alias: cls.__name__ + '.' + methodname})
            
        if hasattr(method, '_require'):
            cls._require_dict.update(
                {cls.__name__ + '.' + methodname: method._require}
            )
    cls._dep_graph = create_dep_graph(cls)

    return cls

@class_register
class DependencyController:
    def __init__(self, _exp_name=None, _exp_path=None, **kwargs):
        self._called_dict = {}
        self._exp_name = _exp_name
        self._exp_path = _exp_path

    def _set_exp_name(self, name):
        self._exp_name = name

    def _set_exp_path(self, path):
        self._exp_path = path

    def _display_dep_graph(self, figsize=(16,8), rotate=True):
        G = self._dep_graph
        node_labels = {n: f"[{G.nodes[n]['alias']}] \n {n.split('.')[-1]}" for n in G.nodes}
        font_colors = {n: 'red' for n in G.nodes}
        for n in G.nodes:
            if 'from_other_graph' in G.nodes[n]:
                obj = getattr(self, G.nodes[n]['from_other_graph'].split('.')[0])
                _alias = G.nodes[n]['alias']
                if _alias in obj._called_dict:
                    font_colors[n] = 'green'

        for _called_alias in self._called_dict.keys():
            font_colors[self._alias2method_name_dict[_called_alias]] = 'green'
        font_colors = list(font_colors.values())
        plt.figure(figsize=figsize)
        # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        # pos = hierarchy_pos(G)
        pos = nx.nx_pydot.pydot_layout(G, prog="dot")
        
        if rotate:
            pos = {node: (-y, x) for node, (x,y) in pos.items()} #65124349
        nx.draw(G, pos=pos, with_labels=True, labels=node_labels, node_color=font_colors, node_shape="s", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1')) # 30344592
        plt.margins(x=0.2, y=0.2)

    

    
    def require(name, *args0, **kwargs0):
        def check_and_call_dep_method(ori_obj, dep_name):
            if "@" in dep_name:
                assert len(dep_name.split('@')) == 2, f"Currently only support A@B format in 'require', now got {dep_name}"
                obj_name = dep_name.split("@")[-1]
                obj = getattr(ori_obj, obj_name)
                dep_name = dep_name.split("@")[0]
            else:
                obj = ori_obj
            if dep_name not in obj._called_dict.keys():
                obj._alias2method_dict[dep_name](obj, *args0, **kwargs0)

        def decorator(func):
            func._require = name
            @functools.wraps(func)
            def wrap(self, *args, **kwargs):
                # print(f"inside wrap: {name}")
                if type(name) == str:
                    
                    check_and_call_dep_method(self, name)
                    # if name not in self._called_dict.keys():
                    #     self._alias2method_dict[name](self, *args0, **kwargs0)

                elif type(name) == list:
                    for n in name:
                        assert type(n) == str
                        check_and_call_dep_method(self,n)
                        # if n not in self._called_dict.keys():
                        #     self._alias2method_dict[n](self, *args0, **kwargs0)

                else:
                    raise ValueError(f"Invalid type {type(name)}")

                return func(self, *args, **kwargs)
            return wrap
        return decorator   

    def track(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            # print(self)
            # print(self._called_dict)
            # print(func.__name__)
            self._called_dict[func._alias] = True
            return func(self, *args, **kwargs)
        return wrap 
    
    def alias(name):
        def new_func(func):
            func._alias = name
            return func
        return new_func
    


def pyip_prompt_confirm(confirm_msg='', confirm_default=True, _confirm_msg_lambda_on_begin=None, **kwargs0):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
                if '_is_last_prompt' not in kwargs0 or kwargs0['_is_last_prompt'] == True:
                    del kwargs["_disable_prompt"]
                return function(*args, **kwargs)
            else:
                if _confirm_msg_lambda_on_begin:
                    # _confirm_on_begin is a lambda that construct the confrim message using "self" (ctx)
                    ctx = args[0]
                    _msg = _confirm_msg_lambda_on_begin(ctx)
                    if click.confirm(_msg, default=confirm_default) == True:
                        return function(*args, **kwargs)
                elif click.confirm(confirm_msg, default=confirm_default) == True:
                    return function(*args, **kwargs)
        return wrapper
    return decorator

def pyip_prompt_input(prompt_msg, target_param, _confirm_msg_lambda_on_begin=None, **kwargs0):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
                if '_is_last_prompt' not in kwargs0 or kwargs0['_is_last_prompt'] == True:
                    del kwargs["_disable_prompt"]
                return function(*args, **kwargs)
            else:
                if _confirm_msg_lambda_on_begin:
                    ctx = args[0]
                    _msg = _confirm_msg_lambda_on_begin(ctx)
                else:
                    _msg = prompt_msg

                _input = pyip.inputStr(_msg + '\n')
                kwargs[target_param] = _input
                return function(*args, **kwargs)
        return wrapper
    return decorator

def pyip_prompt_input_num(prompt_msg, target_param):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
                if '_is_last_prompt' not in kwargs or kwargs['_is_last_prompt'] == True:
                    del kwargs["_disable_prompt"]
                return function(*args, **kwargs)
            else:
                _input = pyip.inputNum(prompt_msg + '\n')
                kwargs[target_param] = _input
                return function(*args, **kwargs)
        return wrapper
    return decorator




def pyip_prompt_menu(prompt_msg, target_param, option_name):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
                if '_is_last_prompt' not in kwargs or kwargs['_is_last_prompt'] == True:
                    del kwargs["_disable_prompt"]
                return function(*args, **kwargs)
            else:
                ctx = args[0]
                _options = getattr(ctx, option_name)
                _input = pyip.inputMenu(_options, prompt=prompt_msg+'\n', lettered=False, numbered=True)
                kwargs[target_param] = _input
                return function(*args, **kwargs)
        return wrapper
    return decorator


def experiment_track(name=None, path=None, desc='', pkl_file_name='df_config.pkl', show_df_config=False, 
                     del_file=False, exp_param_ls = [], del_exp_param_ls = [], exp_param_map={},**kwargs0):
    '''
    Use this decorator to help experiment design;
    If combining with @pyip, should use experiment_track first, i.e.,
        @pyip_XXX
        @experiment_track
        f1(...)
    '''

    _df_file = ''
    def _get_df_file(name, path):
        if path == None:
            path = EXP_PATH
        _df_path = Path(path).joinpath(name)
        _df_path.mkdir(parents=True, exist_ok=True)
        _df_file = _df_path.joinpath(pkl_file_name)
        return _df_file

    if name != None:
        # if path == None:
        #     path = EXP_PATH
        # _df_path = Path(path).joinpath(name)
        # _df_path.mkdir(parents=True, exist_ok=True)
        # _df_file = _df_path.joinpath(pkl_file_name)
        _df_file = _get_df_file(name, path)

        if del_file:
            _df_file.unlink(missing_ok=True)
        if _df_file.exists():
            df = pd.read_pickle(_df_file)
        else:
            df = pd.DataFrame({'name':[],'value':[]}).set_index('name')
            
        if 'experiment' not in df.index:
            df.loc['experiment', 'value'] = name
        if 'desc' not in df.index:
            df.loc['desc', 'value'] = desc
        df.to_pickle(_df_file)

        if len(exp_param_ls) > 0:
            for exp_k in exp_param_ls:
                if exp_k in exp_param_map.keys():
                    exp_k_mapped = exp_param_map[exp_k]
                else:
                    exp_k_mapped = exp_k
                if exp_k_mapped not in df.index and exp_k in kwargs0.keys():
                    df.loc[exp_k_mapped,'value'] = kwargs0[exp_k]

                if exp_k in kwargs0.keys():
                    _value0 = df.loc[exp_k_mapped,'value']
                    _value1 = kwargs0[exp_k]
                    if type(_value0) in [list]:
                        _value0 = str(_value0)
                    if type(_value1) in [list]:
                        _value1 = str(_value1) 
                    assert _value0 == _value1, f"Found experiment {exp_k_mapped} changed during the experiment run, which is not expected! (Can use _show_df_config to view more details.) {df.loc[exp_k_mapped,'value']} --> {kwargs0[exp_k]}"
            df.to_pickle(_df_file)

        if len(del_exp_param_ls) > 0:
            for del_k in del_exp_param_ls:
                if del_k in df.index:
                    df = df.drop(index=del_k)
            df.to_pickle(_df_file)

        if show_df_config:
            display(df)

    def _check_and_set_df_exp_config_row(_df, k, v, allow_k_ls):
        if k in allow_k_ls:
            assert (k not in _df.index) or (_df.loc[k,'value'] == v), f"Found experiment {k} changed during the experiment run, which is not expected! (Can use _show_df_config to view more details.) {_df.loc[k,'value']} --> {v}"
            if k in exp_param_map.keys():
                k = exp_param_map[k]
            _df.loc[k, 'value'] = v
        

    
    # @decorator.decorator # 3972290
    def my_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, _show_df_config=False, _exp_param_ls=[], _del_exp_param_ls=[], **kwargs):
            # if function.__name__ not in ['set_df_reduce_graph_rs_path','set_model_perf_rs_path','set_model_perf_rs_path','set_args']:
            #     print(function) 
            #     print(args)
            #     print(kwargs)
            #     print(type(function))
            #     print(333/0)
            if name == None and path == None:
                ctx = args[0]
                assert ctx._exp_name, f"{ctx} has no _exp_name"
                assert ctx._exp_path, f"{ctx} has no _exp_path"

                _df_file = _get_df_file(ctx._exp_name, ctx._exp_path)
            else:
                _df_file = _get_df_file(name, path)
            _df = pd.read_pickle(_df_file)
            func_spec = inspect.getfullargspec(function)
            func_arg_ls = func_spec.args[:len(args)]
            func_kwarg_ls = func_spec.args[len(args):]
            if func_spec.defaults:
                func_kwarg_dict = dict(zip(func_kwarg_ls, func_spec.defaults))
            else:
                func_kwarg_dict = {}
            func_kwarg_dict.update(kwargs)

            for k, v in zip(func_arg_ls, args):
                if 'self' == k:
                    continue
                _check_and_set_df_exp_config_row(_df, k, v, _exp_param_ls + exp_param_ls)
                if k in _del_exp_param_ls:
                    _df = _df.drop(index=k)

            for k, v in func_kwarg_dict.items():
                _check_and_set_df_exp_config_row(_df, k, v, _exp_param_ls + exp_param_ls)
                if k in _del_exp_param_ls:
                    _df = _df.drop(index=k)

            if _show_df_config:
                display(_df)
            _df.to_pickle(_df_file)
            return function(*args, **kwargs)
        return wrapper

    def my_decorator_no_use(function):
        return function
    
    return my_decorator_no_use

    
