"""
This module contains code for defining new neural networks and their trainable parameters
"""

from ..ad import Node
from .utils import _addindent, make_into_bf_node, _IncompatibleKeys
from collections import OrderedDict
from collections.abc import Iterable
from jax import numpy as jnp
import itertools
import torch
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union


class Parameter(Node):
    """
    A trainable parameter of a neural network layer.

    This is just syntactic sugar for a Node with no inputs.
    """

    def __init__(self, value, name=None):
        super(Parameter, self).__init__(value, name=name)


class Network:
    """
    A neural network with trainable parameters

    Attributes:
        parameters: a list of the trainable parameters for this layer

    Tensorflow analogue: tf.keras.layers.Layer / tf.keras.Model
        * Our Network class encompases concepts from both these keras classes.

    PyTorch analogue: torch.nn.Module
    """

    def __init__(self):
        self._modules: Dict[str, Optional[Network]] = OrderedDict()
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Node]] = OrderedDict()
        self._non_persistent_buffers_set: Set[str] = set()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Perform the forward pass of the network layer
        """
        raise NotImplementedError("Layer __call__ method not implemented!")

    def get_typename(self):
        return self.typename

    # @property
    # def parameters(self):
    #     """
    #     Retrieve the list of trainable parameters for this network
    #     """
    #     params = []
    #     # Get parameters from all attributes of this Network
    #     for key, val in vars(self).items():
    #         params.extend(_get_parameters(val))
    #     return params

    def add_module(self, name: str, module: Optional["Network"]) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Network) and module is not None:
            raise TypeError("{} is not a Network subclass".format(module.get_typename()))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(module.get_typename()))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif "." in name:
            raise KeyError('module name can\'t contain ".", got: {}'.format(name))
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        self._modules[name] = module

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign parameter before Network.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. " "Got {}".format(type(name)))
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                "cannot assign '{}' object to parameter '{}' "
                "(bf.Parameter or None required)".format(type(param), name)
            )
        elif param.inputs:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name)
            )
        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Node], persistent: bool = True) -> None:
        r"""Adds a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. " "Got {}".format(type(name)))
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Node):
            raise TypeError(
                "cannot assign '{}' object to buffer '{}' " "(bf.Node or None required)".format(type(tensor), name)
            )
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def __setattr__(self, name: str, value: Union[Node, "Network"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as parameter '{}' " "(bf.Parameter or None expected)".format(type(value), name)
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Network):
                if modules is None:
                    raise AttributeError("cannot assign module before Network.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as child module '{}' "
                        "(bf.Network or None expected)".format(type(value), name)
                    )
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Node):
                        raise TypeError(
                            "cannot assign '{}' as buffer '{}' " "(bf.Node or None expected)".format(type(value), name)
                        )
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Union[Node, "Network"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _get_name(self):
        return self.__class__.__name__

    def _named_members(self, get_members_fn, prefix="", recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        gen = self._named_members(lambda module: module._parameters.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator[Node]:
        r"""Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Node]]:
        r"""Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            (string, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())

        """
        gen = self._named_members(lambda module: module._buffers.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator["Network"]:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Network"]]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator["Network"]:
        r"""Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
                    print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set["Network"]] = None, prefix: str = "", remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        # for hook in self._load_state_dict_pre_hooks.values():
        #     hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not isinstance(input_param, (torch.Tensor, Node)):
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        "expected torch.Tensor or Tensor-like object or bf.Node from checkpoint but "
                        "received {}".format(key, type(input_param))
                    )
                    continue

                # Hopefully none of the bert parameters are lazy
                # # This is used to avoid copying uninitialized parameters into
                # # non-lazy modules, since they dont have the hook to do the checks
                # # in such case, it will error when accessing the .shape attribute.
                # is_param_lazy = torch.nn.parameter.is_lazy(param)
                # # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                # if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                #     input_param = input_param[0]

                # if not is_param_lazy and input_param.shape != param.shape:
                #     # local shape should match the one in checkpoint
                #     error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                #                       'the shape in current model is {}.'
                #                       .format(key, input_param.shape, param.shape))
                #     continue
                try:
                    # param is the local state, input_param is the new value we want to set it to.
                    input_param: Node = make_into_bf_node(input_param)
                    param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}, "
                        "an exception occurred : {}.".format(key, param.shape, input_param.shape, ex.args)
                    )
            elif strict:
                missing_keys.append(key)

        # Commenting this out for now because we probably don't need extra state. If BERT does have it, then we'll have include this.
        # extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        # if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
        #     if extra_state_key in state_dict:
        #         self.set_extra_state(state_dict[extra_state_key])
        #     elif strict:
        #         missing_keys.append(extra_state_key)
        # elif strict and (extra_state_key in state_dict):
        #     unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                # if key.startswith(prefix) and key != extra_state_key:
                if key.startswith(prefix):
                    input_name = key[len(prefix) :]
                    input_name = input_name.split(".", 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

            # # Note that the hook can modify missing_keys and unexpected_keys.
            # incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            # for hook in module._load_state_dict_post_hooks.values():
            #     out = hook(module, incompatible_keys)
            #     assert out is None, (
            #         "Hooks registered with ``register_load_state_dict_post_hook`` are not"
            #         "expected to return new values, if incompatible_keys need to be modified,"
            #         "it should be done inplace."
            #     )

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys))
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs))
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


def _get_parameters(val):
    if isinstance(val, Parameter):
        return [val]
    elif isinstance(val, dict):
        return [__get_parameters(v) for k, v in val.items()]
    elif isinstance(val, Iterable):
        return [__get_parameters(v) for v in val]
    # Recursively get parameters from Networks
    # (Allows compositionally building complex networks out of simple ones)
    elif isinstance(val, Network):
        return val.parameters
    else:
        return []
