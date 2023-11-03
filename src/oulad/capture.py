"""Capture cache IPython magic implementation."""

import ast
from pathlib import Path

from IPython.core import magic_arguments
from IPython.core.magic import (
    Magics,
    cell_magic,
    magics_class,
    needs_local_scope,
    no_var_expand,
)


@magics_class
class CaptureMagic(Magics):
    """The %%capture magic function caches variable values by name and namespace."""

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("variables", nargs="+")
    @magic_arguments.argument(
        "-ns",
        "--namespace",
        default="default",
        help="""A namespace to cache the variable value. The concatenation of the
        namespace and the variable name should be unique.""",
    )
    @needs_local_scope
    @no_var_expand
    @cell_magic
    def capture(self, line, cell, local_ns):
        """Capture (cache) the values of variables from the cell by namespace.

        Usage:
          %%capture [-ns<N>] var1 var2 ...

        Parameters:
          -ns<N> (str): a namespace string key to cache the values. The combination of
              the namespace and a variable name should be unique.

        Note:
            `cache` would be an more appropriate name for this method, however, IPython
            lexer doesn't support syntax highlighting for custom magics, thus we
            overwrite the existing `capture` magic for now.
        """
        args = magic_arguments.parse_argstring(self.capture, line)
        cache_dir = Path.home() / ".cache/oulad"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cell_ast = ast.parse(cell)
        pickle_dump_asts = []
        for i, node in enumerate(cell_ast.body):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) > 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id not in args.variables:
                continue
            path = cache_dir / ("-".join([args.namespace, target.id]) + ".pkl")
            if path.exists():
                cell_ast.body[i] = ast.parse(
                    f"with open('{path.absolute()}', 'rb') as _:\n"
                    f"  {target.id} = pickle.load(_)"
                ).body[0]
            else:
                pickle_dump_asts.append(
                    ast.parse(
                        f"with open('{path.absolute()}', 'wb') as _:\n"
                        f"  pickle.dump({target.id}, _)"
                    ).body[0]
                )

        for pickle_dump_ast in pickle_dump_asts:
            cell_ast.body.append(pickle_dump_ast)

        # pylint: disable=exec-used
        exec("import pickle", local_ns)  # nosec
        exec(compile(cell_ast, "<magic-capture>", "exec"), local_ns)  # nosec


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    ipython.register_magics(CaptureMagic)
