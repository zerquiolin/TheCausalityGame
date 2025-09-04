# Utils
import sys
import inspect
import importlib

# Types
from typing import Type


def get_class(fqcn: str):
    pos_of_last_point = fqcn.rindex(".")
    module_name = fqcn[:pos_of_last_point]
    class_name = fqcn[pos_of_last_point + 1 :]
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def find_importable_classes(folder_route: str, base_class: Type) -> dict[str, Type]:
    """
    Recursively scan the given folder for Python modules and return all classes
    that are subclasses of `base_class`, mapping class names to class objects.

    Args:
        folder_route (str): Path to the folder (e.g., 'causalitygame/missions').
        base_class (Type): The base class to filter by.

    Returns:
        dict[str, Type]: A dictionary mapping class names to discovered class types.
    """
    import pathlib

    candidates = {}

    folder_path = pathlib.Path(folder_route).resolve()
    project_root = folder_path.parents[
        len(folder_path.parts) - folder_path.parts.index("causalitygame") - 1
    ]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    for py_file in folder_path.rglob("*.py"):
        rel_path = py_file.relative_to(project_root).with_suffix("")
        module_name = ".".join(rel_path.parts)

        try:
            module = importlib.import_module(module_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, base_class) and obj is not base_class:
                    if obj.__module__ == module.__name__:
                        candidates[obj.__name__] = obj
        except Exception as _:
            continue

    return candidates
