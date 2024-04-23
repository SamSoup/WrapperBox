from abc import ABC
import inspect
import os
import importlib


class GenericFactory(ABC):
    """
    Generic factory.

    Loops through the current directory of the extension factory,
    registering any classes implementing the class named identified
    in "interface.py" of that working directory. Ensure that there is
    only one class in "interface.py"
    """

    def __init__(self, filename: str = "interface.py"):
        self._handlers = {}
        # Get the current file path: note that use __class__
        # instead of __file__ because we really want the instantiating
        # subclasses' location
        current_file_path = inspect.getfile(self.__class__)
        # Get the directory of the current file as full path
        self.current_dir = os.path.dirname(current_file_path)
        # Create a module spec from the file location
        spec = importlib.util.spec_from_file_location(
            filename[:-3], os.path.join(self.current_dir, filename)
        )
        # Create a module from the spec
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module_name = module.__name__
        # Get the names of all classes in the module that aren't imported
        class_names = [
            name
            for name, obj in inspect.getmembers(module)
            if inspect.isclass(obj) and obj.__module__ == module_name
        ]
        print(class_names)
        if len(class_names) > 1:
            raise ValueError(
                f"More than one class identified in {filename}"
                f" for directory {self.current_dir}"
            )
        self.interface_name = class_names[0]
        # Get the base name of the current directory
        self.package = os.path.basename(self.current_dir)
        self._load_handlers()

    def register_handler(self, key, handler):
        self._handlers[key] = handler

    def _load_handlers(self):
        # Scan the current package, looking for classes that
        # implement the interface_name
        for file_name in os.listdir(self.current_dir):
            if file_name.endswith(".py") and file_name != os.path.basename(
                __file__
            ):
                # remove .py extension and add '.' for relative import
                module_name = "." + file_name[:-3]
                module = importlib.import_module(
                    module_name, package=self.package
                )
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and self.interface_name in name
                        and not inspect.isabstract(obj)
                    ):
                        self.register_handler(name, obj)

    def get_handler(self, key):
        handler = self._handlers.get(key)
        if not handler:
            raise ValueError(f"No handler found for {key}")
        return handler()
