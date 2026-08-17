"""
This class defines a registry that allows registering datasets and evaluators to
the molmo internal functions
    1. get_dataset_by_name(name, split)
    2. get_default_max_tokens(name)
    3. get_evaluator(name)

This is allows adding options to these functions from outside of the repository,
making it more easy to use the repo like a library, without having to edit several
internal files.

I've decided to go for a shared registry for all types of functions with the 
following naming convention:
    dataset/<dataset_name>
    max_tokens/<dataset_name>
    evaluator/<dataset_name>

Plase see tests/test_registry.py for some guidance on what types should be. 
"""

class OlmoBuilderRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, builder_id: str, entry_point, kwargs=None):
        if builder_id in self._registry:
            raise ValueError(f"{builder_id} already registered")
        name_hints = ("dataset/", "evaluator/", "max_tokens/")
        # check naming hints are followed
        if not max([builder_id.startswith(hint) for hint in name_hints]):
            print("Warning builder name should start with one of", ", ".join(name_hints), "was", builder_id)
        self._registry[builder_id] = {"entry_point": entry_point, "kwargs": kwargs or {}}

    def make(self, builder_id: str, **override_kwargs):
        spec = self._registry.get(builder_id)
        if not spec:
            raise KeyError(f"Unknown env id: {builder_id}")
        entry = spec["entry_point"]
        cfg = {**spec["kwargs"], **override_kwargs}
        if isinstance(entry, str):
            module, cls = entry.split(":")
            mod = __import__(module, fromlist=[cls])
            entry = getattr(mod, cls)
        return entry(**cfg)

    def list(self):
        return list(self._registry.keys())

# follows scoping, but is a bit like a global variable
registry = OlmoBuilderRegistry()