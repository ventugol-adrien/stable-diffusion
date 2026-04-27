import time
import functools
from pydantic import BaseModel, Field, ConfigDict


def timed(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{self.__class__.__name__}] executed in {elapsed:.3f}s")
        return result

    return wrapper


class BaseNodeModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dependencies: list[str] | None = None
    next_nodes: list[str] | None = None


class BaseNode:
    output_key: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            cls.__call__ = timed(cls.__call__)

    def __init__(self, **kwargs):
        base_node_data = BaseNodeModel(**kwargs)
        self.params = kwargs
        self.dependencies = (
            base_node_data.dependencies
            if base_node_data.dependencies is not None
            else []
        )
        self.next_nodes = (
            base_node_data.next_nodes if base_node_data.next_nodes is not None else []
        )

    def __call__(self, *args, **kwds):
        print(f"Executing node: {self} with params: {self.params}")

    def __enter__(self, *args, **kwds):
        print(f"Entering context for node: {self} with params: {self.params}")

    def __exit__(self, *args, **kwds):
        print(f"Exiting context for node: {self} with params: {self.params}")

    def __str__(self):
        return f"{self.__class__.__name__}(node_type={self.node_type}, params={self.params}, dependencies={self.dependencies}, next_nodes={self.next_nodes})"

    def is_terminal(self):
        return len(self.next_nodes) == 0

    def is_source(self):
        return len(self.dependencies) == 0

    def propagate_context(self, result, context: dict, nodes: dict[str, "BaseNode"]):
        node_id = next(k for k, v in nodes.items() if v is self)
        context[node_id] = result
        if self.output_key is not None:
            for next_node_id in self.next_nodes:
                if next_node_id in nodes:
                    setattr(nodes[next_node_id], self.output_key, result)
