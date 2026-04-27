from pydantic import BaseModel, Field


class BaseNodeModel(BaseModel):
    id: str = Field(..., description="Unique identifier for the node")
    dependencies: list[str] | None = None
    next_nodes: list[str] | None = None


class BaseNode:
    def __init__(self, node: BaseNodeModel, **kwargs):
        self.id = node.id
        self.params = kwargs
        self.dependencies = node.dependencies if node.dependencies is not None else []
        self.next_nodes = node.next_nodes if node.next_nodes is not None else []

    def __call__(self, *args, **kwds):
        print(f"Executing node: {self} with params: {self.params}")

    def __enter__(self, *args, **kwds):
        print(f"Entering context for node: {self} with params: {self.params}")

    def __exit__(self, *args, **kwds):
        print(f"Exiting context for node: {self} with params: {self.params}")

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id})"

    def is_terminal(self):
        return len(self.next_nodes) == 0

    def is_source(self):
        return len(self.dependencies) == 0
