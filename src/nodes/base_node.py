from pydantic import BaseModel, Field, ConfigDict


class BaseNodeModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dependencies: list[str] | None = None
    next_nodes: list[str] | None = None


class BaseNode:
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
