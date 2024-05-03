from typing import List,Any

from pydantic import BaseModel


class Parameter(BaseModel):
    name: str
    value: Any
    description: str
    dtype: str


class Task(BaseModel):
    task_id: int
    function_name: str
    parameters: List[Parameter]
    outputs: List[str]
    description: str


class Plan(BaseModel):
    tasks: List[Task]
    
    def __str__(self):
        """
        Override the __str__ method to provide a string representation of the Plan.
        """
        return self.model_dump_json(indent=4)
