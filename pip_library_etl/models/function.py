from pydantic import BaseModel


class Function(BaseModel):
    signature: str
    docs: str
    name: str
