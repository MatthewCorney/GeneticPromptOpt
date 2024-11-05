from pydantic import BaseModel


class BaseInputClass(BaseModel):
    def get_input_fields(self) -> dict[str, str]:
        return {name: getattr(self, name) for name, field in self.__fields__.items() if
                field.json_schema_extra.get("input")}

    def get_output_fields(self) -> dict[str, str]:
        return {name: getattr(self, name) for name, field in self.__fields__.items() if
                field.json_schema_extra.get("output")}

    def form_input_query(self) -> str:
        inputs = [f"{k} : {v}" for k, v in self.get_input_fields().items()]
        return '\n'.join(inputs)

    @property
    def response_model(self) -> type[BaseModel]:
        # Create an Answer instance with the answer field from QuestionAnswer
        return type[BaseModel]


class NewPrompt(BaseModel):
    new_prompt: str
