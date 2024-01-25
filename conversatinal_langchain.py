import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain_pydantic
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")


class CreateIncident(BaseModel):
    summary: str = Field(
        default="",
        description="Description about the incident.",
    )
    notes: str = Field(
        default="N/A",
        description="Optional description of the notes",
    )

    def reset(self):
        """
        Reset all fields to their default values.
        """
        for field_name, field_value in self.__annotations__.items():
            setattr(self, field_name, '')


class CheckIncident(BaseModel):
    ticketnumber: str = Field(
        default="",
        description="ID of the incident",
    )

    def reset(self):
        """
        Reset all fields to their default values.
        """
        for field_name, field_value in self.__annotations__.items():
            setattr(self, field_name, '')


create_incident_details = CreateIncident()
check_incident_details = CheckIncident()
field_to_fill = None


def ask_for_info(ask_for):
    first_prompt = ChatPromptTemplate.from_template(
        """
            Below are some things to ask the user for in a conversational way. You should only ask one question at a time even if you don't get all the information, \
            don't ask as a list! Don't greet the user! Don't say Hi. Explain you need to get some info. If the ask_for is empty then thank them and ask how you can help them \n\n \
            ### ask_for: {ask_for}
        """
    )
    info_gathering_chain = LLMChain(llm=llm, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat


def check_what_is_empty(user_peronal_details):
    ask_for = []
    for field, value in user_peronal_details.dict().items():
        if value in [None, "", 0]:
            print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for


def add_non_empty_details(current_details: CreateIncident, new_details: CreateIncident):
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details


def filter_response(text_input, user_details):
    chain = create_tagging_chain_pydantic(CreateIncident, llm)
    res = chain.run(text_input)
    user_details = add_non_empty_details(user_details, res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for


def conversationalChainInference(query, form_type_list):
    global create_incident_details
    global check_incident_details
    global field_to_fill

    if form_type_list[0] == 'trigger_form_create_incident':
        if field_to_fill:
            create_incident_details, ask_for = filter_response(field_to_fill + ': ' + query, create_incident_details)
        if check_what_is_empty(create_incident_details):
            field_description = CreateIncident.__fields__[check_what_is_empty(create_incident_details)[0]].field_info.description
            ai_response = ask_for_info(field_description)
            field_to_fill = check_what_is_empty(create_incident_details)[0]
            return ai_response, form_type_list
        else:
            field_to_fill = None
            return_string = f"""MIES APIs not integrated yet, therefore the following details cannot be logged in:
                               summary: {create_incident_details.summary}
            """
            create_incident_details.reset()
            return return_string, ['qna']
    else:
        if len(form_type_list) == 2:
            check_incident_details.ticketnumber = form_type_list[1]
        if field_to_fill:
            check_incident_details, ask_for = filter_response(field_to_fill + ': ' + query, check_incident_details)
        if check_what_is_empty(check_incident_details):
            field_description = CheckIncident.__fields__[check_what_is_empty(check_incident_details)[0]].field_info.description
            ai_response = ask_for_info(field_description)
            field_to_fill = check_what_is_empty(check_incident_details)[0]
            return ai_response, form_type_list
        else:
            field_to_fill = None
            check_incident_details.reset()
            return 'MIES APIs have not been integrated yet, therefore cannot access the status of this incident '+form_type_list[1], ['qna']
