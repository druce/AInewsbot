# pip install langchain langchain-openai streamlit
# streamlit run streamlit_qa.py
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (ChatPromptTemplate,
                                    PromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    MessagesPlaceholder)
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ChatMessageHistory

from bb_tools import tool_list
from BB_agent_tool import bb_agent_system_prompt

# import secrets from .env
import dotenv
dotenv.load_dotenv()

MODEL = "gpt-4o"

# ############################################################################################
# only run decorated items once at startup, return cached item thereafter
# (st.cache_data=serializable, resource=unserializable)
# https://docs.streamlit.io/library/advanced-features/caching
# ############################################################################################


@st.cache_resource()
def get_llm():
    """
    Initializes the language model (llm) and returns it.
    With cache_resource decorator is only run once per session

    Returns:
    - llm (ChatOpenAI): The initialized language model.

    """
    return ChatOpenAI(model=MODEL, temperature=0.0, streaming=True)


@st.cache_resource()
def get_memory():
    """
    Initializes and returns the conversational memory.
    Uses the global llm variable. With cache_resource decorator is only run once per session

    Returns:
        ChatMessageHistory: An instance of the ChatMessageHistory class representing the conversational memory.
    """
    return ChatMessageHistory(llm=llm, max_token_limit=1000)


@st.cache_resource()
def get_chain():
    global llm
    prompt_template = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=bb_agent_system_prompt,)),
         MessagesPlaceholder(variable_name='chat_history', optional=True),
         HumanMessagePromptTemplate(prompt=PromptTemplate(
             input_variables=['input'], template='{input}')),
         MessagesPlaceholder(variable_name='agent_scratchpad')]
    )

    agent = create_tool_calling_agent(llm, tool_list, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

    return agent_executor


# Streamlit event loop starts here
title = "Stock Question Answering Chatbot"
st.set_page_config(page_title=title, page_icon=":robot:")
llm = get_llm()
memory = get_memory()
chain = get_chain()

st.title(title)

# show messages sent so far
avatars = {'ai': "🖥️", 'human': '🧑'}
for message in memory.messages:
    if message.type in {'ai', 'human'}:
        with st.chat_message(message.type, avatar=avatars[message.type]):
            st.markdown(message.content)

# show input box, or get new user message if entered
if prompt := st.chat_input("Enter a question about a stock or company:"):
    # add user message to conversational memory and display it in the history
    memory.add_user_message(prompt)
    with st.chat_message("user", avatar=avatars['human']):
        st.markdown(prompt)
    # process the user's question
    with st.chat_message("assistant", avatar=avatars['ai']):
        message_placeholder = st.text("▌")
        response = chain.invoke({"input": prompt})
        text_response = response["output"]
        text_response = escape_output(prompt)

        # for chunk in chain.stream({"messages": memory.messages}):
        #     text_response += chunk.content
        #     message_placeholder.markdown(text_response + "▌")
        #     if hasattr(chunk, 'response_metadata'):
        #         response_metadata = chunk.response_metadata
        message_placeholder.markdown(text_response)
        memory.add_ai_message(text_response)
