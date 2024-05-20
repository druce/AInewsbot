# pip install langchain langchain-openai streamlit
# streamlit run streamlit_qa.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

# import secrets from .env
import dotenv
dotenv.load_dotenv()

MODEL = "gpt-4o"
SYSTEM_PROMPT = """You are a friendly helper chatbot"""

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
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt_template | llm
    return chain


def escape_output(text):
    # escape $ characters to avoid markdown interpreting them as latex
    return text.replace('$', '\$')


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
    prompt = escape_output(prompt)
    memory.add_user_message(prompt)
    with st.chat_message("user", avatar=avatars['human']):
        st.markdown(prompt)
    # process the user's question
    with st.chat_message("assistant", avatar=avatars['ai']):
        message_placeholder = st.text("▌")
        text_response = ""
        for chunk in chain.stream({"messages": memory.messages}):
            text_response += escape_output(chunk.content)
            message_placeholder.markdown(text_response + "▌")
            if hasattr(chunk, 'response_metadata'):
                response_metadata = chunk.response_metadata
        message_placeholder.markdown(text_response)
        memory.add_ai_message(text_response)
