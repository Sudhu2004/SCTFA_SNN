from functools import lru_cache
import transformers
from langchain.llms import HuggingFacePipeline
from torch import bfloat16,cuda
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from nemoguardrails import LLMRails, RailsConfig
from langchain.prompts import PromptTemplate

from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import register_llm_provider

import transformers
from torch import cuda, bfloat16  #to change the size of the model ot bfloat16 and to place the model on cuda
from langchain.vectorstores import Chroma #vector database to store the embeddings and to generate the context of the quera
from langchain.chains import RetrievalQA # retreive the required information based on the query from the chromadb
from langchain.document_loaders import PyPDFLoader #to load pdf
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #to split the text into required length so that it can be passed into the model

from langchain.llms import HuggingFacePipeline # to load LLM
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings


model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

#configuiring the bits to pass in the model
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  #to convert the linear layers to nf4 layes
    bnb_4bit_quant_type='nf4', #choosing which type of layers are used for quantization
    bnb_4bit_use_double_quant=True, #quantizing the layers again
    bnb_4bit_compute_dtype=bfloat16,
    #change any bit of data inputed into bf16
)            #reducing the size of the model


model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth) #accessing the model with the token


model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,  #passing the quantization configuration
    device_map='auto',
    use_auth_token=hf_auth
)
# model.eval()
    #initializing tokens to convert the query to strings

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
MODEL_ARGS = {
    'max_new_tokens': 2000,
    'temperature':0.5,
    'max_length':2000,
    'repetition_penalty':1.1,
    'do_sample' : False,
    'return_full_text' :True,
}
#initializing the pipeline and passing the model
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens = 2000,
    temperature = 0.5,
    max_length = 2000,
    repetition_penalty = 1.1,
    do_sample =  False,
    return_full_text = True,
)

# create the class / type for the 'hf_pipepline_gpt2' LLM engine
# and yes, we need to specify model_kwargs here again
HFPipelineLlama2 = get_llm_instance_wrapper(
    llm_instance=HuggingFacePipeline(pipeline=generate_text,model_kwargs=MODEL_ARGS),
    llm_type='hf_pipeline_llama2'
)
# register it as a LLM provider for Guardrails
register_llm_provider("hf_pipeline_llama2", HFPipelineLlama2)

loader = DirectoryLoader('', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200,separators=["\n"],length_function = len)
texts = text_splitter.split_documents(documents)
from langchain.embeddings import HuggingFaceInstructEmbeddings

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",
                                                      model_kwargs={"device": "cuda"})

embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding)
retriever1 = vectordb.as_retriever(search_kwargs={"k": 3})
import pandas as pd
import csv
from reportlab.lib.pagesizes import A4  #importing the size of the document
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

df = pd.read_csv('insurance_finance.csv')

pdf_filename = 'insurance_finance-1.pdf'

styles = getSampleStyleSheet() # Initializing the style of sheet
style = styles["Normal"]

doc = SimpleDocTemplate(pdf_filename, pagesize=A4) #initializing the type of file with the SimpleDoctemplate
story =[]
df["Text"] = df["question"].astype(str) + "\n" + df["answer"].astype(str)

textdata = df['Text'].tolist()

for item in textdata:
    paragraph = Paragraph(item, style)
    story.append(paragraph)

doc.build(story)
loader = DirectoryLoader('', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200,separators=["\n"],length_function = len)
texts = text_splitter.split_documents(documents)
# from langchain.embeddings import HuggingFaceInstructEmbeddings

# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",
#                                                       model_kwargs={"device": "cuda"})

embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding)
retriever2 = vectordb.as_retriever(search_kwargs={"k": 3})
colang_content = """
# define limits
define user ask HR
    "Tell me about your previous work experience."
    "What interests you about our company?"
    "How do you handle tight deadlines?"
    "Describe a challenging team project you managed."
    "What's your approach to employee conflict resolution?"
    "How do you stay updated on industry trends?"
    "Can you give an example of handling sensitive employee data?"
    "How do you ensure diversity and inclusion?"
    "What's your strategy for talent recruitment?"
    "Discuss your familiarity with labor laws."
    "Experience"
    "Fit"
    "Conflict Resolution"
    "Teamwork"
    "Management"
    "Diversity"
    "Recruitment"
    "Compliance"
    "Performance"
    "Training and Development"
    "Communication"
    "Change Management"
    "Motivation"
    "Legal Knowledge"
    "Employee Relations"
    "Compensation"
    "HR Technology"
    "Onboarding"
    "Workforce Planning"
    "Succession Planning"

define flow HR
    user ask HR
    $answer = execute HR(query=$last_user_message)
    bot $answer

# define RAG intents and flow
define user ask finance
    "What's my annual salary increase this year?"
    "When will I receive my bonus?"
    "What's my salary for the next quarter?"
    "Can I get a raise this year?"
    "When is the next pay review?"
    "Is my paycheck ready for pickup?"
    "What's my hourly wage for overtime?"
    "Any news on my performance bonus?"
    "Can I check my salary statement?"
    "What's the pay scale for my role?"
    "When do raises get processed?"
    "What's my salary for this month?"
    "Is my commission payment in?"
    "What's the salary for new hires?"
    "How much is my annual salary?"
    "Raise"
    "Bonus"
    "Paycheck"
    "Overtime"
    "Performance"
    "Statement"
    "Pay scale"
    "Raises"
    "Commission"
    "New hires"
    "Annual salary"

define flow finance
    user ask finance
    $answer = execute finance(query=$last_user_message)
    bot $answer
"""

yaml_content = """
models:
  - type: main
    engine: hf_pipeline_llama2
    parameters:
      max_new_tokens: 2000
      tokens_to_generate: 2000
      max_length : 2000

instructions:
  - type: general
    content: |
      Below is a conversation between a bot and a user about the recent job reports.
      The bot is factual and concise. If the bot does not know the answer to a
      question, it truthfully says it does not know.

sample_conversation: |
  user "Hello there!"
    express greeting
  bot express greeting
    "Hello! How can I assist you today?"
  user "What can you do for me?"
    ask about capabilities
  bot respond about capabilities
    "I am an AI assistant which helps answer questions based on a given knowledge base."

# The prompts below are the same as the ones from `nemoguardrails/llm/prompts/dolly.yml`.
prompts:
  - task: general
    models:
      - hf_pipeline_llama2
    content: |-
      {{ general_instructions }}

      {{ history | user_assistant_sequence }}
      Assistant:

  # Prompt for detecting the user message canonical form.
  - task: generate_user_intent
    models:
      - hf_pipeline_llama2
    content: |-


      # This is how a conversation between a user and the bot can go:
      {{ sample_conversation | verbose_v1 }}

      # This is how the user talks:
      {{ examples | verbose_v1 }}

      # This is the current conversation between the user and the bot:
      {{ sample_conversation | first_turns(2) | verbose_v1 }}
      {{ history | colang | verbose_v1 }}

    output_parser: "verbose_v1"

  # Prompt for generating the next steps.
  - task: generate_next_steps
    models:
      - hf_pipeline_llama2
    content: |-

      # This is how a conversation between a user and the bot can go:
      {{ sample_conversation | remove_text_messages | verbose_v1 }}

      # This is how the bot thinks:
      {{ examples | remove_text_messages | verbose_v1 }}

      # This is the current conversation between the user and the bot:
      {{ sample_conversation | first_turns(2) | remove_text_messages | verbose_v1 }}
      {{ history | colang | remove_text_messages | verbose_v1 }}

    output_parser: "verbose_v1"

  # Prompt for generating the bot message from a canonical form.
  - task: generate_bot_message
    models:
      - hf_pipeline_llama2
    content: |-
      # This is how a conversation between a user and the bot can go:
      {{ sample_conversation | verbose_v1 }}

      {% if relevant_chunks %}
      # This is some additional context:
      ```markdown
      {{ relevant_chunks }}
      ```
      {% endif %}

      # This is how the bot talks:
      {{ examples | verbose_v1 }}

      # This is the current conversation between the user and the bot:
      {{ sample_conversation | first_turns(2) | verbose_v1 }}
      {{ history | colang | verbose_v1 }}

    output_parser: "verbose_v1"

  # Prompt for generating the value of a context variable.
  - task: generate_value
    models:
      - hf_pipeline_llama2
    content: |-
      # This is how a conversation between a user and the bot can go:
      {{ sample_conversation | verbose_v1 }}

      # This is how the bot thinks:
      {{ examples | verbose_v1 }}

      # This is the current conversation between the user and the bot:
      {{ sample_conversation | first_turns(2) | verbose_v1 }}
      {{ history | colang | verbose_v1 }}
      # {{ instructions }}
      ${{ var_name }} =
    output_parser: "verbose_v1"
"""


# async def retrieve(query: str) -> list:
#     # create query embedding
#     docs = retriever.get_relevant_documents(query)
#     combined_content = []
#     # Iterate through the docs and append the page_content to the list
#     for i in range(len(docs)):
#         combined_content.append(docs[i].page_content)

#     # Combine the page_content into a single string
#     contexts = "\n".join(combined_content)

#     return contexts
async def finance(query: str) ->str:
    print("finance called")
    # place query and contexts into RAG prompt
    docs = retriever2.get_relevant_documents(query)
    combined_content = []
    # Iterate through the docs and append the page_content to the list
    for i in range(len(docs)):
        combined_content.append(docs[i].page_content)

    # Combine the page_content into a single string
    context = "\n".join(combined_content)


#Creating the prompt_template so that this prompt is used before sending any query to the model hiding it to the user
    prompt_template = """Use the following pieces of context to answer the question at the end accuretly..
    If you don't know the answer to a question or if any question is asked outside of the context of finance ,
     just say that you don't know. only generate an answer with less than 50 words

    {context}

    Question: {question}
    Answer :"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    #Initialing the rag_pipeline (Retrieval Augmented Generation) to get the relavant data from the document with the required prompt passed to the model
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=app.llm,
        chain_type='stuff',
        retriever=retriever1,
        chain_type_kwargs=chain_type_kwargs
    )

    return rag_pipeline.run(query)

async def HR(query: str) -> str:
    print("> HR Called")  # we'll add this so we can see when this is being used
    # place query and contexts into RAG prompt
    docs = retriever1.get_relevant_documents(query)
    combined_content = []
    # Iterate through the docs and append the page_content to the list
    for i in range(len(docs)):
        combined_content.append(docs[i].page_content)

    # Combine the page_content into a single string
    context = "\n".join(combined_content)


#Creating the prompt_template so that this prompt is used before sending any query to the model hiding it to the user
    prompt_template = """Use the following pieces of context to answer the question at the end accuretly.
    If any one asks outside the box questions then say I am only trained on giving answers to questions related to HR .
    If you don't know the answer to a question or if any question is asked outside of the context of HR ,
     just say that you don't know. Please do not try to make up an answer and only generate an answer with less than 50 words

    {context}

    Question: {question}
    Answer :"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    #Initialing the rag_pipeline (Retrieval Augmented Generation) to get the relavant data from the document with the required prompt passed to the model
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=app.llm,
        chain_type='stuff',
        retriever=retriever2,
        chain_type_kwargs=chain_type_kwargs
    )

    return rag_pipeline.run(query)

config = RailsConfig.from_content(colang_content, yaml_content)
app = LLMRails(config)

# app.register_action(action=retrieve, name="retrieve")
app.register_action(action=finance, name="finance")
app.register_action(action=HR, name="HR")