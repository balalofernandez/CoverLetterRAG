import torch
torch.cuda.empty_cache()
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

import gradio as gr


def generate_prompt(pipeline, document_path, asked_question):
    mistral_llm = HuggingFacePipeline(pipeline=pipeline)

    #Retrieve your CV
    gr.Info("Reading document")
    loader = PyPDFLoader(document_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    retriever = vectorstore.as_retriever()

    #Create the Prompt template
    prompt_template = """
    ### [INST] Instruction: Based on the following document regarding my past experience:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain 
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    result = rag_chain.invoke(asked_question)
    return result['text']



#Choose the model, make sure that the prompt then follows the correct structure
model_name='mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
"""bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)"""

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, 
    #attn_implementation="flash_attention_2",
    #quantization_config=bnb_config,
)
model.to_bettertransformer()
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
)

# Create the Gradio interface
iface = gr.Interface(
    fn=lambda file_path,text: generate_prompt(text_generation_pipeline,file_path,text),
    inputs=[gr.File(label="Upload your CV"), gr.Textbox(label="Enter Text")],
    outputs="text"
)

# Launch the app
iface.launch(share=True)
