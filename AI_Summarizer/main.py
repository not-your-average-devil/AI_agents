# Downloading the model from Hugging Face only downloads the weights; the code is in the transformers library.

'''The model is basically code that works by taking input numbers (tokens of sentences),
recognizing patterns in those tokens, and outputting numbers again (processed tokens).'''

from transformers import pipeline
# transformers is a Python library made by Hugging Face.

'''pipeline is a function that loads the model, tokenizes input, and handles complicated steps for you.
(Note: In ML, "pipeline" can also mean a series of processing steps, e.g. data cleaning -> tokenizing -> running model.)'''

from langchain_huggingface import HuggingFacePipeline
# LangChain is a Python framework (collection of functions, classes, modules).
# langchain_huggingface is a module from LangChain that helps integrate Hugging Face transformers with LangChain workflows.

from langchain.prompts import PromptTemplate
# langchain.prompts is another module from LangChain that provides tools and classes for dealing with prompts.

from transformers.utils.logging import set_verbosity_error
# transformers.utils.logging helps control the amount of information printed during execution.

set_verbosity_error()
'''This function suppresses warnings and reduces console 'noise', showing only errors,
so output isn't messy with tons of incomprehensible data.'''

model = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
# Using pipeline to generate summaries with the Facebook BART large CNN model on device 0 (GPU).

llm = HuggingFacePipeline(pipeline=model)
# Wrapping the Hugging Face model in LangChain's format.

template = PromptTemplate.from_template(
    "Summarize the following text in a way a {age} year old would understand:\n\n{text}"
)
# This template will be sent as a prompt to the LLM, with placeholders {age} and {text} replaced by actual values.

summarizer_chain = template | llm
# Chaining the prompt template output as input to the LLM using LangChain's pipe operator.

text_to_summarize = input("\nEnter text to summarize:\n")
age = input("Enter target age for simplification:\n")

summary = summarizer_chain.invoke({"text": text_to_summarize, "age": age})
# Running the summarization chain.

print("\n🔹 **Generated Summary:**")
print(summary[0]['summary_text'])
# Extracting and printing the summary text from the output list.
