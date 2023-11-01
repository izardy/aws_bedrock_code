import os
import boto3
from langchain.llms.bedrock import Bedrock
from langchain import PromptTemplate


def get_models():
    
    session = boto3.Session(
        profile_name=os.environ.get("BWB_PROFILE_NAME")
    ) #sets the profile name to use for AWS credentials

    bedrock = session.client(
        service_name='bedrock-runtime', #creates a Bedrock client
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL")
    )
    
    models = bedrock.list_foundation_models()
    
    return [m.get("modelId") for m in models.get("modelSummaries")]


def get_inference_parameters(model): #return a default set of parameters based on the model's provider
    bedrock_model_provider = model.split('.')[0] #grab the model provider from the first part of the model id
    
    if (bedrock_model_provider == 'anthropic'): #Anthropic model
        return { #anthropic
            "max_tokens_to_sample": 512,
            "temperature": 0, 
            "top_k": 250, 
            "top_p": 1, 
            "stop_sequences": ["\n\nHuman:"] 
           }
    
    elif (bedrock_model_provider == 'ai21'): #AI21
        return { #AI21
            "maxTokens": 512, 
            "temperature": 0, 
            "topP": 0.5, 
            "stopSequences": [], 
            "countPenalty": {"scale": 0 }, 
            "presencePenalty": {"scale": 0 }, 
            "frequencyPenalty": {"scale": 0 } 
           }
    
    else: #Amazon
        #For the LangChain Bedrock implementation, these parameters will be added to the 
        #textGenerationConfig item that LangChain creates for us
        return { 
            "maxTokenCount": 512, 
            "stopSequences": [], 
            "temperature": 0, 
            "topP": 0.9 
        }

    
def get_llm(model):
    
    model_kwargs = get_inference_parameters(model)
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id=model, #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm



def get_prompt_for_model(model, input_content):
    
    bedrock_model_provider = model.split('.')[0] #grab the model provider from the first part of the model id
    
    if (bedrock_model_provider == 'anthropic'): #Anthropic model requires "\n\nHuman:" + "\n\nAssistant:" format for prompts
        prompt_template = PromptTemplate.from_template("\n\nHuman:{input_content}\n\nAssistant:")

        prompt = prompt_template.format(input_content=input_content)
        
        return prompt
    else:
        return input_content



def get_text_responses_from_models(models, input_content, callback_handler): #text-to-text client function
    
    for model in models:
        llm = get_llm(model)
        prompt = get_prompt_for_model(model, input_content)
        
        response_text = llm.predict(prompt) #return a response to the prompt
        callback_handler(model, response_text)
