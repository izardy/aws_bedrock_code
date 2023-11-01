import os
import json
import boto3
import math


session = boto3.Session(
    profile_name=os.environ.get("BWB_PROFILE_NAME")
) #sets the profile name to use for AWS credentials

bedrock = session.client(
    service_name='bedrock-runtime', #creates a Bedrock client
    region_name=os.environ.get("BWB_REGION_NAME"),
    endpoint_url=os.environ.get("BWB_ENDPOINT_URL")
) 



def get_embedding(text):
    body = json.dumps({"inputText": text})
    model_d = 'amazon.titan-embed-text-v1'
    mime_type = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=model_d, accept=mime_type, contentType=mime_type)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding


#Build the list of embeddings to compare
embeddings = []

embeddings.append(get_embedding("Can you please tell me how to get to the bakery?"))
embeddings.append(get_embedding("I need directions to the bread shop"))
embeddings.append(get_embedding("Cats, dogs, and mice"))
embeddings.append(get_embedding("Felines, canines, and rodents"))
embeddings.append(get_embedding("Four score and seven years ago"))


#print the table of embeddings
i = 1
print("", end="\t")
for e2 in embeddings: #print the column headers
    print(i, end="\t")
    i = i + 1
    
print() #new line

i = 1
for e1 in embeddings:
    #print the row
    print(i, end="\t")
    for e2 in embeddings:
        dist = math.dist(e1, e2) #find the Euclidean distance between each embedding
        
        dist_round = round(dist, 2)
        
        print(dist_round, end="\t")
    
    print() #new line
    i = i + 1

#print(embeddings[0])