import streamlit as st #all streamlit commands will be available through the "st" alias
import model_comparison_lib as glib #reference to local lib script



st.set_page_config(page_title="Model Comparison") #HTML title
st.title("Model Comparison") #page title



class ResponseProcessor(): #create a class that can capture and display streamed output
    def __init__(self, output_container):
        self.output_container = output_container
        self.combined_output = []
    
    def response_handler(self, model, response):
        st.write("### " + model)
        st.write(response)


@st.cache_resource
def load_models():
    models = glib.get_models()
    return models



#get a list of available models in Amazon Bedrock
available_models = load_models()

# You may need to adjust the list below if newer models have become available to you:
suggested_models = ["ai21.j2-ultra-v1","ai21.j2-ultra-v1","anthropic.claude-v2"]

#only show suggested models if they are available:
presented_models = [m for m in suggested_models if m in available_models]



selected_models = st.multiselect("Select models", options=presented_models, default=presented_models)

input_text = st.text_area("Input text") #display a multiline text box with no label
go_button = st.button("Go", type="primary") #display a primary button



if go_button: #code in this if block will be run when the button is clicked
    
    response_output = st.empty() #create a container to hold the streaming output
    
    response_processor = ResponseProcessor(response_output)
    
    with st.spinner("Working..."): #show a spinner while the code in this with block runs
        
        glib.get_text_responses_from_models(selected_models, input_text, response_processor.response_handler)
        
