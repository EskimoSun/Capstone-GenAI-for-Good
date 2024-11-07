import time
import os
import base64
import io
import mesop as me
import google.generativeai as genai
import chromadb
import PyPDF2  # For extracting text from PDF files
import csv
import pandas as pd

# Remember to set your API key here
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC52jr2wKQnpg592NZgw1dq2LkeZL5GDaI'

# Initialize API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("my_collection")

# Instantiate the GenAI model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
)
chat_session = model.start_chat(history=[])

# Specify filepath to LiarPlus dataset
liar_plus_filepath = "data/train_100.csv"

@me.stateclass
class State:
    input: str
    output: str
    in_progress: bool
    db_input: str
    db_output: str
    file: me.UploadedFile  # Storing the uploaded file
    pdf_text: str = ""  # Store extracted PDF text

# Master function for the page
@me.page(path="/")
def page():
    with me.box(
        style=me.Style(
            background="#fff",
            min_height="calc(100% - 48px)",
            padding=me.Padding(bottom=16),
        )
    ):
        with me.box(
            style=me.Style(
                width="min(720px, 100%)",
                margin=me.Margin.symmetric(horizontal="auto"),
                padding=me.Padding.symmetric(horizontal=16),
            )
        ):
            header_text()
            uploader()
            display_pdf_text()
            chat_input()
            output()
            db_input()
            db_output()
            process_dataset_button()
            # convert_store_lp_data()
            # convert_store_predai_data()
        footer()

def header_text():
    with me.box(
        style=me.Style(
            padding=me.Padding(top=64, bottom=36),
        )
    ):
        me.text(
            "Veracity Machine",
            style=me.Style(
                font_size=36,
                font_weight=700,
                background="linear-gradient(90deg, #4285F4, #AA5CDB, #DB4437) text",
                color="transparent",
            ),
        )

# Upload function for PDF article
def uploader():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding.all(15))):
        me.uploader(
            label="Upload PDF",
            accepted_file_types=["application/pdf"],
            on_upload=handle_upload,
            type="flat",
            color="primary",
            style=me.Style(font_weight="bold"),
        )

        if state.file and state.file.mime_type == 'application/pdf':
            with me.box(style=me.Style(margin=me.Margin.all(10))):
                me.text(f"File name: {state.file.name}")
                me.text(f"File size: {state.file.size} bytes")
                me.text(f"File type: {state.file.mime_type}")
                # Extract text from the PDF after upload
                extract_text_from_pdf(state.file)

def handle_upload(event: me.UploadEvent):
    state = me.state(State)
    state.file = event.file

def extract_text_from_pdf(file: me.UploadedFile):
    """Extracts text from the uploaded PDF file and stores it in the state."""
    state = me.state(State)
    
    # Wrap the bytes content in a BytesIO object
    pdf_file = io.BytesIO(file.getvalue())

    # Initialize the PDF reader
    pdf_reader = PyPDF2.PdfReader(pdf_file)  
    extracted_text = ""
    
    # Extract text from each page
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text()

    state.pdf_text = extracted_text  # Store extracted PDF text in state
    
    # Debugging: Print extracted text to ensure it's correct
#    if state.pdf_text:
#        print(f"Extracted PDF Text: {state.pdf_text[:100]}")  # Display first 100 characters

def display_pdf_text():
    """Display the extracted PDF text."""
    state = me.state(State)
    if state.pdf_text:
        with me.box(style=me.Style(padding=me.Padding.all(10))):
            me.text("Extracted PDF Content:")
            me.text(state.pdf_text)  # Display the entire text

# User input for GenAI prompting
def chat_input():
    state = me.state(State)
    with me.box(
        style=me.Style(
            padding=me.Padding.all(8),
            background="white",
            display="flex",
            width="100%",
            border=me.Border.all(
                me.BorderSide(width=0, style="solid", color="black")
            ),
            border_radius=12,
            box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a",
        )
    ):
        with me.box(
            style=me.Style(
                flex_grow=1,
            )
        ):
            me.native_textarea(
                value=state.input,
                autosize=True,
                min_rows=4,
                placeholder="Enter your prompt",
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    background="white",
                    outline="none",
                    width="100%",
                    overflow_y="auto",
                    border=me.Border.all(
                        me.BorderSide(style="none"),
                    ),
                ),
                on_blur=textarea_on_blur,
            )
        with me.content_button(type="icon", on_click=click_send):
            me.icon("send")

def textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.input = e.value

def click_send(e: me.ClickEvent):
    state = me.state(State)
    if not state.input:
        return
    state.in_progress = True
    input_text = state.input
    combined_input = combine_pdf_and_prompt(input_text, state.pdf_text)  # Combine prompt with PDF text
    
    # Debugging: Log the combined input
#    print(f"Combined input for AI:\n{combined_input}")

    state.input = ""
    yield

    for chunk in call_api(combined_input):
        state.output += chunk
        yield
    state.in_progress = False
    yield

def combine_pdf_and_prompt(prompt: str, pdf_text: str) -> str:
    """Combines the user's prompt with the extracted PDF text."""
    if not pdf_text:
        return prompt
    return f"Prompt: {prompt}\n\nPDF Content: {pdf_text}"  # Combine entire text

# Sends API call to GenAI model with user input
def call_api(input_text):
    # Disabled until implementation of database retrieval
    # results = collection.query(
    #    query_texts=[input_text],
    #    n_results=3,
    #    include=["documents", "metadatas"]
    # )

    context = " "#.join(results['documents'][0]) if results['documents'] else ""
    # Add context to the prompt
    full_prompt = f"Context: {context}\n\nUser: {input_text}"
    response = chat_session.send_message(full_prompt)
    time.sleep(0.5)
    yield response.candidates[0].content.parts[0].text


# Display output from GenAI model
def output():
    state = me.state(State)
    if state.output or state.in_progress:
        with me.box(
            style=me.Style(
                background="#F0F4F9",
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin(top=36),
            )
        ):
            if state.output:
                me.markdown(state.output)
            if state.in_progress:
                with me.box(style=me.Style(margin=me.Margin(top=16))):
                    me.progress_spinner()

# Manual add function for database (would be deprecated when dataset processing is fully automized)
def db_input():
    state = me.state(State)
    with me.box(
        style=me.Style(
            padding=me.Padding.all(8),
            background="white",
            display="flex",
            width="100%",
            border=me.Border.all(
                me.BorderSide(width=0, style="solid", color="black")
            ),
            border_radius=12,
            box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a",
            margin=me.Margin(top=36),
        )
    ):
        with me.box(
            style=me.Style(
                flex_grow=1,
            )
        ):
            me.native_textarea(
                value=state.db_input,
                autosize=True,
                min_rows=4,
                placeholder="Enter statement to be added to database",
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    background="white",
                    outline="none",
                    width="100%",
                    overflow_y="auto",
                    border=me.Border.all(
                        me.BorderSide(style="none"),
                    ),
                ),
                on_blur=db_textarea_on_blur,
            )
        with me.content_button(type="icon", on_click=click_add_to_db):
            me.icon("add")

def db_textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.db_input = e.value

def click_add_to_db(e: me.ClickEvent):
    state = me.state(State)
    if not state.db_input:
        return
    
    # Process inputted text
    analysis = analyze_statement(state.db_input)
    score = extract_veracity_score(analysis)

    # Add to Chroma database
    if score is not None:
        collection.add(
            documents=[state.db_input],
            metadatas=[{"analysis": analysis, "veracity_score": score}],
            ids=[f"doc_{int(time.time())}"]  # Using timestamp as unique ID
        )
    else: 
        print(f"Failed to extract veracity score for statement: {state.db_input[:50]}...")

    state.db_output = f"Manually added to database: {state.db_input[:50]}..."
    state.db_input = ""
    yield

# Display database state for added vector
def db_output():
    state = me.state(State)
    if state.db_output:
        with me.box(
            style=me.Style(
                background="#F0F4F9",
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin(top=36),
            )
        ):
            me.text(state.db_output)

# Page footer
def footer():
    with me.box(
        style=me.Style(
            position="sticky",
            bottom=0,
            padding=me.Padding.symmetric(vertical=16, horizontal=16),
            width="100%",
            background="#F0F4F9",
            font_size=14,
        )
    ):
        me.html(
            "Made with <a href='https://google.github.io/mesop/'>Mesop</a>",
        )





# Load dataset and collect string statements
def process_dataset(file_path):
    statements = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row if present
        for row in reader:
            statements.append(row[2])
    return statements

# API call to GenAI for each dataset statement
def analyze_statement(statement):
    # Note: Need to make sure GenAI outputs the correct format.
    #       Also replace factors with explanation of factuality factors.
    prompt = f"""
    Analyze the following political statement for factuality:
    
    "{statement}"
    
    Consider the following factors:
    1. Verifiability of claims
    2. Use of reliable sources
    3. Logical consistency
    4. Context and completeness
    5. Potential biases
    
    Based on these factors, rate the veracity of the statement on a scale of 0 to 10, 
    where 0 is completely false and 10 is completely true.
    
    Provide your analysis and the final veracity score in the following format:
    Analysis: [Your detailed analysis here]
    Veracity Score: [Score between 0 and 10, do not include any string explanation]

    If the final veracity score could not be determined due to crucial missing information, follow the following format:
    Analysis: [Your detailed analysis here]
    Score Missing Reason: [Your reason for not reaching a score]
    """
    
    response = chat_session.send_message(prompt)
    return response.text

# Helper function to extract veracity score from GenAI output
def extract_veracity_score(analysis):
    lines = analysis.split('\n')
    for line in lines:
        if line.startswith("Veracity Score:"):
            return float(line.split(':')[1].strip())
    return None

# Iterate over dataset statements, send API calls, and store into database
def process_and_store_statements(statements):
    for statement in statements:
        analysis = analyze_statement(statement)
        score = extract_veracity_score(analysis)
        
        # Store in chromaDB
        if score is not None:
            # Stores original statement, full outputted analysis, and veracity score
            collection.add(
                documents=[statement],
                metadatas=[{"analysis": analysis, "veracity_score": score}],
                ids=[f"doc_{int(time.time())}"]
            )
            print(f"Added to database: {statement[:50]}...")
        else:
            print(f"Failed to extract veracity score for statement: {statement[:50]}...")

        # avoid rate limit
        time.sleep(1)

# Buttom element for initiating dataset processing
def process_dataset_button():
    with me.box(style=me.Style(margin=me.Margin(top=36))):
        me.button(
            "Process LiarPlus Dataset",
            on_click=click_process_dataset,
            color="primary",
        )

# Dataset processing
def click_process_dataset(e: me.ClickEvent):
    state = me.state(State)
    state.in_progress = True
    yield

    # Assuming the dataset is stored in a CSV file
    statements = process_dataset(liar_plus_filepath)
    process_and_store_statements(statements)

    state.in_progress = False
    state.output = "Political statements processed and stored in the database."
    yield

#converting lierplus dataset and store in datebase
def convert_store_lp_data():
    train_data = pd.read_csv('data/train2.tsv', sep='\t',header=None, dtype=str)
    test_data = pd.read_csv('data/test2.tsv', sep='\t', header=None, dtype=str)
    validate_data = pd.read_csv('data/val2.tsv', sep='\t',header=None, dtype=str)

    datasets = [
        {"data": train_data, "source": "train"},
        {"data": test_data, "source": "test"},
        {"data": validate_data, "source": "validate"}
    ]
   
    for dataset in datasets:
        source = dataset["source"]
        data = dataset["data"]
     # Iterate over each row, combining it into a paragraph and processing it
        for idx, row in data.iterrows():
            # Combine row data into a single string (statement + metadata)
            statement = ', '.join(row.astype(str))

            # Store statement and metadata in ChromaDB
            collection.add(
                documents=[statement],         
                metadatas=[{"source": source, "row_index": idx}],            
                ids=[f"{source}_doc_{idx}"]   
            )

    print("All data has been successfully processed and stored in ChromaDB.")


#converting predictive ai generated dataset and store in datebase
def convert_store_predai_data():
    train_data = pd.read_csv('PredictiveAI/train_data_full.tsv', sep='\t',header=None, dtype=str)
    test_data = pd.read_csv('PredictiveAI/test_data_full.tsv', sep='\t', header=None, dtype=str)
    validate_data = pd.read_csv('PredictiveAI/val_data_full.tsv', sep='\t',header=None, dtype=str)
    micro_factors = pd.read_csv('PredictiveAI/average_scores.tsv', sep='\t',header=None, dtype=str)

    datasets = [
        {"data": train_data, "source": "train"},
        {"data": test_data, "source": "test"},
        {"data": validate_data, "source": "validate"},
        {"data": micro_factors, "source": "factors"}
    ]
   
    for dataset in datasets:
        source = dataset["source"]
        data = dataset["data"]
     # Iterate over each row, combining it into a paragraph and processing it
        for idx, row in data.iterrows():
            # Combine row data into a single string (statement + metadata)
            statement = ', '.join(row.astype(str))

            # Store statement and metadata in ChromaDB
            collection.add(
                documents=[statement],         
                metadatas=[{"source": source, "row_index": idx}],            
                ids=[f"{source}_doc_{idx}"]   
            )

    print("All PredAI data has been successfully processed and stored in ChromaDB.")

# # Verify the data count in ChromaDB
# doc_count = collection.count()
# print(f"Total documents stored in ChromaDB: {doc_count}")