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
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDYzYQNtdff7sxg23uUpVavrxl9mNe_1CY'

# Initialize API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("my_collection", metadata={"hnsw:space": "cosine"})

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
                placeholder="Enter your customized prompt, or will do default FCOT if empty.",
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
    if not state.input.strip():  # Check if input is empty or contains only whitespace
        state.input = "Default input text here."  # Default FCT prompt if no input is provided
        return
    
    state.in_progress = True
    input_text = state.input
    top_100_statements = get_top_100_statements(input_text)
    fct_prompt = generate_fct_prompt(input_text)
    combined_input = combine_pdf_and_prompt(fct_prompt, state.pdf_text)  # Combine prompt with PDF text

    state.input = ""
    yield

    for chunk in call_api(combined_input):
        state.output += chunk
        yield

    state.output += '\n\n' + 'The probability of the statement truthness:\n\n' + str(top_100_statements) 
    state.output += '\n\n' + 'Analysis completed using the provided or default settings.'
    state.in_progress = False
    yield

# Fractal COT & Function Call
# Define the complex objective functions
frequency_heuristic = [
    {"description": "Repetition Analysis", "details": "Analyzing wider coverage helps assess consensus. If multiple independent sources confirm manipulation, it strengthens the claim. Even widespread agreement about deceptive editing wouldn't automatically justify government action against CBS. The First Amendment protects against content-based restrictions."},
    {"description": "Origin Tracing", "details": "Confirmed sources are critical. Discrepancies between reporting and original sources raise red flags."},
    {"description": "Evidence Verification", "details": "Expert analysis is the most crucial element. Expert opinions on video manipulation are essential. Expert testimony would be necessary for any legal action alleging manipulation, though such a case would face significant First Amendment hurdles."}
]

misleading_intentions = [
    {"description": "Omission Checks", "details": "Assess the omissions' impact. Did they distort the message? Did they create a demonstrably false representation? (Proving this is difficult)."},
    {"description": "Exaggeration Analysis", "details": "Evaluate the 'scandal' claim. Does the evidence support it, or is it hyperbole? Does the situation, even if accurately reported, justify calls for license revocation under existing legal and constitutional frameworks?"},
    {"description": "Target Audience Assessment", "details": "Analyze audience manipulation. Identify targeting tactics (language, framing). While such tactics can be ethically questionable, they are generally protected speech unless they involve provable falsehoods and meet the very high legal bar for defamation or incitement."}
]

def generate_fct_prompt(input_text, iterations=3):
    prompt = f"Initial Evaluation: {input_text}\n\n"
    for i in range(1, iterations + 1):
        prompt += f"Iteration {i}: Evaluate the text based on the following objectives:\n"
        prompt += "\nFrequency Heuristic:\n"
        for fh in frequency_heuristic:
            prompt += f"{fh['description']}: {fh['details']}\n"
        prompt += "\nMisleading Intentions:\n"
        for mi in misleading_intentions:
            prompt += f"{mi['description']}: {mi['details']}\n"
        prompt += "\nProvide a percentage score and explanation for each objective function ranking.\n\n"
    prompt += "Final Evaluation: Return a veracity score for the text."
    return prompt

def combine_pdf_and_prompt(prompt: str, pdf_text: str) -> str:
    """Combines the user's prompt with the extracted PDF text."""
    if not pdf_text:
        return prompt
    return f"Prompt: {prompt}\n\nPDF Content: {pdf_text}"  # Combine entire text

def chunk_pdf_text(pdf_text: str) -> list[str]:
    if not pdf_text:
        return []
        
    prompt = """Split the following text into logical chunks (max 10 chunks) of reasonable size. 
    Preserve complete paragraphs and maintain context. Return ONLY the chunks as a numbered list, with no additional text.
    Format each chunk like:
    1. [chunk text]
    2. [chunk text]
    etc.

    Text to split:
    {text}
    """

    response = chat_session.send_message(prompt.format(text=pdf_text))
    chunks_text = response.text.strip()
    
    # Split on numbered lines and clean up
    chunks = []
    for line in chunks_text.split('\n'):
        # Skip empty lines
        if not line.strip():
            continue
        # Remove the number prefix and clean whitespace
        chunk = line.split('.', 1)[-1].strip()
        if chunk:
            chunks.append(chunk)
            
    return chunks

# Sends API call to GenAI model with user input
def call_api(input_text):
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

    collection.add(
                documents=[state.db_input],         
                metadatas=[{"source": 'manual', "row_index": 'none'}],            
                ids=[f"doc_{int(time.time())}"]   
            )
    
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

def get_top_100_statements(user_input):
    # Query ChromaDB for top 100 similar inputs based on cosine similarity
    results = collection.query(
        query_texts=[user_input],
        n_results=100,
        include=["documents"],
        where = {"source": {"$in": ["train", "test", "validate"]}}
    )

    #store and return the number of each 100 statements in dictionary
    statement_dic = {}
    for i in range(100):
        statement = results['documents'][0][i].split(', ')[2]
        if statement not in statement_dic:
            statement_dic[statement] = 1
        else:
            statement_dic[statement] += 1   
    return statement_dic


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