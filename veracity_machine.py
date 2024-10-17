import time
import os
import base64
import io
import mesop as me
import google.generativeai as genai
import chromadb
import PyPDF2  # For extracting text from PDF files

os.environ['GOOGLE_API_KEY'] = 'your_api_key'

# Initialize API and DB
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("my_collection")

# Create the model
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
            display_pdf_text()  # Show the extracted PDF text
            chat_input()
            output()
            db_input()
            db_output()
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
    if state.pdf_text:
        print(f"Extracted PDF Text: {state.pdf_text[:1000]}")  # Display first 1000 characters

def display_pdf_text():
    """Display the extracted PDF text."""
    state = me.state(State)
    if state.pdf_text:
        with me.box(style=me.Style(padding=me.Padding.all(10))):
            me.text("Extracted PDF Content:")
            me.text(state.pdf_text)  # Display the entire text

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
    print(f"Combined input for AI:\n{combined_input}")

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

def call_api(input_text):
    # Query the database for relevant context
    results = collection.query(
        query_texts=[input_text],  # Replace this input with relevant conditions for searching
        n_results=2
    )

    context = " ".join(results['documents'][0]) if results['documents'] else ""
    print(context)
    # Add context to the prompt
    full_prompt = f"Context: {context}\n\nUser: {input_text}"
    
    response = chat_session.send_message(full_prompt)
    time.sleep(0.5)
    yield response.text

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
                placeholder="Enter text to add to the database",
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

    # Add to Chroma database
    collection.add(
        documents=[state.db_input],
        ids=[f"doc_{int(time.time())}"]  # Using timestamp as a simple unique ID
    )

    state.db_output = f"Added to database: {state.db_input[:50]}..."
    state.db_input = ""
    yield

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
