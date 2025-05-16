import gradio as gr
from rag_system import DatacenterRAG
import os
import openai

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
solvtra_logo_path = os.path.join(current_dir, "solvtra.png")

def initialize_rag(api_key_input):
    """Initialize the RAG system with the provided API key"""
    if not api_key_input or api_key_input.strip() == "":
        return "Please enter your OpenAI API key.", None
    
    api_key = api_key_input.strip()
    print(f"[app.py] Received API Key for init: {'sk-...' + api_key[-4:] if len(api_key) > 7 and api_key.startswith('sk-') else 'Key format potentially incorrect'}")

    try:
        # Set for OpenAI client globally
        openai.api_key = api_key
        
        print(f"[app.py] Testing API key with model 'gpt-4o-mini'...")
        # Test API key with a simple call
        test_response = openai.chat.completions.create(
            model="gpt-4o-mini", # Consistent with RAG system
            messages=[{"role": "user", "content": "Hello"}], # Simple test prompt
            max_tokens=5
        )
        print(f"[app.py] API key test successful. Test response: {test_response.choices[0].message.content}")
        
        # If we get here, the key works
        print(f"[app.py] Initializing DatacenterRAG with API key...")
        rag_system = DatacenterRAG(api_key=api_key) # Pass the key
        print("[app.py] DatacenterRAG initialized successfully.")
        return "RAG system initialized successfully!", rag_system

    except openai.APIAuthenticationError:
        error_msg = "OpenAI API Key Error: Invalid API Key. Please check your key and try again."
        print(f"[app.py] OpenAI API AuthenticationError: {error_msg}")
        return error_msg, None
    except openai.RateLimitError:
        error_msg = "OpenAI API Error: Rate limit exceeded. Please try again later."
        print(f"[app.py] OpenAI API RateLimitError: {error_msg}")
        return error_msg, None
    except openai.APIConnectionError:
        error_msg = "OpenAI API Error: Could not connect to OpenAI. Please check your network."
        print(f"[app.py] OpenAI API ConnectionError: {error_msg}")
        return error_msg, None
    except openai.NotFoundError as e: # e.g. model not found
        error_msg = f"OpenAI Error: Model ('gpt-4o-mini' or embeddings) not found or API issue: {str(e)}"
        print(f"[app.py] OpenAI API NotFoundError: {error_msg} - {str(e)}")
        return error_msg, None
    except Exception as e:
        # This will catch errors from DatacenterRAG init too, or other unexpected OpenAI errors
        error_msg = f"Error initializing RAG system: {type(e).__name__} - {str(e)}"
        print(f"[app.py] Error during RAG initialization: {error_msg}")
        return error_msg, None

def respond(message, chat_history, rag_system):
    """Respond to user messages"""
    if not rag_system:
        return "Please initialize the RAG system first by entering your OpenAI API key.", chat_history
    
    try:
        response = rag_system.query(message)
        chat_history.append((message, response))
        return "", chat_history
    except Exception as e:
        error_message = f"Error during query: {type(e).__name__} - {str(e)}"
        print(f"[app.py] Error during RAG query: {error_message}")
        chat_history.append((message, error_message))
        return "", chat_history

# Custom CSS for better styling
custom_css = """
#main-logo {
    display: block;
    margin: 0 auto;
    max-width: 200px;
    padding: 20px 0;
}
.title-container {
    text-align: center;
    width: 100%;
    margin: 20px 0;
}
.title-container h1 {
    margin: 0;
    padding: 0;
}
.powered-by {
    text-align: center;
    margin: 20px 0;
    font-size: 1.1em;
    color: #666;
    width: 100%;
    position: relative;
    bottom: 0;
}
.powered-by a {
    text-decoration: none;
    color: #00008B;  /* Dark blue color */
    font-weight: 500;
}
.powered-by a:hover {
    color: #0000CD;  /* Slightly lighter blue on hover */
}
.chatbot {
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.disclaimer {
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
    text-align: center;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    # Main logo at the top
    try:
        gr.Image(
            solvtra_logo_path,
            elem_id="main-logo",
            show_label=False,
            show_download_button=False,
            show_fullscreen_button=False,
            show_share_button=False
        )
    except Exception as e:
        print(f"Error loading main logo: {e}")
        gr.Markdown("### Solvtra")
    
    gr.Markdown("<div class='title-container'><h1>Datacenter Location Assistant</h1></div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API key here...",
                type="password"
            )
            gr.Markdown("<div class='disclaimer'>Your API key is used only during this session and never stored.</div>")
            init_button = gr.Button("Initialize RAG System", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(
        placeholder="Ask a question about datacenter locations...",
        show_label=False
    )
    clear = gr.Button("Clear", variant="secondary")
    
    # Powered by section at the bottom
    gr.Markdown(
        """
        <div class="powered-by">
            Powered by <a href="https://power-agent.github.io/" target="_blank">PowerAgent</a>
        </div>
        """
    )
    
    rag_system = gr.State(None)
    
    init_button.click(
        initialize_rag,
        inputs=[api_key],
        outputs=[status, rag_system]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, rag_system],
        outputs=[msg, chatbot]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
