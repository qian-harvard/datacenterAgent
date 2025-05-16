# Datacenter Location Assistant

An AI-powered assistant that helps users find optimal datacenter locations based on various criteria.

## Features

- Interactive chat interface for datacenter location queries
- Powered by OpenAI's GPT-4
- Real-time location suggestions with map links
- Detailed information about existing datacenters and potential locations

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```
4. Run the application:
   ```bash
   python datacenterAgent/app.py
   ```

## Deployment to Gradio Spaces

1. Push your code to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Click "Create new Space"
4. Choose "Gradio" as the SDK
5. Connect your GitHub repository
6. Set your OpenAI API key as a Space Secret:
   - Go to Space Settings
   - Navigate to "Repository Secrets"
   - Add a new secret with key `OPENAI_API_KEY` and your API key as the value
7. Deploy!

## Project Structure

```
datacenterAgent/
├── app.py              # Main Gradio application
├── rag_system.py       # RAG system implementation
├── data/              # Data files
│   ├── us_datacenters.csv
│   └── us_possible_locations.csv
└── prompts/           # Prompt templates
    └── datacenter_site_selector.txt
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## License

MIT License 