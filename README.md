**Directions**

1. Create a folder called "conceptnet_embeddings" and then download the ConceptNet numberbatch embeddings pickle into that folder from https://huggingface.co/datasets/metaphorreasoning5632/conceptnet-embeddings.

2. Open a virtual environment and run ``pip install -r requirements.txt``

3. Assign the OPENAI_API_KEY and DEEPSEEK_API_KEY environment variables to your API keys. This is would incur and API cost. To only test open models, comment out references to these models within ask_llms_vllm.py.

4. Run the run_everything.sh file to run all experiments in the paper.

