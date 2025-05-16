from rag_system import DatacenterRAG

def main():
    print("Initializing Datacenter RAG System...")
    rag = DatacenterRAG()
    print("\nWelcome to the Datacenter Site Selection Assistant!")
    print("You can ask questions about datacenter locations, site selection criteria, and more.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
            
        print("\nAssistant: ", end="")
        response = rag.query(user_input)
        print(response)

if __name__ == "__main__":
    main() 