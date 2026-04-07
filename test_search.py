# test_search.py
# This proves that other files can import and use your database!

from database import search_vault

def run_test():
    # Pick a question related to one of the papers you downloaded
    question = "How are exoplanets detected?" 
    print(f" Searching for: '{question}'\n")

    # Call your handover function
    results = search_vault(question, n_results=2)

    # Extract the text and the metadata (the citations)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    # Print the results nicely
    for i in range(len(documents)):
        print(f"--- Match {i+1} ---")
        print(f" Source File: {metadatas[i]['source']}")
        print(f" Text Snippet: {documents[i][:250]}...\n")

if __name__ == "__main__":
    run_test()