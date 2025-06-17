import glob
import re
import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from google import genai
from llama_cpp import Llama

# process documents

def text_splitter(filePath,fileType):
    data = {}
    for file in glob.glob(f"{filePath}*.{fileType}"):
        doc = pymupdf.open(file)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            chunks = re.split("\\n",page.get_text())
            chunk_id = 0
            for chunk in chunks:
                if len(chunk.strip()) == 0:
                    continue
                if len(chunk.strip()) > 1000:
                    sentences = chunk.strip().split(".")
                    for sentence in sentences:
                        data[(file,page_num,chunk_id)] = sentence[:1000].strip()
                        chunk_id += 1
                else:
                    data[((file,page_num,chunk_id))] = chunk.strip()
                    chunk_id += 1
    return data


# Function to extract text from a PDF
def extract_text_from_pdf(filePath,fileType):
    data = {}
    for file in glob.glob(f"{filePath}*.{fileType}"):
        doc = pymupdf.open(file)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            if len(page.get_text().strip()) != 0:
                data[(file,page_num)] = page.get_text().strip()
    return data

# Extract text from all uploaded PDF files
# pdf_texts = text_splitter('','pdf')
pdf_texts = extract_text_from_pdf('../RAG Docs/','pdf')

# Display extracted text from each PDF file
# for pdf_file, text in pdf_texts.items():
#     print(f"--- {pdf_file} ---")
#     print(text[:100])  # Display the first 500 characters of each document
#     print("\n")
    # chunk text as necessary

# vectorize chunks



# Convert text documents to TF-IDF vectors
documents = list(pdf_texts.values())
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

# Create a FAISS index
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_vectors)

# get prompt

def search_documents(query, top_k=3, threshold=1.3):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vector, top_k)
    results = []
    for idx, i in enumerate(indices[0]):
        if distances[0][idx] <= threshold:
            results.append((documents[i],distances[0][idx]))
    # results = [(documents[i], distances[0][i]) for i in indices[0]]
    return results

# Example query
query = "Who has a dog?"
search_results = search_documents(query)
for doc, dist in search_results:
    print(doc)
    print(dist)

context = "\n".join([result[0] for result in search_results])
# print(f"Context: {context}\n\nQuestion: {query}")


# Initialize the Llama model with the path to your GGUF model file
# llm = Llama(model_path="/Users/alexanderratnasamy/Library/Application Support/nomic.ai/GPT4All/qwen2.5-coder-7b-instruct-q4_0.gguf")

# # Generate a completion
# output = llm(f"Context: {context}\n\nQuestion: {query}", max_tokens=128, temperature=0.7)

# print(output["choices"][0]["text"])


# with open("./API-KEY.key", "r") as file:
#     client = genai.Client(api_key=file.read().strip())


# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents=f"Context: {context}\n\nQuestion: {query}"
# )
# print(response.text)
