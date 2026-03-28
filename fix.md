# Change interface
# Instructions: Refactor WijaiWai System to Pinecone & Google OAuth

I need you to refactor my current project, **WijaiWai** (an AI Research Assistant). The goal is to migrate from local embedding storage to **Pinecone** and implement **Google OAuth** for user management.

## 1. Technical Requirements

### Vector Database & RAG Migration
- **Current State:** The system uses local embedding generation and local storage (Chroma).
- **Target State:** - Use **Pinecone** for vector storage and similarity search.
    - Implement a workflow to upsert document embeddings into a specific index.
    - Ensure retrieval-augmented generation (RAG) queries the Pinecone index before sending context to the LLM.
- **Index Name:** `wijaiwai`

### User Management & Authentication
- **OAuth:** Implement **Google OAuth 2.0** for user sign-in.
- **Local Metadata:** Store user profile information (User ID, Email, Name) in a **SQLite** database after a successful Google login.
- **Data Isolation:** When querying Pinecone, use the User ID (from Google/SQLite) as a `namespace` or `metadata filter` to ensure users only access their own research data.

### LLM Integration
- **Model:** Use **OpenThaiGPT**.
- **Endpoint:** `http://thaillm.or.th/api/openthaigpt/v1/chat/completions`
- Remark : Already implement , Skip

## 2. Environment Variables (.env)
Please use the following configurations for the implementation:




Based on the requirements for the **WijaiWai** research assistant and the constraints of the Pinecone Free Quota (Starter Plan), the following configuration is recommended for optimal performance with Thai language processing.

## 1. Core Index Settings
    -Metric : Cosine
    -Dimension : 768
    -Model : intfloat/multilingual-e5-large

---

## 2. Multi-Tenancy Strategy (Google OAuth Integration)
Since the Free Tier is limited to **1 Index**, use **Namespaces** to isolate research data for each user:

* **Logic:** Assign the Google User ID (e.g., `google_102xxxx`) as the `namespace`.
* **Security:** This ensures that User A's uploaded papers never appear in User B's search results.
* **Performance:** Querying within a specific namespace is faster than global filtering on the free plan.

---

## UI For login 
    - Splash screen with Wijaiwai logo
    - log in button with Google logo 
    - Neat and clean

Final  
  -Clean and remove the old database system related files
