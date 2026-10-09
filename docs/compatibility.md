# Compatibility matrix

| Contract | Required value | Owner | Status |
|---|---|---|---|
| Source model | `nomic-ai/nomic-embed-text-v1.5` | Embedserve | Preserved |
| API alias | `nomic-embed-text:v1.5` | Embedserve/SlackQuery | Preserved |
| `/api/tags` digest | `0a109f...e59f` (full 64-char value in config) | Compatibility config | Preserved |
| Native output | 768 raw floats | Embedserve | Preserved |
| Encoder normalization | Disabled | Embedserve | Preserved |
| Encoder sequence limit | 2048 tokens | Embedserve | Preserved |
| Document prefix | `search_document: ` | SlackQuery client | Preserved |
| Query prefix | `search_query: ` | SlackQuery client | Preserved |
| Stored dimension | First 512 dimensions | SlackQuery client | Preserved |
| Stored normalization | L2 after truncation | SlackQuery client | Preserved |
| Request ordering | Input order | Embedserve | Tested |
| Ollama endpoint | `/api/embed` | Embedserve | Supported subset |
| OpenAI endpoint | `/v1/embeddings` | Embedserve | Supported subset |

The model weight revision is pinned separately from the compatibility digest. This is
intentional: the deployed SlackQuery generation was created with the historical Ollama
digest while PyTorch loads Hugging Face commit
`e9b6763023c676ca8431644204f50c2b100d9aab` and reviewed remote code commit
`7710840340a098cfb869c4f65e87cf2b1b70caca`.

Synthetic fixture acceptance is predeclared as:

- identical vector count and 768 dimensions;
- finite values;
- component-wise `abs(new - baseline) <= 1e-6 + 1e-5 * abs(baseline)`;
- cosine similarity at least `0.999999`.

Exact SHA-256 vector matches are reported when achieved but are not required across
compatible CUDA kernels.
