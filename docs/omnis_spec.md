# Project Specification: Omnis AI Network Assistant

## 1. Goal

Build a small, locally-deployable language model that can autonomously diagnose network issues by calling NETSCOUT Omnis MCP tools. The model must learn to **chain multiple tool calls with explicit reasoning between each step** — the skill that frontier LLMs handle naturally but small open-source models struggle with.

The end result is an AI assistant that a NOC operator can ask "anything wrong on the network?" and get a multi-step investigation with a root-cause diagnosis and actionable recommendations — without ever calling OpenAI or Anthropic APIs at inference time.

---

## 2. Why this matters

Today, the Omnis MCP tools work well with frontier models (Claude, GPT-4). But NETSCOUT's enterprise customers — finance, defense, government — often cannot send network telemetry to cloud LLMs. A fine-tuned small model running on-premises or on a single GPU instance solves that constraint. Development cost is under $50. Runtime cost is under $5/month.

---

## 3. What the model must learn

### 3.1 Tool selection (the easy part)

Given a user question, the model must select the correct tool from a catalog of 31 Omnis MCP tools across 10 categories: discovery, health, performance, troubleshooting, traffic, protocol deep-dive, location, host analysis, topology, and advanced diagnostics.

### 3.2 Chained reasoning (the hard part)

The model must learn to execute multi-step diagnostic workflows where:

- It calls tool A, receives structured JSON results
- It **reasons explicitly** about what the results mean (the `<think>` tag pattern)
- It **extracts specific values** from tool A's result (an IP address, an application name, an error rate) and uses them as parameters for tool B
- It **cross-references** findings across multiple steps to reach a diagnosis
- It **recovers from dead ends** — when a tool result is inconclusive, it pivots to a different investigation approach

### 3.3 The 10 diagnostic workflows the model must master

| # | Workflow | Typical steps | What it teaches |
|---|---------|---------------|-----------------|
| 1 | **Health triage** | 2-3 | Broad scan → identify worst application → drill into it |
| 2 | **DNS investigation** | 3-4 | Anomaly detection → query-level analysis → host profiling → cascading effects |
| 3 | **Site performance** | 2-3 | Site-level metrics → per-application breakdown → TCP health at that site |
| 4 | **Application performance** | 2-3 | App summary → slow transaction identification → connection health |
| 5 | **Host investigation** | 2-4 | Activity profile → connection mapping → reachability/blast radius |
| 6 | **Cross-site path analysis** | 2-3 | Topology discovery → path tracing → per-hop latency analysis |
| 7 | **Bandwidth analysis** | 2 | Top talkers → detailed profile of biggest consumer |
| 8 | **MySQL troubleshooting** | 3 | Healthy-looking average hiding slow tail → server-side root cause |
| 9 | **Security scan** | 2-3 | Anomaly detection → DNS analysis for DGA → reachability assessment |
| 10 | **Wrong path + recovery** | 3 | Try approach A → inconclusive → pivot to approach B |

---

## 4. The training data

### 4.1 Source

Training data is generated synthetically using the Anthropic API (Claude Sonnet). Each example is a complete multi-turn conversation showing the model calling tools, receiving results, reasoning, and continuing the investigation.

Tool responses in the training data are grounded in real environment data from a live Omnis deployment (European enterprise WAN: Francfort DC hub + 7 branch offices, 171 hosts, 640 connections, 11 monitored applications).

### 4.2 The `<think>` tag pattern

Between each tool call, the model produces structured reasoning inside `<think>` tags:

```
<think>
DNS shows 20% error rate — 87K failures out of 433K transactions.
All other applications are below 1%. I should investigate DNS anomalies.
</think>
<tool_call>
{"name": "get_anomalies", "arguments": {"application": "DNS"}}
</tool_call>
```

This gives small models a learnable "scratchpad" for reasoning before acting.

### 4.3 Dataset composition

| Type | % of dataset | Purpose |
|------|-------------|---------|
| 2-3 step diagnostic chains | 40% | Standard investigation workflows |
| 3-4 step deep investigations | 15% | Complex multi-tool reasoning |
| Wrong path + recovery chains | 5% | Learning to pivot when results are unclear |
| Single-turn tool selection | 40% | Basic tool selection accuracy |

### 4.4 Question personas

Questions are generated across 5 personas to ensure diversity:
- **NOC operator**: Short, informal, wants fast answers ("any issues?", "DNS ok?")
- **Network engineer**: Technical, references IPs and protocols
- **IT manager**: Business-oriented, wants summaries and trends
- **Security analyst**: Focused on anomalies and threats
- **Application owner**: Focused on their specific service

### 4.5 Seed examples

A set of 7-10 hand-crafted gold-standard chain examples that set the quality bar. These are manually written with perfect reasoning bridges, correct parameter extraction between steps, and actionable diagnoses. They serve as both training data and quality reference.

---

## 5. The base model and fine-tuning approach

### 5.1 Base model: Qwen 2.5 7B Instruct

- 7 billion parameters, strong baseline tool-calling capability
- Fits in 24GB VRAM with 4-bit quantization (QLoRA)
- Good balance between capability and inference speed

### 5.2 Fine-tuning method: QLoRA (4-bit quantized LoRA)

- LoRA rank 16, alpha 32 (starting point — autoresearch will optimize)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Training format: ChatML with `<think>` tags
- Framework: Unsloth (optimized for single-GPU fine-tuning)
- Training time per run: ~6-8 minutes on A10G GPU

### 5.3 Infrastructure: AWS EC2

- Instance type: **g5.2xlarge** (1x NVIDIA A10G, 24GB VRAM, 32GB RAM)
- Pricing: ~$0.40/hr spot, ~$1.21/hr on-demand
- Region: eu-west-1 (Ireland) or eu-central-1 (Frankfurt)
- Storage: S3 bucket for training data and model artifacts
- Estimated cost per overnight autoresearch run: ~$3-5 on spot

---

## 6. The autoresearch loop

### 6.1 Concept (inspired by Karpathy's autoresearch)

An autonomous optimization loop where an AI agent:
1. Modifies training configuration (hyperparameters, data mix, prompt templates)
2. Runs a fine-tuning experiment (~6-8 minutes)
3. Evaluates against a fixed benchmark
4. Keeps the configuration if accuracy improved, discards if not (ratcheting)
5. Repeats indefinitely overnight

### 6.2 What is mutable (the agent can change)

- Training hyperparameters: learning rate, LoRA rank, epochs, batch size, warmup steps
- Data mix: weighting between chain vs single-turn examples
- Prompt template: `<think>` tag format, system prompt wording
- Training schedule: cosine vs linear decay, gradient accumulation steps

### 6.3 What is immutable (never changes)

- The evaluation benchmark (50 questions with expected answers)
- The evaluation script (measures accuracy deterministically)
- The base model (always starts from Qwen 2.5 7B Instruct)
- The training data content (only weighting/mixing changes)

### 6.4 Primary metric: composite accuracy score

A weighted score combining:
- **Tool selection accuracy** (40%): Did the model pick the right tool?
- **Argument accuracy** (25%): Did it pass the correct parameters?
- **Chain continuation rate** (20%): Did the model continue investigating (vs stopping after one tool)?
- **Reasoning relevance** (15%): Does the `<think>` block reference actual values from the tool result?

### 6.5 Expected throughput

- ~60-80 experiments per 8-hour overnight run on g5.2xlarge spot
- Each experiment: modify config → fine-tune (~6 min) → evaluate (~2 min) → log → next

---

## 7. Evaluation benchmark

### 7.1 Structure

50 canonical questions spanning all 10 workflows. Each question has:
- The user question text
- The expected first tool call (name + arguments)
- For chain questions: the expected second tool call (possibly with a placeholder for a value extracted from the first result)
- The minimum chain length expected
- Difficulty rating (easy / medium / hard)

### 7.2 Example benchmark entries

```
Question: "How is the network doing?"
Expected tool: get_health_overview
Expected args: {}
Min chain length: 2
Difficulty: easy

Question: "DNS seems broken, investigate"
Expected tool: get_anomalies
Expected args: {"application": "DNS"}
Min chain length: 3
Difficulty: hard

Question: "What is host 10.0.3.22 doing?"
Expected tool: get_client_activity_summary
Expected args: {"client_ip": "10.0.3.22"}
Min chain length: 2
Difficulty: medium
```

### 7.3 Evaluation method

The evaluation script:
1. Loads the fine-tuned model in inference mode
2. Sends each benchmark question
3. Parses the model's first output for `<tool_call>` tags
4. Compares tool name and arguments against expected values
5. If chain question: feeds back a synthetic tool result and checks whether the model continues with a second tool call
6. Computes the composite accuracy score
7. Outputs a single number (0.0 to 1.0) that autoresearch uses for keep/discard decisions

---

## 8. Deployment

### 8.1 Model export

After autoresearch finds the best configuration, the final model is:
1. Fine-tuned one last time with the optimal config
2. Merged (LoRA adapters into base model)
3. Exported to GGUF format (Q4_K_M quantization) for Ollama

### 8.2 Runtime options

**Option A — Local laptop via Ollama:**
The GGUF model runs on any machine with 8GB+ RAM. Inference speed: ~10-20 tokens/sec on CPU, faster with GPU.

**Option B — EC2 instance:**
Deploy on a small GPU instance (g5.xlarge, ~$1/hr) alongside the Omnis lab environment. The model serves as a local API that the MCP orchestrator calls.

### 8.3 Integration with MCP

At inference time, the deployed model:
1. Receives a user question
2. Produces `<think>` reasoning + `<tool_call>` output
3. An orchestration layer parses the `<tool_call>`, calls the real Omnis MCP server, and feeds the result back
4. The model continues reasoning and calling tools until it produces a final answer (no `<tool_call>` tag)

This orchestration layer is a simple loop — not part of the model training.

---

## 9. Target metrics

| Metric | Target | How measured |
|--------|--------|-------------|
| Tool selection accuracy | >90% | Benchmark: correct tool name on first call |
| Argument accuracy | >85% | Benchmark: correct parameters |
| Chain continuation rate | >75% | Benchmark: model calls 2+ tools when expected |
| Reasoning relevance | >70% | Benchmark: `<think>` block contains data values from tool result |
| No hallucinated data | 100% | Manual review: model never invents numbers |
| Inference speed | >5 tok/s | Measured on deployment target |

---

## 10. Cost summary

| Phase | Estimated cost | Notes |
|-------|---------------|-------|
| Training data generation (Claude Sonnet API) | $10-15 | ~350 examples |
| AWS g5.2xlarge — initial fine-tune + testing | $2-3 | ~2 hours on-demand |
| AWS g5.2xlarge — overnight autoresearch | $3-5 | ~8 hours on spot |
| Additional autoresearch runs (if needed) | $3-5 per night | Usually 1-2 nights enough |
| **Total** | **$20-30** | |

---

## 11. File structure (for Claude Code reference)

```
omnis-sft/
├── data/                          # Training data generation
│   ├── tool_catalog.json          # All 31 MCP tool definitions + environment context
│   ├── seed_chains.json           # Hand-crafted gold-standard examples
│   ├── generate_chains.py         # Synthetic data generator (uses Anthropic API)
│   └── [generated outputs]        # .jsonl and .json training files
├── training/                      # Fine-tuning
│   ├── finetune.py                # Unsloth QLoRA training script
│   ├── config.json                # Hyperparameters (mutable by autoresearch)
│   └── requirements.txt           # Python dependencies
├── eval/                          # Evaluation
│   ├── benchmark.json             # 50 canonical test questions
│   └── eval_runner.py             # Automated accuracy measurement
├── autoresearch/                  # Autonomous optimization
│   ├── program.md                 # Agent instructions
│   ├── runner.py                  # Experiment loop orchestrator
│   └── constants.py               # Immutable constraints
├── deploy/                        # Model export and deployment
│   ├── Modelfile                  # Ollama configuration
│   └── export_gguf.py             # GGUF export script
├── infra/                         # AWS infrastructure
│   ├── setup_instance.sh          # EC2 bootstrap (CUDA, Python, Unsloth)
│   └── upload_to_s3.sh            # Data transfer helper
└── CLAUDE.md                      # Project instructions for Claude Code
```

---

## 12. Omnis MCP tools reference

The model has access to 31 tools. Here is the complete catalog organized by category.

### Discovery (3 tools)

**list_applications** — List all monitored applications with data counts.
No required parameters.

**list_tables** — List ClickHouse database tables, optionally filtered by category (f=facts, d=dimensions, a=aggregates, s=staging).
Optional: `category` (string)

**describe_table** — Get schema details for a specific table.
Required: `table_name` (string)

### Health (2 tools)

**get_health_overview** — Cross-application health: total transactions, errors, avg latency, per-application breakdown.
Optional: `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`, `limit`

**get_application_summary** — Summary metrics for one application: transaction counts, error rates, latency stats, top endpoints.
Required: `application` (string). Optional: `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`, `limit`

### Performance (2 tools)

**get_performance_metrics** — Latency distribution (P50/P90/P95/P99), throughput over time, error breakdown for an application.
Required: `application`. Optional: `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

**get_connection_health** — TCP connection health: setup times, resets, retries, failure rates.
Required: `application`. Optional: `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

### Troubleshooting (3 tools)

**get_errors** — Failed transactions and error details, summarized by error type.
Optional: `application`, `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`, `page`, `page_size`

**get_slow_transactions** — Transactions with response times above threshold.
Optional: `application`, `threshold_ms` (default 1000), `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`, `page`, `page_size`

**get_anomalies** — Detect unusual patterns: high error rates, latency spikes, single-occurrence domains (DGA), unusual ports.
Optional: `application`, `hours`, `limit`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

### Traffic (4 tools)

**get_top_talkers** — Top traffic pairs by total bytes transferred.
Optional: `limit`

**get_traffic_overview** — Overall traffic patterns: hourly distribution, top clients, top servers.
Optional: `hours`, `limit`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

**get_flow_summary** — Aggregated flow statistics with optional filters.
Optional: `client_ip`, `server_ip`, `application`, `hours`, `limit`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

**get_flow_details** — Raw flow records. Requires at least one of: client_ip, server_ip, application.
Optional: `client_ip`, `server_ip`, `application`, `hours`, `page`, `page_size`, `sensor_ip`, `sensor_name`

### Protocol deep-dive (2 tools)

**get_dns_queries** — DNS query records. Requires at least one filter.
Optional (but at least one required): `client_ip`, `server_ip`, `domain_contains`, `response_code`, `hours`, `page`, `page_size`, `sensor_ip`, `sensor_name`

**get_http_requests** — HTTP request records. Requires at least one filter.
Optional (but at least one required): `client_ip`, `server_ip`, `host_contains`, `uri_contains`, `status_code`, `hours`, `page`, `page_size`, `sensor_ip`, `sensor_name`

### Location (3 tools)

**get_site_summary** — Metrics for a specific WAN site.
Required: `site_name`. Optional: `hours`, `limit`, `sensor_ip`, `sensor_name`

**get_community_summary** — Metrics for a named IP community.
Required: `community_name`. Optional: `hours`, `limit`, `sensor_ip`, `sensor_name`

**resolve_location** — Smart resolver: searches sites, communities, and host geo for a location name.
Required: `location_name`. Optional: `hours`

### Host analysis (2 tools)

**get_client_activity_summary** — What applications and servers a client IP communicated with.
Required: `client_ip`. Optional: `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

**get_server_activity_summary** — What clients connected to a server and which applications were used.
Required: `server_ip`. Optional: `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

### Topology (7 tools)

**get_topology_summary** — Summary counts: hosts, connections, breakdown by site and application.
No required parameters.

**get_topology_schema** — Neo4j graph schema: node labels, relationship types, property keys.
No required parameters.

**get_topology_by_site** — All hosts and connections for a specific site.
Required: `site_name`

**get_topology_by_application** — All connections using a specific application/protocol.
Required: `application`. Optional: `limit`

**get_host_connections** — All connections for a host (as client and as server).
Required: `ip`

**get_host_reachability** — Find all hosts reachable from a given host within N hops.
Required: `ip`. Optional: `max_hops` (1-5, default 3)

**find_shortest_path** — Shortest network path between two hosts.
Required: `from_ip`, `to_ip`

**query_topology** — Custom read-only Cypher query on the Neo4j graph.
Required: `cypher` (string)

### Advanced diagnostics (2 tools)

**get_session_hop_analysis** — Trace a session across multiple sensors. Identify where latency occurs.
Required: `client_ip`, `server_ip`, `server_port`. Optional: `application`, `hours`, `site_name`, `community_name`, `sensor_ip`, `sensor_name`

**execute_query** — Custom read-only ClickHouse SQL query.
Required: `sql`. Optional: `parameters`, `page`, `page_size`, `explain`

---

## 13. Live environment context

The training data and evaluation are grounded in this real environment:

**Topology**: Hub-and-spoke architecture, 171 hosts, 640 connections.

**Sites**: Francfort DC (hub, 16 hosts), Paris (2), London (1), Madrid (2), Athens (2), Vienna (2), Zurich (2), Prague (1).

**Applications**: DNS (433K txn, 20% error rate), MYSQL (403K txn, 0% errors, 8ms avg), SSH (138K txn, 93ms avg), Web Store (78K txn, 106ms avg), Shop Backend API (44K txn, 60ms avg), NTP, ICMP, DHCP, Generic, HTTPS.

**Known issues for training data realism**:
- DNS: 100% failure rate from 10.0.3.20, .21, .22 and 10.1.6.10 to Google DNS 8.8.8.8 (firewall/routing block)
- DNS: Internal server 10.0.2.100 showing 50-87ms latency spikes (14x standard deviation) — likely from retry storms
- MySQL: Server 10.0.5.20 showing 1100ms average for slow queries from app servers 10.0.4.10/11/12
- Web Store: P95 latency 285ms, P99 399ms — worse for branch offices due to WAN distance
- Bandwidth: 10.0.4.12 consuming 57MB from Fastly CDN (151.101.2.132)

**Key server IPs**: 10.0.2.100 (DNS), 10.0.5.20 (MySQL), 10.0.2.10 (Web Store), 151.101.2.132 (Fastly CDN), 8.8.8.8 (Google DNS — blocked)

---

## 14. Key design decisions (agreed)

1. **SFT only, no pretraining** — We ride on Qwen 2.5 7B's existing language and reasoning capabilities. We only teach it which Omnis tools to use and how to reason between calls.

2. **Keep all 31 tools** (not reduce to 2) — Tool selection accuracy is >90% achievable with 31 well-defined tools. The 2-tool SQL-only approach would require the model to generate complex ClickHouse SQL against 145-column tables with cryptic names, which is much harder for a 7B model.

3. **`<think>` tag pattern for reasoning** — Structurally simple, learnable by small models, provides explicit scratchpad for inter-step reasoning.

4. **Autoresearch for hyperparameter optimization** — Inspired by Karpathy's autoresearch framework. Ratcheting mechanism (keep improvements, discard regressions) over 60-80 overnight experiments.

5. **G5 spot instance (A10G)** — Best cost/performance ratio. $3-5 per overnight run. Bigger GPUs (A100, H100) give diminishing returns for this workload — they're faster but the 60-80 experiment sweet spot is already reachable on A10G.

6. **ChatML format with Unsloth** — Matched to Qwen 2.5's native chat template. Unsloth provides the fastest single-GPU QLoRA fine-tuning.

7. **Ollama GGUF for deployment** — Universal deployment: runs on laptop (CPU), on EC2 (GPU), or alongside the Omnis lab. No API dependency at inference time.
