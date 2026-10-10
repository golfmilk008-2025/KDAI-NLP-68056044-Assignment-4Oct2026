# RAG: Baseline และ Improved

งานนี้เปรียบเทียบระบบ RAG สองระบบด้วยเอกสาร Markdown 4 ไฟล์ และคำถาม 18 ข้อ ใช้ Jupyter ในเครื่อง โดยค้นหาเอกสารในเครื่องและใช้ Qwen ใน Ollama สร้างบริบท และใช้ Qwen เดียวกันใน Ollama ตอบคำถาม

## Tech Stack

```text
Jupyter Notebook (Python)
 ├─ LlamaIndex → แบ่ง Markdown ตามหัวข้อ
 ├─ Ollama
 │   ├─ qwen3:4b-instruct → สร้างบริบทและคำตอบสุดท้าย
 │   └─ bge-m3 → Embedding ของเอกสารและคำถาม
 ├─ Transformers → tokenizer ของ BGE-M3 สำหรับแบ่ง chunks
 ├─ Sentence Transformers
 │   └─ BGE reranker → CrossEncoder เฉพาะ Improved
 ├─ ChromaDB → ค้นหาเวกเตอร์ด้วย cosine
 ├─ SQLite FTS5 → ค้นหา BM25
 │   └─ PyThaiNLP newmm → แบ่งคำไทยเฉพาะ Improved
 └─ urllib.request → เรียก Ollama API ในเครื่อง
```

| เทคโนโลยี | หน้าที่ในงาน | ทำงานที่ไหน |
|---|---|---|
| Python + JupyterLab | เปิด Notebook และรันตามลำดับเซลล์ | ในเครื่อง |
| LlamaIndex (`llama-index-core`) | ใช้ `MarkdownNodeParser` แบ่งข้อความตามหัวข้อ | ในเครื่อง |
| Ollama `qwen3:4b-instruct` | สร้างบริบทและตอบคำถามผ่าน `/api/chat` ใช้ทั้งสองระบบ | Ollama ในเครื่อง |
| Ollama `bge-m3` | แปลงข้อความและคำถามเป็นเวกเตอร์ผ่าน `/api/embed` ใช้ทั้งสองระบบ | Ollama ในเครื่องที่ `http://localhost:11434` |
| Transformers | โหลดเฉพาะ tokenizer ของ `BAAI/bge-m3` เพื่อนับ tokens และแบ่ง chunks | Python ในเครื่อง; ดาวน์โหลด tokenizer ครั้งแรก |
| Sentence Transformers + PyTorch | รัน BGE reranker ผ่าน CrossEncoder เฉพาะ Improved | Python ในเครื่อง; auto เลือก CUDA → MPS → CPU; ดาวน์โหลดน้ำหนัก reranker ครั้งแรก |
| `BAAI/bge-reranker-v2-m3` | ให้คะแนนคู่คำถาม–chunk เฉพาะ Improved ผ่าน `CrossEncoder` | ในเครื่อง |
| ChromaDB | เก็บเวกเตอร์ด้วย `PersistentClient` และค้นหาด้วย cosine | ในเครื่อง: `output/chroma/` |
| SQLite FTS5 (`sqlite3` ของ Python) | สร้างดัชนีข้อความและค้นหาด้วย BM25 | ในหน่วยความจำ; สร้างใหม่เมื่อ Run All |
| PyThaiNLP | แบ่งคำไทยด้วย `newmm` ก่อน BM25 ทั้งเอกสารและคำถามใน Improved | ในเครื่อง |
| python-dotenv | โหลดค่าตั้งอุปกรณ์ rerankerจาก `.env` โดย environment ที่ตั้งไว้มีลำดับความสำคัญกว่า | ในเครื่อง |

เวอร์ชัน dependencies ระบุใน [requirements.txt](requirements.txt) tokenizer และ reranker ตรึง revision ใน Notebook ส่วน embedding และโมเดลสร้างบริบทบันทึกเวอร์ชัน Ollama, model digest และ quantization ในผลแต่ละรัน คำตอบสุดท้ายใช้ `qwen3:4b-instruct` ใน Ollama เหมือนกันทั้งสองระบบ บันทึก backend และ digest ในผลรัน

## ระบบทำงานอย่างไร

**เตรียมข้อมูล:** อ่านเอกสาร → แบ่ง chunks → ส่งเอกสารเต็มและ chunk ให้ Qwen `qwen3:4b-instruct` ใน Ollama สร้างบริบท → ทำ embedding ข้อความ chunk รวมบริบทด้วย Ollama BGE-M3 → นำเข้า ChromaDB และ SQLite FTS5

**ค้นและตอบ:** คำถาม → ค้น Vector และ BM25 → รวมอันดับด้วย RRF → reranking เฉพาะ Improved → เลือกหลักฐาน 5 chunks → ส่งหลักฐานและคำถามให้ Qwen `qwen3:4b-instruct` ผ่าน Ollama ตอบภาษาไทยพร้อมอ้างอิง `[D1]`, `[D2]`

ทั้งสองระบบใช้งบข้อความหลักฐาน 7,000 ตัวอักษร จึงอาจตัดข้อความหรือใช้หลักฐานไม่ครบทั้ง 5 chunks เมื่อถึงงบ บริบทที่สร้างไว้มี cache สำหรับ backend, model digest และ request ที่ตรงกันใน `output/context_cache/` ส่วนคำตอบไม่มี cache

## Baseline กับ Improved ต่างกันตรงไหน

| ส่วน | Baseline — ข้อ (1) | Improved — ข้อ (2) |
|---|---|---|
| Notebook | [1)rag_baseline.ipynb](<1)rag_baseline.ipynb>) | [2)rag_improved.ipynb](<2)rag_improved.ipynb>) |
| Chunking | แบ่งตามหัวข้อ; หัวข้อที่เกิน 1,024 tokens แบ่งย่อยด้วย overlap 64 tokens ก่อน embedding | แบ่งภายในหัวข้อไม่เกิน 384 tokens และ overlap 64 tokens ตาม tokenizer ของ BGE-M3 |
| BM25 | ใช้ `unicode61` ที่เก็บสระและวรรณยุกต์ไทย | แบ่งคำด้วย `newmm` ก่อนใช้ `unicode61` |
| Candidates | Vector และ BM25 อย่างละ 10 | Vector และ BM25 อย่างละ 20 |
| รวมอันดับ | RRF ค่า 60 แล้วเลือก 5 | RRF ค่า 60 แล้วนำ 20 อันดับแรกไป rerank |
| Reranking | ไม่มี | BGE reranker เลือก 5 อันดับแรก |
| Embedding | Ollama `bge-m3` | Ollama `bge-m3` และ digest เดียวกัน |
| LLM สร้างบริบท | Ollama `qwen3:4b-instruct` | โมเดล, digest, prompt และพารามิเตอร์เดียวกัน |
| LLM คำตอบสุดท้าย | Qwen `qwen3:4b-instruct` ผ่าน Ollama | โมเดล, prompt, temperature และงบข้อความเดียวกัน |

Improved คาดว่าจะช่วยให้ chunks มีขนาดเหมาะสม รักษาข้อมูลรอยต่อ ค้นคำไทยได้ดีขึ้น และเลือกหลักฐานที่ตรงคำถามมากขึ้น แต่มีต้นทุนเพิ่มจากจำนวน chunks การสร้างบริบท และ reranking ต้องใช้คะแนนจริงยืนยันว่าดีขึ้นหรือไม่ การทดลองนี้เปลี่ยนหลายส่วนพร้อมกัน จึงอธิบายผลรวมได้ แต่แยกผลของแต่ละส่วนไม่ได้

Baseline ดัดแปลงส่วนฐานข้อมูลจาก OpenSearch ในสไลด์เป็น ChromaDB + SQLite FTS5 ทั้งสองระบบใช้ backend ใหม่นี้ `unicode61` และ `newmm` จึงต่างจาก standard/Thai analyzer ของ OpenSearch ผลเปรียบเทียบนี้เป็นของระบบที่ดัดแปลงแล้ว

Baseline คงการแบ่งตามหัวข้อ แต่เพิ่มการแบ่งย่อยเฉพาะหัวข้อที่เกิน 1,024 tokens ก่อนส่งเข้า Ollama `bge-m3` พร้อม overlap 64 tokens เพื่อหลีกเลี่ยงข้อจำกัด context ของ embedding model โดยยังอ้างตำแหน่งข้อความต้นฉบับได้ ใช้เพดานรวม chunk กับบริบท 1,536 tokens และบันทึกค่าเหล่านี้ใน `results.json`

## วิธีรันในเครื่อง

ถือว่า Ollama, `bge-m3` และ `qwen3:4b-instruct` พร้อมใช้งานแล้ว เปิด Terminal ในโฟลเดอร์งานที่มี Notebook และ `requirements.txt` แล้วติดตั้งลง Python ในเครื่อง:

```bash
python3 -m pip install -r requirements.txt jupyterlab
```

ไม่ต้องใช้ API key; `.env` ใช้ตั้ง `RAG_DEVICE` สำหรับ reranker ได้

เปิด Jupyter จากโฟลเดอร์เดียวกัน เพื่อให้ Notebook อ่าน `.env`, corpus และเฉลยได้:

```bash
python3 -m jupyter lab
```

เปิด Notebook ที่ต้องการ แล้วเลือก **Restart Kernel → Run All** หรือเปิดไฟล์โดยตรงด้วยคำสั่งใดคำสั่งหนึ่ง:

```bash
python3 -m jupyter lab '1)rag_baseline.ipynb'
python3 -m jupyter lab '2)rag_improved.ipynb'
```

เลือก kernel ของ Python ที่ติดตั้ง dependencies ข้างต้น รันแต่ละ Notebook แยกกัน โดยแต่ละไฟล์เปิดและรันได้เอง

| ค่าในเซลล์ตั้งค่า | สิ่งที่เกิดขึ้น |
|---|---|
| `RUN_LIVE=False` | ตรวจ corpus และ self-check โดยไม่โหลดน้ำหนักโมเดล ไม่เรียก API และไม่ส่งออกผลทดลอง |
| `RUN_LIVE=True` | โหลด tokenizer และ reranker เฉพาะ Improved เรียก Ollama สร้างบริบทและทำ embedding สร้างดัชนี ทดลองคำถาม 18 ข้อ ใช้ Ollama ตอบคำถามและส่งออกผลจริง |

ปัจจุบันทั้งสองไฟล์ตั้ง `RUN_LIVE=True` หากต้องการตรวจ offline ให้เปลี่ยนเป็น `False` ก่อนรัน การสร้างบริบท embedding และคำตอบทำงานในเครื่อง การดาวน์โหลด tokenizer และ reranker ครั้งแรกต้องใช้อินเทอร์เน็ต

ค่าใน `.env` ใช้สำหรับตั้งอุปกรณ์ reranker เช่น `RAG_DEVICE=auto`

### เลือก CPU/GPU สำหรับ Reranker

Reranker ใช้ `RAG_DEVICE=auto` เป็นค่าเริ่มต้น: เลือก CUDA ถ้ามี แล้ว MPS สำหรับ GPU ของ Mac และ CPU เมื่อไม่มี GPU ที่ PyTorch ใช้งานได้ สามารถตั้งใน `.env` ข้าง Notebook:

```dotenv
RAG_DEVICE=auto
```

หากต้องการบังคับ reranker ใช้ CPU ให้เปลี่ยนบรรทัดเดียวกันเป็น:

```dotenv
RAG_DEVICE=cpu
```

หลังเปลี่ยนเลือก **Restart Kernel → Run All** ดูอุปกรณ์จริงจากข้อความ `reranker device:` และ `reranker_device` ใน `results.json` ค่าจาก environment ที่ตั้งไว้มีลำดับความสำคัญกว่า `.env` ตัวเลือกนี้ใช้เฉพาะ reranker ของ Improved ส่วน Ollama จัดการ GPU ของบริบทและ embedding เอง และคำตอบสุดท้ายใช้ Ollama ในเครื่อง

## รันแล้วได้อะไร

Notebook แสดงคะแนน คำตอบ และหลักฐานของระบบนั้นโดยตรง เมื่อรันครบจะส่งออก 4 ไฟล์ใน `output/baseline/run-.../` หรือ `output/improved/run-.../`:

| ไฟล์ | เนื้อหา |
|---|---|
| `results.json` | Configuration จำนวน chunks ผลค้น คำตอบ หลักฐานที่ส่งจริง และเวลา ครบ 18 คำถาม |
| `retrieval_metrics.csv` | คะแนนการค้นและเวลาของแต่ละคำถาม |
| `summary.csv` | คะแนนเฉลี่ยและเวลาเฉลี่ยสำหรับเทียบสองระบบ |
| `answer_review.csv` | คำตอบกับเฉลย 18 แถว พร้อมช่องว่างให้ตรวจความถูกต้อง ความครบถ้วน และข้อความที่ไม่มีหลักฐานรองรับ |

ใช้ `summary.csv` จากการรันของทั้งสองระบบที่มี Ollama embedding digest, context model/digest, tokenizer revision, ค่าโมเดลและ prompt ตรงกัน เติมผลที่วัดได้ใน [rag_comparison.csv](rag_comparison.csv) สำหรับข้อ (3) ใช้ผลรันใหม่ที่มีโมเดลและ backend ตรงกันในการเปรียบเทียบ:

- `hit_at_5`: สัดส่วนคำถามที่พบหลักฐานใน 5 อันดับแรก
- `mrr_at_5`: คะแนนตามอันดับของหลักฐานแรก ยิ่งพบเร็วคะแนนยิ่งสูง
- `evidence_recall_at_5`: สัดส่วนหลักฐานอ้างอิงที่ค้นพบใน 5 อันดับแรก
- `mean_search_total_ms`: เวลาค้นรวม reranking; `mean_rerank_ms` แสดงต้นทุน reranking

คะแนนการค้นเฉลี่ยคำนวณจาก 15 คำถามที่มีคำตอบ ส่วนเวลาเฉลี่ยใช้ครบ 18 คำถามหลัง warm-up อีก 3 คำถามที่ไม่มีคำตอบใช้ตรวจว่าระบบปฏิเสธการตอบได้เหมาะสมหรือไม่ แบบตรวจคำตอบของทั้งสองระบบรวม 36 แถว ต้องตรวจด้วยคน และแยกประโยชน์ที่คาดหวังจากผลที่วัดได้จริง

ผลที่รันด้วยงบหลักฐาน 12,000 characters เป็นผลเก่า ต้องรัน Baseline และ Improved ใหม่ด้วยงบ 7,000 characters ก่อนนำ `summary.csv` มาเปรียบเทียบ

## แหล่งที่มาของงาน

- สไลด์หน้า 22: [โค้ดนำเข้าเอกสาร](https://github.com/aekanun2020/2025-authenticRAG/blob/4567d3d5d377f6e38ac18afd62133476425f98bf/authenticRAG.py)
- สไลด์หน้า 25: [โค้ดค้นและตอบ](https://github.com/aekanun2020/2025-authenticRAG/blob/4567d3d5d377f6e38ac18afd62133476425f98bf/onlysearchAuthenticRAG.py)

รายละเอียดการดัดแปลง แหล่งอ้างอิง และ self-check อยู่ใน Notebook ของแต่ละระบบ

Embedding ใช้ [Ollama `/api/embed`](https://docs.ollama.com/api/embed) พร้อม `truncate=False` ส่วน reranker คง CrossEncoder เพราะ [Ollama 0.35.0](https://github.com/ollama/ollama/blob/v0.35.0/server/routes.go) ไม่มี API ให้คะแนน rerank คู่คำถาม–ข้อความ

บริบทใช้ [Ollama `/api/chat`](https://docs.ollama.com/api/chat) โดยกำหนด `num_ctx=32768`, `num_predict=512`, temperature `0.1`, `stream=False`, `think=False`, `truncate=False` และ `shift=False` ตาม [Ollama 0.35.0](https://github.com/ollama/ollama/blob/v0.35.0/api/types.go) ใช้ timeout 600 วินาที ต้องจบด้วย `done=True`, `done_reason="stop"` และข้อความไม่ว่างก่อนเขียน cache หาก prompt เกินความจุให้หยุดแทนการตัดเอกสาร ทั้งสองระบบใช้ค่าเดียวกัน

หลังเปลี่ยนคำตอบสุดท้ายเป็น Ollama ต้องรันใหม่ทั้งสอง Notebook ห้ามนำผลที่ใช้โมเดลตอบคำถามเดิมมาเทียบกับผลใหม่ โมเดลตอบคำถามเปลี่ยนเป็น `qwen3:4b-instruct` เหมือนกันทั้งสองระบบ
