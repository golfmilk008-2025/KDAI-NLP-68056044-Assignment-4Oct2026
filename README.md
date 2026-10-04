# RAG: Baseline และ Improved

งานนี้เปรียบเทียบระบบ RAG สองระบบด้วยเอกสาร Markdown 4 ไฟล์ และคำถาม 18 ข้อ ใช้ Jupyter ในเครื่อง โดยค้นหาเอกสารในเครื่องและเรียก Qwen ผ่าน OpenRouter เพื่อสร้างบริบทและคำตอบ

## Tech Stack

```text
Jupyter Notebook (Python)
 ├─ LlamaIndex → แบ่ง Markdown ตามหัวข้อ
 ├─ Sentence Transformers
 │   ├─ BGE-M3 → Embedding
 │   └─ BGE reranker → จัดอันดับใหม่เฉพาะ Improved
 ├─ ChromaDB → ค้นหาเวกเตอร์ด้วย cosine
 ├─ SQLite FTS5 → ค้นหา BM25
 │   └─ PyThaiNLP newmm → แบ่งคำไทยเฉพาะ Improved
 └─ OpenAI Python SDK
     └─ OpenRouter
         └─ Qwen → สร้างบริบทและคำตอบ
```

| เทคโนโลยี | หน้าที่ในงาน | ทำงานที่ไหน |
|---|---|---|
| Python + JupyterLab | เปิด Notebook และรันตามลำดับเซลล์ | ในเครื่อง |
| LlamaIndex (`llama-index-core`) | ใช้ `MarkdownNodeParser` แบ่งข้อความตามหัวข้อ | ในเครื่อง |
| Sentence Transformers + Transformers + PyTorch | โหลดและรันโมเดล embedding และ reranker | ในเครื่อง; ดาวน์โหลดโมเดลจาก Hugging Face ครั้งแรก |
| `BAAI/bge-m3` | แปลงข้อความและคำถามเป็นเวกเตอร์ ใช้ทั้งสองระบบ | ในเครื่อง |
| `BAAI/bge-reranker-v2-m3` | ให้คะแนนคู่คำถาม–chunk เฉพาะ Improved ผ่าน `CrossEncoder` | ในเครื่อง |
| ChromaDB | เก็บเวกเตอร์ด้วย `PersistentClient` และค้นหาด้วย cosine | ในเครื่อง: `output/chroma/` |
| SQLite FTS5 (`sqlite3` ของ Python) | สร้างดัชนีข้อความและค้นหาด้วย BM25 | ในหน่วยความจำ; สร้างใหม่เมื่อ Run All |
| PyThaiNLP | แบ่งคำไทยด้วย `newmm` ก่อน BM25 ทั้งเอกสารและคำถามใน Improved | ในเครื่อง |
| OpenAI Python SDK (`openai`) | ส่งคำขอไปยัง endpoint ของ OpenRouter | เรียกผ่านอินเทอร์เน็ต |
| OpenRouter + Qwen | ใช้ `qwen/qwen-2.5-72b-instruct` เป็นค่าเริ่มต้นสำหรับสร้างบริบทและคำตอบ | ผ่าน OpenRouter API |
| python-dotenv | โหลด API key และค่าตั้งระบบจาก `.env` โดย environment ที่ตั้งไว้มีลำดับความสำคัญกว่า | ในเครื่อง |

เวอร์ชัน dependencies ระบุใน [requirements.txt](requirements.txt) โมเดล embedding และ reranker ตรึง revision ใน Notebook ส่วนชื่อ Qwen เปลี่ยนได้ด้วย `RAG_LLM_MODEL` และต้องใช้ค่าเดียวกันทั้งสองระบบเมื่อเปรียบเทียบ

## ระบบทำงานอย่างไร

**เตรียมข้อมูล:** อ่านเอกสาร → แบ่ง chunks → ส่งเอกสารเต็มและ chunk ให้ Qwen สร้างบริบท → ทำ embedding ข้อความ chunk รวมบริบทด้วย BGE-M3 → นำเข้า ChromaDB และ SQLite FTS5

**ค้นและตอบ:** คำถาม → ค้น Vector และ BM25 → รวมอันดับด้วย RRF → reranking เฉพาะ Improved → เลือกหลักฐาน 5 chunks → ส่งหลักฐานและคำถามให้ Qwen ตอบภาษาไทยพร้อมอ้างอิง `[D1]`, `[D2]`

ทั้งสองระบบใช้งบข้อความหลักฐาน 12,000 ตัวอักษร จึงอาจตัดข้อความหรือใช้หลักฐานไม่ครบทั้ง 5 chunks เมื่อถึงงบ บริบทที่สร้างไว้มี cache สำหรับ request ที่ตรงกันใน `output/context_cache/` ส่วนคำตอบไม่มี cache

## Baseline กับ Improved ต่างกันตรงไหน

| ส่วน | Baseline — ข้อ (1) | Improved — ข้อ (2) |
|---|---|---|
| Notebook | [1)rag_baseline.ipynb](<1)rag_baseline.ipynb>) | [2)rag_improved.ipynb](<2)rag_improved.ipynb>) |
| Chunking | แบ่งตามหัวข้อ ขนาดแต่ละ chunk ขึ้นกับเอกสาร | แบ่งภายในหัวข้อไม่เกิน 384 tokens และ overlap 64 tokens ตาม tokenizer ของ BGE-M3 |
| BM25 | ใช้ `unicode61` ที่เก็บสระและวรรณยุกต์ไทย | แบ่งคำด้วย `newmm` ก่อนใช้ `unicode61` |
| Candidates | Vector และ BM25 อย่างละ 10 | Vector และ BM25 อย่างละ 20 |
| รวมอันดับ | RRF ค่า 60 แล้วเลือก 5 | RRF ค่า 60 แล้วนำ 20 อันดับแรกไป rerank |
| Reranking | ไม่มี | BGE reranker เลือก 5 อันดับแรก |
| Embedding | BGE-M3 | BGE-M3 เดียวกัน |
| LLM และ prompt | Qwen สำหรับบริบทและคำตอบ | Qwen และ prompt เดียวกัน รวมถึง temperature และงบข้อความ |

Improved คาดว่าจะช่วยให้ chunks มีขนาดเหมาะสม รักษาข้อมูลรอยต่อ ค้นคำไทยได้ดีขึ้น และเลือกหลักฐานที่ตรงคำถามมากขึ้น แต่มีต้นทุนเพิ่มจากจำนวน chunks การสร้างบริบท และ reranking ต้องใช้คะแนนจริงยืนยันว่าดีขึ้นหรือไม่ การทดลองนี้เปลี่ยนหลายส่วนพร้อมกัน จึงอธิบายผลรวมได้ แต่แยกผลของแต่ละส่วนไม่ได้

Baseline ดัดแปลงส่วนฐานข้อมูลจาก OpenSearch ในสไลด์เป็น ChromaDB + SQLite FTS5 ทั้งสองระบบใช้ backend ใหม่นี้ `unicode61` และ `newmm` จึงต่างจาก standard/Thai analyzer ของ OpenSearch ผลเปรียบเทียบนี้เป็นของระบบที่ดัดแปลงแล้ว

## วิธีรันในเครื่อง

เปิด Terminal ในโฟลเดอร์งานที่มี Notebook และ `requirements.txt` แล้วติดตั้งลง Python ในเครื่อง:

```bash
python3 -m pip install -r requirements.txt jupyterlab
```

ใส่ key หลัง `OPENROUTER_API_KEY=` ในไฟล์ `.env` ข้าง Notebook:

```dotenv
OPENROUTER_API_KEY=
```

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
| `RUN_LIVE=True` | โหลดโมเดล สร้างดัชนี ทดลองคำถาม 18 ข้อ เรียก OpenRouter และส่งออกผลจริง |

ปัจจุบันทั้งสองไฟล์ตั้ง `RUN_LIVE=True` หากต้องการตรวจ offline ให้เปลี่ยนเป็น `False` ก่อนรัน การทดลองจริงต้องใช้อินเทอร์เน็ตและ OpenRouter API key โดยมีค่าใช้จ่ายตามการใช้งาน API การโหลดโมเดลและสร้างบริบทครั้งแรกใช้เวลา

หาก key ว่าง Notebook จะหยุดและแจ้งให้เติม `.env` หลังแก้ key ให้ Restart Kernel แล้ว Run All ไฟล์ `.env` อยู่ใน `.gitignore` และ key ไม่ถูกพิมพ์หรือส่งออกพร้อมผลทดลอง

## รันแล้วได้อะไร

Notebook แสดงคะแนน คำตอบ และหลักฐานของระบบนั้นโดยตรง เมื่อรันครบจะส่งออก 4 ไฟล์ใน `output/baseline/run-.../` หรือ `output/improved/run-.../`:

| ไฟล์ | เนื้อหา |
|---|---|
| `results.json` | Configuration จำนวน chunks ผลค้น คำตอบ หลักฐานที่ส่งจริง และเวลา ครบ 18 คำถาม |
| `retrieval_metrics.csv` | คะแนนการค้นและเวลาของแต่ละคำถาม |
| `summary.csv` | คะแนนเฉลี่ยและเวลาเฉลี่ยสำหรับเทียบสองระบบ |
| `answer_review.csv` | คำตอบกับเฉลย 18 แถว พร้อมช่องว่างให้ตรวจความถูกต้อง ความครบถ้วน และข้อความที่ไม่มีหลักฐานรองรับ |

ใช้ `summary.csv` จากการรันของทั้งสองระบบที่มีค่าโมเดลและ prompt ตรงกัน เติมผลที่วัดได้ใน [rag_comparison.csv](rag_comparison.csv) สำหรับข้อ (3):

- `hit_at_5`: สัดส่วนคำถามที่พบหลักฐานใน 5 อันดับแรก
- `mrr_at_5`: คะแนนตามอันดับของหลักฐานแรก ยิ่งพบเร็วคะแนนยิ่งสูง
- `evidence_recall_at_5`: สัดส่วนหลักฐานอ้างอิงที่ค้นพบใน 5 อันดับแรก
- `mean_search_total_ms`: เวลาค้นรวม reranking; `mean_rerank_ms` แสดงต้นทุน reranking

คะแนนการค้นเฉลี่ยคำนวณจาก 15 คำถามที่มีคำตอบ ส่วนเวลาเฉลี่ยใช้ครบ 18 คำถามหลัง warm-up อีก 3 คำถามที่ไม่มีคำตอบใช้ตรวจว่าระบบปฏิเสธการตอบได้เหมาะสมหรือไม่ แบบตรวจคำตอบของทั้งสองระบบรวม 36 แถว ต้องตรวจด้วยคน และแยกประโยชน์ที่คาดหวังจากผลที่วัดได้จริง

## แหล่งที่มาของงาน

- สไลด์หน้า 22: [โค้ดนำเข้าเอกสาร](https://github.com/aekanun2020/2025-authenticRAG/blob/4567d3d5d377f6e38ac18afd62133476425f98bf/authenticRAG.py)
- สไลด์หน้า 25: [โค้ดค้นและตอบ](https://github.com/aekanun2020/2025-authenticRAG/blob/4567d3d5d377f6e38ac18afd62133476425f98bf/onlysearchAuthenticRAG.py)

รายละเอียดการดัดแปลง แหล่งอ้างอิง และ self-check อยู่ใน Notebook ของแต่ละระบบ
