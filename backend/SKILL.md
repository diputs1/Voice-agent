---
name: vin-agent-refactor
description: Refactor code trong project Vin Agent (FastAPI + LangGraph backend, Next.js frontend) theo clean code và SOLID. LUÔN dùng skill này khi được yêu cầu refactor, dọn code, tách class/function, sửa vi phạm SOLID, hoặc chuẩn bị code trước khi thêm feature mới trong backend/app hoặc frontend/app — kể cả khi user chỉ nói "dọn lại code này" hoặc "sao cho sạch hơn" mà không nhắc rõ SOLID.
---

# Vin Agent — Refactor Clean Code / SOLID

Skill này áp dụng cho project Vin Agent: voice/text Q&A agent cho Vinpearl Safari, dùng FastAPI + LangGraph (backend), Next.js (frontend), Postgres/pgvector cho KB.

Mục tiêu: refactor **không đổi hành vi quan sát được** (API contract, SSE event shape, DB schema) trừ khi user yêu cầu rõ ràng. Refactor là thay đổi cấu trúc bên trong, không phải rewrite tính năng.

## Cấu trúc file/module mục tiêu

**Nguyên tắc chọn khi nào tách file mới:** tách theo *lý do thay đổi* (reason to change), không tách theo độ dài dòng code. Nếu 1 file bị sửa vì 2 lý do khác nhau (VD: sửa vì đổi routing logic HOẶC vì đổi cách gọi LLM) thì đó là dấu hiệu nên tách, bất kể file đang dài hay ngắn. Tách file chỉ vì "dài quá" mà không theo ranh giới trách nhiệm sẽ tạo thêm file rối chứ không giảm coupling — không tách máy móc theo số dòng.

Cấu trúc đích cho backend (áp dụng dần, không bắt buộc làm 1 lần):

```
backend/app/
  routers/
    chat.py            # route /chat/stream — chỉ nhận request, gọi service, format SSE
    admin.py            # route /admin/* — chỉ nhận request, gọi service
  services/
    chat_service.py     # orchestrate graph.astream + save thread turn
    ingestion_service.py # logic ingest/crawl hiện đang nằm trong main.py
  agents/
    graph.py             # CHỈ add_node/add_edge/conditional_edges, không chứa logic node
    nodes/
      supervisor.py
      contextualize.py
      safari_knowledge.py
      offer_freshness.py
      retry_query.py
      escalation.py
      voice_answer.py
    schemas.py            # SafariState, structured intent schema
    prompts.py             # prompt string tách khỏi logic, không lẫn vào node function
  kb.py                    # giữ nguyên, hoặc tách repositories/ nếu phình to
```

`main.py` sau refactor chỉ còn: khởi tạo app, lifecycle (`startup`), include router — không chứa route handler hay business logic trực tiếp.

## Quy trình bắt buộc trước khi sửa bất kỳ file nào

1. Chạy `pytest` và `ruff check app` trước để có baseline — nếu đang fail sẵn thì ghi lại, đừng đổ lỗi cho refactor sau này.
2. Đọc toàn bộ node function liên quan trong `backend/app/agents/graph.py` trước khi tách — các node phụ thuộc lẫn nhau qua `SafariState` (shared dict), đổi key state ở một node có thể làm node sau silently fail (KeyError hoặc default sai) chứ không throw rõ ràng.
3. Sau MỖI lần refactor một đơn vị nhỏ (1 class/1 node/1 module): chạy lại `pytest` + `ruff check app` + `python -m app.evals.run_eval` ngay. Không gộp nhiều thay đổi lớn rồi mới test — nếu fail sẽ khó biết đổi chỗ nào gây ra.
4. Không refactor và thêm feature trong cùng một commit/lần sửa.

## Nguyên tắc SOLID áp dụng cụ thể vào kiến trúc này

### S — Single Responsibility

**Vi phạm điển hình trong project này:** node function trong `graph.py` đang gộp 3 việc: (a) đọc state, (b) gọi business logic (search KB, gọi LLM), (c) format lại thành `SafariState` update dict. Ví dụ `_safari_knowledge_node` vừa gọi `kb.search()` vừa tự transform kết quả thành `_citation()`/`_hit_payload()`.

**Cách tách đúng:**
- Node function trong graph CHỈ làm điều phối: đọc state → gọi service/repository → trả update dict. Không chứa business logic (parse, scoring, formatting) trực tiếp.
- Business logic (classify intent, rewrite query, format citation, tính confidence) chuyển thành function/class riêng trong module tách biệt (`app/services/`), có thể unit test độc lập không cần dựng cả graph.
- `main.py` hiện đang vừa định nghĩa route vừa chứa logic build KB, xử lý ingest, format SSE — mỗi route handler nên mỏng, delegate cho service layer.

### O — Open/Closed

**Vi phạm điển hình:** logic `classify_category`, `_valid_until`, việc thêm category time-sensitive mới đều phải sửa trực tiếp if/else trong `ingestion.py` hoặc `graph.py`.

**Cách tách đúng:** Dùng registry/strategy pattern cho những thứ hay mở rộng — VD danh mục category, rule intent classification — thay vì if/elif dài. Category → TTL mapping nên là config dict/table tra cứu, không phải hard-code trong function logic.

### L — Liskov Substitution

**Điểm đã làm đúng, giữ nguyên khi refactor:** `PostgresKnowledgeBase` và `InMemoryKnowledgeBase` cùng implement chung interface `KnowledgeBase` (`search`, `upsert_chunks`, `save_thread_turn`, `ensure_ready`). Khi refactor, đảm bảo:
- Không thêm method chỉ có ở 1 implementation rồi gọi trực tiếp từ `main.py` qua type check (`isinstance`) — phá vỡ tính thay thế được.
- Nếu thêm implementation mới (VD `RedisKnowledgeBase` cho cache), method signature và exception behavior phải khớp interface hiện có, kể cả trường hợp lỗi (raise cùng loại exception).

### I — Interface Segregation

**Vi phạm điển hình:** nếu `KnowledgeBase` interface phình to (thêm method riêng cho admin/ingest không liên quan tới path Q&A chính), mọi implementation phải cài đặt cả những method không dùng tới.

**Cách tách đúng:** Tách interface theo use-case — `Retriever` (chỉ `search`) dùng cho graph node, `IngestionSink` (chỉ `upsert_chunks`) dùng cho admin endpoints, `ThreadStore` (chỉ `save_thread_turn`/load history) dùng cho `contextualize_query`. Một class có thể implement nhiều interface nhỏ, nhưng caller chỉ phụ thuộc interface nó thực sự cần.

### D — Dependency Inversion

**Vi phạm điển hình:** node function trong `graph.py` gọi trực tiếp `self.kb.search(...)`, `self.llm.astream(...)` — nếu `SafariAgentGraph` tự khởi tạo các dependency này bên trong thay vì nhận qua constructor thì không test được node độc lập (phải mock bằng cách monkeypatch thay vì inject).

**Cách tách đúng:** Đảm bảo `SafariAgentGraph.__init__` nhận toàn bộ dependency (`kb`, `llm`, `settings`) qua tham số — đã có phần này đúng theo mô tả hiện tại (`SafariAgentGraph(app.state.kb, settings)`), giữ nguyên pattern này khi thêm class mới. Không để business logic class tự `import` và khởi tạo client bên trong method.

## Bottleneck hiệu năng cần kiểm tra riêng (không tự tin refactor cấu trúc là xong)

Refactor cho gọn KHÔNG tự động làm hệ thống nhanh hơn — đây là 2 việc khác nhau, phải kiểm tra riêng. Khi refactor các vùng dưới đây, luôn hỏi lại: "thao tác này có đang block event loop hoặc gọi tuần tự không cần thiết không?"

- [ ] **Postgres driver sync vs async**: nếu `PostgresKnowledgeBase` dùng driver sync (`psycopg2`, `conn.execute` không `await`) bên trong FastAPI async app, MỌI query sẽ block toàn bộ event loop — nghẽn nghiêm trọng nhất trong hệ thống vì ảnh hưởng tất cả request đồng thời, không chỉ 1 user. Ưu tiên đổi sang `asyncpg`/`psycopg3 AsyncConnection`, hoặc tạm thời bọc bằng `await asyncio.to_thread(...)` nếu chưa đổi driver ngay được.
- [ ] **Retry loop trong `offer_freshness` → `retry_query` → `safari_knowledge`**: mỗi lần retry là 1 round-trip embedding API + 1 query vector search tuần tự, cộng dồn vào latency trước token đầu tiên. Không re-embed câu hỏi gốc nếu retry chỉ expand query dựa trên câu cũ — cache embedding gốc trong state.
- [ ] **KB search fetch quá nhiều rows**: `LIMIT max(limit * 6, 20)` kéo cả cột `content` dài rồi rerank ở Python dù chỉ dùng top 5 — cân nhắc đẩy phần lọc sơ bộ xuống SQL (kết hợp `WHERE`/full-text filter với vector search) thay vì fetch hết rồi lọc ở application layer.
- [ ] **`/admin/crawl` gọi `wait_for_crawl` chặn trong request handler**: nếu job Firecrawl chạy lâu, request đó treo tương ứng; nếu share process với `/chat/stream` sẽ ảnh hưởng cả người dùng thật. Chuyển sang enqueue job (`BackgroundTasks` hoặc queue riêng) + trả `job_id` ngay, client poll trạng thái qua endpoint riêng — đã có sẵn `crawl_jobs` dict nên tận dụng đúng mục đích này thay vì block.

## Checklist refactor riêng cho các vấn đề đã audit trước đó

Khi đụng tới các khu vực này, tiện thể refactor luôn kèm theo fix (nếu user đồng ý mở rộng scope — nếu không, chỉ note lại bằng TODO/comment, đừng tự ý fix behavior khi task chỉ là refactor):

- [ ] `_build_knowledge_base`: `except Exception` quá rộng, nên catch cụ thể loại lỗi connection và log ở mức ERROR + có cờ trạng thái đọc được từ `/health`, không chỉ warning im lặng.
- [ ] Threshold `confidence < 0.05` trong `offer_freshness` nên là named constant có comment giải thích nguồn gốc con số, không phải magic number rải trong logic.
- [ ] `/admin/*` endpoints: khi refactor route layer, thêm chỗ trống rõ ràng (dependency injection slot) để cắm auth middleware sau, kể cả khi chưa implement auth ngay trong lần refactor này.
- [ ] `app.state.crawl_jobs = {}`: nếu refactor phần job tracking, tách thành `JobStore` interface ngay từ đầu (dù implementation vẫn là in-memory tạm thời) để sau này swap sang Postgres/Redis không phải sửa call site.

## Quy tắc đặt tên và cấu trúc module

- Node function trong `graph.py`: tiền tố `_xxx_node`, chỉ chứa orchestration, giữ nguyên như hiện tại — không đổi tên node đã có trong graph edges nếu không cập nhật đồng bộ cả `astream_pre_answer` và test.
- Business logic thuần (không phụ thuộc `SafariState`): đặt trong `app/services/<domain>.py`, phải là pure function hoặc class không side-effect ẩn, dễ test bằng input/output rõ ràng.
- Repository/KB access: giữ trong `app/kb.py` hoặc tách `app/repositories/` nếu file quá dài, luôn qua interface `KnowledgeBase`, không query SQL trực tiếp từ route handler hay node function.

## Thứ tự ưu tiên khi thực hiện

Không làm tất cả cùng lúc. Theo thứ tự này trừ khi user chỉ định khác:

| # | Việc | Loại | Vì sao trước |
|---|---|---|---|
| 1 | Xác nhận Postgres driver sync/async, fix nếu đang block event loop | Bottleneck | Ảnh hưởng mọi request đồng thời, rủi ro cao nhất |
| 2 | `/admin/crawl` chuyển sang background job thật (không block) | Bottleneck | Cô lập ảnh hưởng của admin task khỏi user thật |
| 3 | Tách `main.py` thành `routers/` + `services/` | Cấu trúc | Giảm rủi ro khi sửa các phần sau |
| 4 | Tách `graph.py` thành `nodes/` package theo SRP | Cấu trúc | Cho phép unit test từng node độc lập |
| 5 | Áp DIP/ISP cho `KnowledgeBase` interface | Cấu trúc | Làm nền cho việc thêm implementation mới sau này |
| 6 | Giảm fetch rows dư thừa trong KB search | Bottleneck | Tối ưu sau khi cấu trúc đã ổn định, tránh vừa sửa cấu trúc vừa đổi query cùng lúc |

## Sau khi refactor xong

1. Diff lại toàn bộ SSE event shape gửi về frontend (`status`, `token`, `replace`, `done`) — đây là contract với `frontend/app/page.tsx`, tuyệt đối không đổi field name/type mà không sửa frontend tương ứng.
2. Chạy lại full suite: `ruff check app`, `pytest`, `python -m app.evals.run_eval`.
3. Tóm tắt lại theo đúng format Context → Problem → Solution cho từng nhóm thay đổi để dễ review, không liệt kê diff thô.
