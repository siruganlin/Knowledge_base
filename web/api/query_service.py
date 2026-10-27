from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse
from processor.query_porcessor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import get_recent_messages, clear_history
from utils.sse_utils import create_sse_queue, sse_generator
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result

app = FastAPI()

class QueryRequest(BaseModel):
    query:str
    session_id:str
    is_stream:bool

@app.get("/chat.html")
async def Chat():
    current_dir_parent_path = Path(__file__).absolute().parent .parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"

    return FileResponse(str(chat_html_path))


@app.post("/query") # 这里为什么要用post，因为参数提交的是一个对象，所以要用post去封装，get没有办法去封装这种对象
async def query(backgroundtasks: BackgroundTasks, query:QueryRequest):
    user_query = query.query
    session_id = query.session_id
    is_stream = query.is_stream

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream) # 记录任务进度

    if is_stream:
        backgroundtasks.add_task(run_query_graph,session_id,user_query,is_stream)  # 调用工作流
        return {"message":"任务已经开始，请耐心等待","session_id":session_id}
    else:
        run_query_graph(session_id,user_query,is_stream)
        answer = get_task_result(session_id,"answer","")
        return {
            "message":"处理完成",
            "session_id":session_id,
            "answer":answer
        }

def run_query_graph(session_id:str,user_query:str,is_stream:bool):
    print("调用搜索工作流")

    init_state = { # 初始状态
        "original_query":user_query,
        "session_id":session_id,
        "is_stream":is_stream
    }
    workflow = KBQueryWorkflow() # 开始执行图

    # 注意：流式输出需要使用执行器执行,流式输出默认返回的是一个可执行的对象
    for chunk in workflow.run(init_state,stream=is_stream):
        pass

# 寻找结果
@app.get("/stream/{session_id}")
async def stream(session_id:str,request:Request):
    print("追踪结果生成...")

    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream", # 1. 告诉浏览器：这是 Server-Sent Events 长连接，持续接收消息。
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    清空指定会话的历史记录
    """
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


"""
BackgroundTasks 负责生产（跑 LangGraph 图），sse_generator 才是把队列内容吐给前端的那个函数。它俩一个是生产者、一个是消费者。
这里如果假设后台启动了4个uvcorn进程，前端给哪个进程发的请求，哪个进程后端才会有消息队列，如果前端读到对应的进程就会出现没有数据的现象
post请求打到哪个内存worker里哪个内存，消息队列就会在哪个进程里面创建
"""

# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)