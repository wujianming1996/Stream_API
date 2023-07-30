from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


async def predict(prompt, history):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    for response, history in model.stream_chat(tokenizer,
                                               prompt,
                                               history=history,
                                               max_length=8192,
                                               top_p=0.75,
                                               temperature=0.95):
        answer = {
            "response": response,
            "history": history,
            "status": 200,
            "time": time
        }
        # log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'

        yield json.dumps(answer, ensure_ascii=False)


app = FastAPI()


@app.post("/stream")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    return StreamingResponse(predict(prompt=prompt, history=history))


@app.post("/chat")
async def create_chat(request: Request):
    global model, tokenizer
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')

    response, history = model.chat(tokenizer, prompt, history=history, max_length=8192, top_p=0.75, temperature=0.95)
    answer = {"response": response,
              "history": history,
              "status": 200,
              "time": time
              }
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("chatglm2-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
