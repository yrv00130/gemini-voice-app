import os
import json
import sys
import subprocess
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

DYNAMIC_TOOLS_FILE = "dynamic_tools.json"
dynamic_tools_store = {}
dynamic_tool_declarations = []

def auto_install_package(module_name: str) -> bool:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
        return True
    except Exception:
        return False

def load_dynamic_tools():
    if not os.path.exists(DYNAMIC_TOOLS_FILE):
        return
    try:
        with open(DYNAMIC_TOOLS_FILE, "r", encoding="utf-8") as f:
            tools_data = json.load(f)
        for item in tools_data:
            func_name, code, desc = item["function_name"], item["python_code"], item["description"]
            local_scope = {}
            try:
                exec(code, globals(), local_scope)
            except (ModuleNotFoundError, ImportError) as ie:
                pkg = getattr(ie, "name", None)
                if pkg and auto_install_package(pkg):
                    exec(code, globals(), local_scope)
            if func_name in local_scope:
                fn = local_scope[func_name]
                setattr(fn, "_source_code", code)
                dynamic_tools_store[func_name] = fn
                dynamic_tool_declarations.append(
                    types.FunctionDeclaration(name=func_name, description=desc, parameters=types.Schema(type="OBJECT", properties={}))
                )
    except Exception as e:
        print(f"Error loading tools: {e}")

load_dynamic_tools()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Live API 接続設定
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(parts=[types.Part.from_text(
            "あなたは完全自律型の音声AIアシスタントです。機能がない場合は request_new_function ツールを呼び出してPythonコードを作成・実行してください。"
        )]),
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="request_new_function",
                description="新機能のPythonコードを動的生成・実行します。",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "function_name": types.Schema(type="STRING"),
                        "python_code": types.Schema(type="STRING"),
                        "description": types.Schema(type="STRING")
                    },
                    required=["function_name", "python_code", "description"]
                )
            )
        ] + dynamic_tool_declarations)]
    )

    try:
        async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-preview-12-2025", config=config) as session:
            # ブラウザからの PCM 音声データを受信して Live API に転送するタスク
            async def forward_inputs():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await session.send_realtime_input(audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"))
                except WebSocketDisconnect:
                    pass

            # Gemini からの音声出力をブラウザに転送するタスク
            async def forward_outputs():
                async for response in session.receive():
                    # ツール呼び出し（Function Calling）の処理
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call and hasattr(tool_call, "function_calls"):
                        for fc in tool_call.function_calls:
                            if fc.name == "request_new_function":
                                args = fc.args
                                name, code, desc = args.get("function_name"), args.get("python_code"), args.get("description", "")
                                local_scope = {}
                                try:
                                    exec(code, globals(), local_scope)
                                except (ModuleNotFoundError, ImportError) as ie:
                                    pkg = getattr(ie, "name", None)
                                    if pkg:
                                        auto_install_package(pkg)
                                        exec(code, globals(), local_scope)
                                
                                res = "成功"
                                if name in local_scope:
                                    dynamic_tools_store[name] = local_scope[name]
                                    res = str(local_scope[name]())
                                
                                await session.send_tool_response(function_responses=[
                                    types.FunctionResponse(name=fc.name, id=fc.id, response={"result": res})
                                ])

                    # 音声再生データの転送
                    server_content = getattr(response, "server_content", None)
                    if server_content and server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await websocket.send_bytes(part.inline_data.data)

            await asyncio.gather(forward_inputs(), forward_outputs())
    except Exception as e:
        print(f"WebSocketエラー: {e}")