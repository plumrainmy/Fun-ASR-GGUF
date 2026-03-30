from fun_asr_gguf import FunASREngine, ASREngineConfig
import requests
import json
import os


def asr_voice(audio_file):
    # ==================== 配置区域 ====================

    # 音频文件路径
    # audio_file = "input.mp3"

    # 语言设置（None=自动检测, "中文", "英文", "日文" 等）
    language = None

    # 上下文信息（留空则不使用）
    context = "这是1004期睡前消息节目，主持人叫督工，助理叫静静"

    # 是否启用 CTC 辅助（True=提供时间戳和热词, False=仅LLM）
    enable_ctc = True

    # 是否打印详细信息
    verbose = True

    # 是否以 JSON 格式输出结果
    json_output = False

    # 模型文件路径
    model_dir = "./model"
    encoder_onnx_path = f"{model_dir}/Fun-ASR-Nano-Encoder-Adaptor.fp16.onnx"
    ctc_onnx_path = f"{model_dir}/Fun-ASR-Nano-CTC.fp16.onnx"
    decoder_gguf_path = f"{model_dir}/Fun-ASR-Nano-Decoder.q5_k.gguf"
    tokens_path = f"{model_dir}/tokens.txt"
    hotwords_path = "./hot.txt"  # 可选，留空则不使用热词

    # ==================== 语言说明 ====================

    """
    Fun-ASR-Nano-2512
        中文、英文、日文

    Fun-ASR-MLT-Nano-2512
        中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
        印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
        匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
        斯洛伐克语、斯洛文尼亚语、瑞典语 
    """

    # ==================== 执行区域 ====================

    # 准备热词
    hotwords = []
    if hotwords_path and os.path.exists(hotwords_path):
        with open(hotwords_path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    config = ASREngineConfig(
        encoder_onnx_path=encoder_onnx_path,
        ctc_onnx_path=ctc_onnx_path,
        decoder_gguf_path=decoder_gguf_path,
        tokens_path=tokens_path,
        hotwords=hotwords,
        similar_threshold=0.6,
        max_hotwords=10,
        enable_ctc=enable_ctc,
        onnx_provider='cuda',
        verbose=verbose,
    )
    engine = FunASREngine(config)

    print(f'\n预跑一遍，分配内存......\n')
    result = engine.transcribe(
        audio_file,
        language=language,
        context=context,
        verbose=False,
        duration=5.0,
    )
    result = engine.transcribe(
        audio_file,
        language=language,
        context=context,
        verbose=verbose,
        segment_size=60.0,
        overlap=4.0,
        srt=True,
        temperature=0.4
    )

    # 输出结果
    if json_output:
        import json
        print("\n" + "="*70)
        print("识别结果 (JSON)")
        print("="*70)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 清理资源
    engine.cleanup()
    print(result)
    return result.segments


def merge_content(voiceTime, timestamps):
    speaker_content = []
    for i in voiceTime["data"]:
        start = i["start"]
        end = i["end"]
        speaker = i["speaker"]
        content = ""
        for j in timestamps:
            if start <= j[1] <= end:
                content = content + j[0]
        if "陌生人" in speaker:
            speaker = "陌生人"
        speaker_content.append({"speaker": speaker, "content": content})

    # 合并连续相同说话人的内容
    merged_content = []
    for item in speaker_content:
        if merged_content and merged_content[-1]["speaker"] == item["speaker"]:
            merged_content[-1]["content"] += item["content"]
        else:
            merged_content.append(item)

    print("\n=== 合并后的对话 ===")
    for item in merged_content:
        print(f"{item['speaker']}：{item['content']}")


def get_voice_time(voiceUrl):
    url = "http://120.195.113.82:34188/api/voiceprint/get_voice_time"
    payload = json.dumps({
        "workPhone": "19928270165",
        "userId": "343",
        "voiceUrl": voiceUrl,
        "callRecordId": "1"
    })
    headers = {
        'token': 'b2a7849007264cdfa8d8369c1f8496a3',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    voiceTime = response.json()
    print(response.json())
    return voiceTime


def download_file(voiceUrl_out, wav_path):
    # 下载voiceUrl文件到 ./wav 目录下
    print(f"正在下载：{voiceUrl_out}")
    with requests.get(voiceUrl_out, stream=True) as r:
        r.raise_for_status()
        with open(wav_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"文件已保存到：{wav_path}")


def main():
    voiceUrl = "https://file.meibaokeji.com/media/tools/cr/2026/03/11/52753757821/2026/03/11/13f61124-c748-4066-961a-9903161259af.mp3"
    voiceUrl_out = "https://file.meibaokeji.com/media/tools/cr/2026/03/11/52753757821/2026/03/11/13f61124-c748-4066-961a-9903161259af_out.mp3"
    wav_dir = "./wav"
    os.makedirs(wav_dir, exist_ok=True)
    # 从 URL 提取文件名
    file_name = voiceUrl_out.split("/")[-1]
    wav_path = os.path.join(wav_dir, file_name)
    download_file(voiceUrl_out, wav_path)
    voiceTime = get_voice_time(voiceUrl)
    timestamps = asr_voice(wav_path)
    merge_content(voiceTime, timestamps)


if __name__ == '__main__':
    main()
