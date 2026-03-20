from fun_asr_gguf import create_asr_engine
import requests
import json
import os


def asr_voice(wav_path):
    # 创建并初始化引擎 (推荐使用单例或长期持有实例)
    engine = create_asr_engine(
        encoder_onnx_path="model/Fun-ASR-Nano-Encoder-Adaptor.int4.onnx",
        ctc_onnx_path="model/Fun-ASR-Nano-CTC.int4.onnx",
        decoder_gguf_path="model/Fun-ASR-Nano-Decoder.q5_k.gguf",
        tokens_path="model/tokens.txt",
        hotwords_path="hot.txt",  # 可选：热词文件路径，支持运行期间实时修改
        similar_threshold=0.6,  # 可选：热词模糊匹配阈值，默认 0.6
        max_hotwords=10,  # 可选：最多提供给 LLM 的热词数量，默认 10
    )
    engine.initialize()
    result = engine.transcribe(wav_path, language="中文")
    print(result.text)
    return result.segments


def merge_content(voiceTime, timestamps):
    speaker_content = []
    for i in voiceTime["data"]:
        start = i["start"]
        end = i["end"]
        speaker = i["speaker"]
        content = ""
        for j in timestamps:
            if start <= j["start"] <= end:
                content = content + j["char"]
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
    voiceUrl = "https://file.meibaokeji.com/media/tools/cr/2026/03/05/52753757821/2026/03/05/fc7ef3c2-602a-4514-9624-eea450f37445.mp3"
    voiceUrl_out = "https://file.meibaokeji.com/media/tools/cr/2026/03/05/52753757821/2026/03/05/fc7ef3c2-602a-4514-9624-eea450f37445_out.mp3"
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
