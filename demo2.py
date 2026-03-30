from pydub import AudioSegment
import numpy as np

# ===================== 配置 =====================
AUDIO_FILE = "./wav/2ffd34cc-7311-4997-8fcc-63ff3a0eedd9_out.mp3"  # 替换成你的文件
SILENCE_THRESHOLD = -30  # 静音阈值（越小越灵敏，-30 适合大多数场景）
MIN_SOUND_DURATION = 100  # 最小有声片段（毫秒，过滤噪音）


# ===================== 核心函数 =====================
def get_sound_ranges(track, threshold=SILENCE_THRESHOLD, min_duration=MIN_SOUND_DURATION):
    """
    传入单声道音轨，返回【所有有声音的时间区间】(开始ms, 结束ms)
    """
    samples = np.array(track.get_array_of_samples())
    sample_rate = track.frame_rate
    ms_per_sample = 1000 / sample_rate

    # 转分贝
    dB = 20 * np.log10(np.abs(samples) + 1e-10)
    is_sound = dB > threshold

    ranges = []
    start = None

    for i, sound in enumerate(is_sound):
        t = i * ms_per_sample
        if sound and start is None:
            start = t
        elif not sound and start is not None:
            duration = t - start
            if duration >= min_duration:
                ranges.append((round(start), round(t)))
            start = None

    # 结尾还在响
    if start is not None:
        duration = (len(is_sound) * ms_per_sample) - start
        if duration >= min_duration:
            ranges.append((round(start), round(len(is_sound) * ms_per_sample)))

    return ranges


# ===================== 执行 =====================
if __name__ == '__main__':
    # 1. 加载音频
    audio = AudioSegment.from_file(AUDIO_FILE)

    # 2. 拆分成 左(音轨1)、右(音轨2) 两个单声道
    track1 = audio.split_to_mono()[0]  # 左声道 = 音轨1
    track2 = audio.split_to_mono()[1]  # 右声道 = 音轨2

    # 3. 获取每条音轨的有声时间
    track1_ranges = get_sound_ranges(track1)
    track2_ranges = get_sound_ranges(track2)

    # ===================== 输出结果 =====================
    print("=" * 60)
    print("🎵 音轨1（左声道）有声时间区间：")
    for s, e in track1_ranges:
        print(f"  {s / 1000:.2f}s ~ {e / 1000:.2f}s  (时长：{(e - s) / 1000:.2f}s)")

    print("\n🎵 音轨2（右声道）有声时间区间：")
    for s, e in track2_ranges:
        print(f"  {s / 1000:.2f}s ~ {e / 1000:.2f}s  (时长：{(e - s) / 1000:.2f}s)")
    print("=" * 60)