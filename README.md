# 語音分類和轉錄 
# Speech diarization and Transcribe

會議秘書處會把會議內容錄音, 再回後將對話內容打成文字. 這個過程可以想象又費時並不容易, 坦白說價值亦不大.  最近公司IT部找了開發系統團隊, 開發了一個用來將會議上議會者的對話內容變成文字的AI程序. 目標是減輕秘書的工作.
其實如果我們懂得運用現今的大模型程序, 這個開發非常簡單. 以後我就用最簡單的方法, 把這個編程邏輯展示下來. 供大家參考. 

# Whisper Model
由OpenAI開發嘅Whisper模型係一種先進嘅自動語音識別（ ASR ）系統。 它於2022年9月作為開源軟件發佈。
The Whisper model, developed by OpenAI, is an advanced automatic speech recognition (ASR) system. It was released as open-source software in September 2022.

首先安裝所需要Whisper大語言模型

`pip install whisperx`

接着安裝所需語音識別工具, Pyannote.audio 是一個語言大模型專門設計用來做語音識別. 這次我直接用它就可以簡化大量工作. 他是一個免費的開源軟件, 可以在Hugging Face登記就可以免費試用. 

`pip install pyannote.audio`


安裝完畢後, 我們就可以開始了調用這些語言模型. Load the Whisper Model and Speaker Diarization toolkit

`import whisperx`

`from pyannote.audio import Pipeline`

`model = whisperx.load_model("large-v2", device="cpu", compute_type="float32")`

首先我們利用模型, 根據議會者的講話聲音, 進行分類. 

`pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="hf_apikey")`

假如我們已經有一個會議的錄音. 他以WAV format儲存在電腦上, output.wav. 這個模型可以預先輸入最多或最少會議會者人數, 以增加準確率. Let's transcribe the audio "output.wav"
 
`diarization = pipeline("./source/repos/whisper/output.wav", min_speakers=5, max_speakers=17)`

我們來看看這個模型分析之後的結果. 我這段錄音總共有20多人參與會議, 有10多位同事都發言. 模型根據語音分析, 這個會議有17個不同的人講話. 

`diarization`
![image](https://github.com/user-attachments/assets/5f5f2e1a-feb8-4488-b320-10d035c2af2d)

亦可以將它輸出為RTTM格式. dump the diarization output to disk using RTTM format

`with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)`

![image](https://github.com/user-attachments/assets/65781414-11b9-49d4-a46c-ec9585e09ca6)

接下來就是把會議語音變成文字記錄下來. 

`result = model.transcribe("./source/repos/whisper/output.wav", language="zh")`

然後再根據會議記錄及公事公辦的語音分析合併起來, 就能得出我們想要的結果. 

`transcription = result['segments']`

`last_segment = transcription[-1]`

`last_diarization_end = last_segment['end']`

`def find_best_match(diarization, start_time, end_time):`
    `best_match = None`
    `max_intersection = 0`

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turn_start = turn.start
        turn_end = turn.end

        # Calculate intersection manually
        intersection_start = max(start_time, turn_start)
        intersection_end = min(end_time, turn_end)

        if intersection_start < intersection_end:
            intersection_length = intersection_end - intersection_start
            if intersection_length > max_intersection:
                max_intersection = intersection_length
                best_match = (turn_start, turn_end, speaker)

    return best_match

    def merge_consecutive_segments(segments):
    merged_segments = []
    previous_segment = None

    for segment in segments:
        if previous_segment is None:
            previous_segment = segment
        else:
            if segment[0] == previous_segment[0]:
                # Merge segments of the same speaker that are consecutive
                previous_segment = (
                    previous_segment[0],
                    previous_segment[1],
                    segment[2],
                    previous_segment[3] + segment[3]
                )
            else:
                merged_segments.append(previous_segment)
                previous_segment = segment

    if previous_segment:
        merged_segments.append(previous_segment)

    return merged_segments

`speaker_transcriptions = []`

`for chunk in transcription:`

    chunk_start = chunk['start']    
    chunk_end = chunk['end']    
    segment_text = chunk['text'
    best_match = find_best_match(diarization, chunk_start, chunk_end)    
    if best_match:
        speaker = best_match[2]  # Extract the speaker label
        speaker_transcriptions.append((speaker, chunk_start, chunk_end, segment_text))
        
`speaker_transcriptions = merge_consecutive_segments(speaker_transcriptions)`

最後, 將合併結果輸出為CSV格式. 

`import numpy as np`

`array_speaker_transcriptions = np.array(speaker_transcriptions)`

`np.savetxt('array_speaker_transcriptions.csv', array_speaker_transcriptions, delimiter=',',fmt='%s', encoding='utf-8')`

![image](https://github.com/user-attachments/assets/85977851-8cd8-4ba9-a4cf-bfd69d006e83)


