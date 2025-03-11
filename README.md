# 語音分類和翻譯 
# Speech diarization and Transcribe

會議秘書處會把會議內容錄音, 再回後將對話內容打成文字. 這個過程對我們來說,尤其IT人,可以想象並不容易.  最近公司IT部找了開發系統團隊, 開發了一個用來將會議上議會者的對話內容變成文字的AI程序. 目標是減輕秘書的工作.
其實如果我們懂得運用現今的大模型程序, 這個開發非常簡單. 以後我就用最簡單的方法, 把這個編程邏輯展示下來. 供大家參考. 

# Whisper Model
由OpenAI開發嘅Whisper模型係一種先進嘅自動語音識別（ ASR ）系統。 它於2022年9月作為開源軟件發佈。
The Whisper model, developed by OpenAI, is an advanced automatic speech recognition (ASR) system. It was released as open-source software in September 2022.

首先安裝所需要Whisper大語言模型

`pip install whisperx`


`import whisperx
from pyannote.audio import Pipeline`

# Load the WhisperX model
`model = whisperx.load_model("large-v2", device="cpu", compute_type="float32")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="hf_apikey")`
# Transcribe audio
`diarization = pipeline("./source/repos/whisper/output.wav", min_speakers=5, max_speakers=17)`


# dump the diarization output to disk using RTTM format
`with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)`
