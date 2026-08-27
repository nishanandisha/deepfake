---
license: cc-by-4.0
task_categories:
- audio-classification
language:
- en
tags:
- deepfake
- synthetic-speech
- audio
- tts
- voice-cloning
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: label
    dtype:
      class_label:
        names:
          '0': real
          '1': fake
  splits:
  - name: train
    num_bytes: 891744087.486
    num_examples: 1866
  download_size: 574464278
  dataset_size: 891744087.486
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Deepfake Audio Detection Dataset (v4)

## Dataset Description

This dataset contains 1,866 audio samples (933 real, 933 synthetic) for training deepfake audio detection models. It is specifically designed for binary classification tasks to distinguish between authentic human speech and AI-generated synthetic audio.

### What's New in v4

- **52% larger**: Increased from 1,224 to 1,866 samples (642 new samples)
- **Expanded TTS coverage**: Added Hume AI as 6th synthetic voice platform
- **More diverse real audio**: Expanded YouTube source coverage with improved speaker diversity
- **Advanced processing**: Sophisticated two-pass audio splitting algorithm with concatenation
- **Better data utilization**: Rescued short audio segments through intelligent concatenation (Pass 1.5)
- **Consistent quality**: All chunks are 2.5-13 seconds with natural speech boundaries
- **Traceable provenance**: File naming conventions indicate processing method (_c_ for concatenated, _p2_ for sub-chunked)

### Dataset Summary

- **Total Samples**: 1,866 FLAC audio files
- **Real Audio**: 933 samples from 14 YouTube recordings
- **Synthetic Audio**: 933 samples generated using:
  - Amazon Polly TTS (prefix: `po_`) - 209 samples
  - ElevenLabs voice synthesis (prefix: `el_`) - 173 samples
  - Hexgrad Kokoro TTS (prefix: `hg_`) - 68 samples
  - Hume AI TTS (prefix: `hu_`) - 116 samples
  - Luvvoice TTS (prefix: `lv_`) - 156 samples
  - Speechify TTS (prefix: `sp_`) - 211 samples
- **Format**: FLAC (lossless audio compression, 16kHz mono)
- **Chunk Duration**: 2.5-13 seconds (optimized for model training)
- **Language**: English
- **Task**: Binary audio classification (real vs fake)

### Dataset Structure

```text
data/
├── fake/          # 933 synthetic audio samples
│   ├── el_*.flac  # ElevenLabs generated (173 samples)
│   ├── hg_*.flac  # Hexgrad Kokoro generated (68 samples)
│   ├── hu_*.flac  # Hume AI generated (116 samples)
│   ├── lv_*.flac  # Luvvoice generated (156 samples)
│   ├── po_*.flac  # Amazon Polly generated (209 samples)
│   └── sp_*.flac  # Speechify generated (211 samples)
└── real/          # 933 authentic audio samples
    └── yt_*.flac  # YouTube recordings (14 source videos)
```

### Audio Processing

All audio files have been processed using a sophisticated two-pass splitting algorithm:

1. **Pass 1 - Silence Detection**: Splits audio at natural pauses using silence detection (300ms threshold at -40 dBFS)
2. **Pass 1.5 - Concatenation**: Combines short segments (<2.5s) to reach minimum duration, creating more usable training data from fake audio sources
3. **Pass 2 - VAD Sub-chunking**: Uses Voice Activity Detection to intelligently split long segments (>13s) at speech boundaries

**File Naming Conventions:**

- Regular chunks: `filename_part_001.flac` (direct from Pass 1)
- Concatenated chunks: `filename_c_part_002.flac` (combined short segments)
- Sub-chunked segments: `filename_p2_part_003.flac` (VAD-split long segments)

## Usage

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("garystafford/deepfake-audio-detection")

# Access splits
train_dataset = dataset["train"]

# The dataset will have 'audio' and 'label' columns
# label: 0 = real, 1 = fake (alphabetically assigned by audiofolder)
```

### Use with Wav2Vec2

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from datasets import load_dataset, Audio

# Load dataset
dataset = load_dataset("garystafford/deepfake-audio-detection")

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base"
)

# Prepare audio data
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Extract features
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    return inputs

dataset = dataset.map(preprocess_function, batched=True)
```

## Data Collection

### Real Audio (YouTube)

- **Source**: Public YouTube videos
- **Content**: Natural human speech from various speakers and contexts
- **Processing Pipeline**:
  1. Extracted audio from MP4 videos using FFmpeg
  2. Converted to FLAC format (16kHz mono)
  3. Split into 2.5-13 second chunks using silence detection
  4. Balanced to match synthetic sample count

### Synthetic Audio

Generated from the same source text using multiple TTS platforms to ensure diversity:

- **Amazon Polly**: Standard and neural TTS voices with multiple speaker profiles
- **ElevenLabs**: High-quality voice synthesis with various voice presets and emotional tones
- **Hexgrad Kokoro**: Open-weight TTS model with 82 million parameters
- **Hume AI**: Empathic voice interface with emotion-aware speech synthesis
- **Luvvoice**: Online text-to-speech with multiple voice options
- **Speechify**: Commercial TTS service with natural-sounding voices

**Processing Pipeline**:

1. Generated audio from text using each TTS platform
2. Converted to FLAC format (16kHz mono)
3. Split using two-pass algorithm (silence detection + concatenation)
4. Resulted in 933 balanced synthetic samples

## Intended Use

### Primary Use Cases

- Training binary classifiers for deepfake audio detection
- Fine-tuning pre-trained audio models (Wav2Vec2, HuBERT, etc.)
- Research in synthetic speech detection
- Benchmarking audio authenticity detection systems

### Out-of-Scope Use

- This dataset is relatively small and should be used for fine-tuning or evaluation rather than training from scratch
- Not suitable for speaker identification or verification tasks
- Limited to English language samples

## Limitations

- **Dataset Size**: While improved to 1,866 samples, this is still relatively small for training from scratch. Best suited for fine-tuning pre-trained models.
- **TTS Platform Coverage**: Limited to six specific TTS platforms. May not generalize to all synthetic speech generation techniques or newer models.
- **Language**: English only - may not generalize to other languages
- **Temporal Bias**: Samples collected in December 2024 - newer TTS systems may produce different artifacts
- **Audio Characteristics**:
  - All chunks are 2.5-13 seconds (may not represent longer-form deepfakes)
  - Some fake audio chunks are concatenated from shorter segments (marked with `_c_`)
  - Limited background noise or acoustic diversity
- **Detection Arms Race**: Deepfake generation techniques evolve rapidly; models trained on this data may not detect future synthetic audio

## Ethical Considerations

This dataset is intended for defensive purposes to improve detection of synthetic audio. Users should:

- Use responsibly for research and detection systems
- Not use to create misleading or harmful synthetic audio
- Consider privacy implications when using real audio samples
- Be aware that detection systems trained on this data may have limitations

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{deepfake_audio_detection_v4_2024,
  author = {Gary Stafford},
  title = {Deepfake Audio Detection Dataset v4},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/garystafford/deepfake-audio-detection}
}
```

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC-BY-4.0).

## Contact

For questions or issues, please open an issue in the associated GitHub repository.
