"""ASR adapters with hallucination guards and VAD support."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .backend_config import (
    ASR_AUTO_BACKEND,
    ASR_BACKEND_CHOICES,
    ASR_BACKEND_MODULES,
    ASR_FAST_BACKEND,
    ASR_INDIC_CONFORMER_BACKEND,
    ASR_INDICWHISPER_BACKEND,
    ASR_WHISPER_BACKEND,
    INDIC_CONFORMER_LANGUAGES,
    INDIC_CONFORMER_MODEL_ID,
    INDICWHISPER_DEFAULT_SIZE,
    INDICWHISPER_LANGUAGES,
    INDICWHISPER_MODEL_FAMILY,
    INDICWHISPER_MODEL_SIZES,
    BackendResolution,
    normalize_asr_language,
)
from .exceptions import MissingDependencyError, ProcessingError
from .media import extract_audio, media_duration_seconds
from .models import TranscriptSegment

# Whisper sometimes hallucinates these phrases for silent or noisy stretches.
_DEFAULT_HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "subscribe to the channel",
    "please like and subscribe",
    "अधिक जानकारी के लिए सब्सक्राइब",
    "ご視聴ありがとうございました",
    "♪♪",
)


def transcribe_video(
    video_path: Path,
    *,
    backend: str = ASR_AUTO_BACKEND,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
    vad: bool = False,
    no_speech_threshold: float = 0.6,
    drop_hallucinations: bool = True,
    initial_prompt: str | None = None,
    conformer_decoder: str = "ctc",
) -> list[TranscriptSegment]:
    resolution = resolve_asr_backend(backend, language=language)
    transcriber = _ASR_TRANSCRIBERS.get(resolution.selected)
    if transcriber is None:
        raise MissingDependencyError(f"Unsupported ASR backend: {backend}")
    segments = transcriber(
        video_path,
        model_name=model_name,
        language=language,
        device=device,
        vad=vad,
        initial_prompt=initial_prompt,
        conformer_decoder=conformer_decoder,
    )

    return _post_process_segments(
        segments,
        no_speech_threshold=no_speech_threshold,
        drop_hallucinations=drop_hallucinations,
    )


def resolve_asr_backend(
    requested_backend: str,
    *,
    language: str | None,
    module_available: Callable[[str], bool] | None = None,
) -> BackendResolution:
    requested = (requested_backend or ASR_AUTO_BACKEND).strip()
    if requested not in ASR_BACKEND_CHOICES:
        raise MissingDependencyError(f"Unsupported ASR backend: {requested_backend}")
    if requested != ASR_AUTO_BACKEND:
        return BackendResolution(
            requested=requested,
            selected=requested,
            reason=f"user selected --asr-backend {requested}",
        )

    available = module_available or _module_available
    normalized_language = normalize_asr_language(language)
    if normalized_language in INDICWHISPER_LANGUAGES or normalized_language == "auto":
        if available(ASR_BACKEND_MODULES[ASR_INDICWHISPER_BACKEND][0]):
            detail = (
                "language auto; Hindi/Kannada are the primary target languages"
                if normalized_language == "auto"
                else f"language {normalized_language} is supported by IndicWhisper"
            )
            return BackendResolution(
                requested=requested,
                selected=ASR_INDICWHISPER_BACKEND,
                reason=detail,
            )
        return BackendResolution(
            requested=requested,
            selected=ASR_WHISPER_BACKEND,
            reason="IndicWhisper dependencies are not installed; falling back to Whisper",
        )
    return BackendResolution(
        requested=requested,
        selected=ASR_WHISPER_BACKEND,
        reason=f"language {normalized_language} is not in the IndicWhisper support set",
    )


def asr_backend_available(backend: str) -> bool:
    modules = ASR_BACKEND_MODULES.get(backend)
    return bool(modules and _module_available(modules[0]))


def asr_backend_install_hint(backend: str) -> str:
    modules = ASR_BACKEND_MODULES.get(backend)
    if modules is None:
        return ""
    return modules[1]


def resolve_indicwhisper_model_id(model_name: str, language: str | None) -> str:
    raw_name = (model_name or "").strip()
    normalized = raw_name.lower()
    if normalized in {"", "base", "tiny"}:
        normalized = INDICWHISPER_DEFAULT_SIZE
    if normalized in INDICWHISPER_MODEL_SIZES:
        lang = normalize_asr_language(language)
        env_candidates = []
        if lang != "auto":
            env_candidates.append(f"BURNSUB_INDICWHISPER_{lang.upper()}_{normalized.upper()}_MODEL_ID")
        env_candidates.extend(
            [
                f"BURNSUB_INDICWHISPER_{normalized.upper()}_MODEL_ID",
                "BURNSUB_INDICWHISPER_MODEL_ID",
            ]
        )
        for env_var in env_candidates:
            value = os.environ.get(env_var)
            if value:
                return value
        family = os.environ.get("BURNSUB_INDICWHISPER_MODEL_FAMILY", INDICWHISPER_MODEL_FAMILY)
        if normalized == INDICWHISPER_DEFAULT_SIZE:
            return family
        return f"{family}-{normalized}"
    return raw_name


def resolve_indic_conformer_model_id(model_name: str) -> str:
    raw_name = (model_name or "").strip()
    if not raw_name or raw_name in {"base", "small", "medium", "large"}:
        return os.environ.get("BURNSUB_INDIC_CONFORMER_MODEL_ID", INDIC_CONFORMER_MODEL_ID)
    return raw_name


def _transcribe_with_openai_whisper(
    video_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
    initial_prompt: str | None,
    **_unused: Any,
) -> list[TranscriptSegment]:
    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "Python package 'openai-whisper' is not installed. "
            "Install with: python -m pip install -e '.[asr]'"
        ) from exc

    kwargs: dict[str, Any] = {}
    if device and device != "auto":
        kwargs["device"] = device
    try:
        model = whisper.load_model(model_name, **kwargs)
        transcribe_kwargs: dict[str, Any] = {
            "language": None if language in {None, "auto"} else language,
            "task": "transcribe",
        }
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        if device in {None, "auto", "cpu"}:
            transcribe_kwargs["fp16"] = False
        transcription = model.transcribe(str(video_path), **transcribe_kwargs)
    except Exception as exc:
        raise ProcessingError(f"Whisper transcription failed: {exc}") from exc
    return _segments_from_whisper_payload(transcription)


def _transcribe_with_faster_whisper(
    video_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
    vad: bool,
    initial_prompt: str | None,
    **_unused: Any,
) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "Python package 'faster-whisper' is not installed. "
            "Install with: python -m pip install -e '.[asr-fast]'"
        ) from exc

    try:
        model = WhisperModel(model_name, device="auto" if device in {None, "auto"} else device)
        kwargs: dict[str, Any] = {
            "language": None if language in {None, "auto"} else language,
            "task": "transcribe",
            "vad_filter": vad,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segments, _info = model.transcribe(str(video_path), **kwargs)
        results: list[TranscriptSegment] = []
        for index, segment in enumerate(segments):
            results.append(
                TranscriptSegment(
                    index=index,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(getattr(segment, "text", "") or "").strip(),
                    confidence=_safe_float(getattr(segment, "avg_logprob", None)),
                    no_speech_prob=_safe_float(getattr(segment, "no_speech_prob", None)),
                )
            )
        return results
    except Exception as exc:
        raise ProcessingError(f"faster-whisper transcription failed: {exc}") from exc


def _transcribe_with_indicwhisper(
    video_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
    initial_prompt: str | None,
    **_unused: Any,
) -> list[TranscriptSegment]:
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import pipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "IndicWhisper requires 'transformers' and 'torch'. "
            "Install with: python -m pip install -e '.[asr-indic]'"
        ) from exc

    model_id = resolve_indicwhisper_model_id(model_name, language)
    try:
        pipe_kwargs: dict[str, Any] = {"model": model_id}
        pipeline_device = _transformers_pipeline_device(device, torch)
        if pipeline_device is not None:
            pipe_kwargs["device"] = pipeline_device
        dtype = _transformers_torch_dtype(device, torch)
        if dtype is not None:
            pipe_kwargs["torch_dtype"] = dtype
        asr_pipeline = pipeline("automatic-speech-recognition", **pipe_kwargs)
        call_kwargs: dict[str, Any] = {
            "return_timestamps": True,
            "chunk_length_s": 30,
        }
        generate_kwargs = _whisper_generate_kwargs(
            asr_pipeline,
            language=language,
            initial_prompt=initial_prompt,
        )
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs
        try:
            payload = asr_pipeline(str(video_path), **call_kwargs)
        except TypeError:
            call_kwargs.pop("chunk_length_s", None)
            payload = asr_pipeline(str(video_path), **call_kwargs)
    except Exception as exc:
        raise ProcessingError(f"IndicWhisper transcription failed: {exc}") from exc
    return _segments_from_transformers_payload(
        payload,
        duration=_safe_media_duration(video_path),
    )


def _transcribe_with_indic_conformer(
    video_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str | None,
    conformer_decoder: str,
    **_unused: Any,
) -> list[TranscriptSegment]:
    decoder = conformer_decoder.strip().lower()
    if decoder not in {"ctc", "rnnt"}:
        raise ProcessingError("--asr-conformer-decoder must be one of: ctc, rnnt")
    lang = normalize_asr_language(language)
    if lang == "auto":
        lang = "hi"
    if lang not in INDIC_CONFORMER_LANGUAGES:
        raise ProcessingError(
            f"IndicConformer does not support ASR language '{lang}'. "
            "Use one of the IN-22 language codes such as hi or kn."
        )

    model_id = resolve_indic_conformer_model_id(model_name)
    with TemporaryDirectory(prefix="burnsub-asr-") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio(video_path, wav_path)
        try:
            text = _run_indic_conformer(model_id, wav_path, lang, decoder, device=device)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ProcessingError(f"IndicConformer transcription failed: {exc}") from exc

    duration = _safe_media_duration(video_path) or 0.0
    return [
        TranscriptSegment(
            index=0,
            start=0.0,
            end=max(duration, 0.0),
            text=text.strip(),
        )
    ]


def _segments_from_whisper_payload(payload: dict[str, Any]) -> list[TranscriptSegment]:
    raw_segments = payload.get("segments") or []
    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(raw_segments):
        text = segment.get("text") or ""
        segments.append(
            TranscriptSegment(
                index=index,
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(text).strip(),
                confidence=_safe_float(segment.get("avg_logprob")),
                no_speech_prob=_safe_float(segment.get("no_speech_prob")),
            )
        )
    return segments


def _segments_from_transformers_payload(
    payload: Any,
    *,
    duration: float | None = None,
) -> list[TranscriptSegment]:
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return []

    chunks = payload.get("chunks") or []
    segments: list[TranscriptSegment] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        start, end = _timestamp_pair(chunk.get("timestamp"))
        if start is None:
            start = 0.0
        if end is None:
            end = duration if duration is not None else start
        text = str(chunk.get("text") or "").strip()
        segments.append(TranscriptSegment(index=index, start=start, end=end, text=text))
    if segments:
        return segments

    text = str(payload.get("text") or "").strip()
    if not text:
        return []
    return [
        TranscriptSegment(
            index=0,
            start=0.0,
            end=max(float(duration or 0.0), 0.0),
            text=text,
        )
    ]


def _post_process_segments(
    segments: Iterable[TranscriptSegment],
    *,
    no_speech_threshold: float,
    drop_hallucinations: bool,
) -> list[TranscriptSegment]:
    cleaned: list[TranscriptSegment] = []
    for segment in segments:
        if segment.end < segment.start:
            continue
        if segment.no_speech_prob is not None and segment.no_speech_prob >= no_speech_threshold:
            continue
        if drop_hallucinations and _looks_like_hallucination(segment.text):
            continue
        cleaned.append(segment)
    for new_index, segment in enumerate(cleaned):
        segment.index = new_index
    return cleaned


def _looks_like_hallucination(text: str) -> bool:
    if not text:
        return False
    folded = text.casefold().strip()
    return any(phrase in folded for phrase in _DEFAULT_HALLUCINATION_PHRASES)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_pair(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    return _safe_float(value[0]), _safe_float(value[1])


def _safe_media_duration(video_path: Path) -> float | None:
    try:
        return media_duration_seconds(video_path)
    except Exception:
        return None


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _transformers_pipeline_device(device: str | None, torch_module) -> int | str | None:
    normalized = (device or "auto").lower()
    if normalized == "cpu":
        return -1
    if normalized in {"cuda", "gpu"}:
        return 0
    if normalized == "mps":
        return "mps"
    if normalized == "auto":
        try:
            if torch_module.cuda.is_available():
                return 0
            if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                return "mps"
        except Exception:
            return -1
        return -1
    return device


def _transformers_torch_dtype(device: str | None, torch_module):
    normalized = (device or "auto").lower()
    try:
        if normalized in {"cuda", "gpu"}:
            return torch_module.float16
        if normalized == "auto" and torch_module.cuda.is_available():
            return torch_module.float16
    except Exception:
        return None
    return None


def _whisper_generate_kwargs(
    asr_pipeline,
    *,
    language: str | None,
    initial_prompt: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    lang = normalize_asr_language(language)
    tokenizer = getattr(asr_pipeline, "tokenizer", None)
    if lang != "auto" and hasattr(tokenizer, "get_decoder_prompt_ids"):
        kwargs["forced_decoder_ids"] = tokenizer.get_decoder_prompt_ids(
            language=lang,
            task="transcribe",
        )
    processor = getattr(asr_pipeline, "processor", None)
    if initial_prompt and hasattr(processor, "get_prompt_ids"):
        kwargs["prompt_ids"] = processor.get_prompt_ids(initial_prompt)
    return kwargs


def _run_indic_conformer(
    model_id: str,
    wav_path: Path,
    language: str,
    decoder: str,
    *,
    device: str | None,
) -> str:
    nemo_error: Exception | None = None
    if _module_available("nemo.collections.asr"):
        try:
            return _run_indic_conformer_with_nemo(
                model_id,
                wav_path,
                language,
                decoder,
            )
        except Exception as exc:  # pragma: no cover - depends on NeMo model runtime
            nemo_error = exc

    try:
        return _run_indic_conformer_with_transformers(
            model_id,
            wav_path,
            language,
            decoder,
            device=device,
        )
    except MissingDependencyError:
        if nemo_error is not None:
            raise ProcessingError(
                "IndicConformer failed with NeMo and the transformers fallback is unavailable: "
                f"{nemo_error}"
            ) from nemo_error
        raise


def _run_indic_conformer_with_nemo(
    model_id: str,
    wav_path: Path,
    language: str,
    decoder: str,
) -> str:
    try:
        import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "IndicConformer requires NeMo. "
            "Install with: python -m pip install -e '.[asr-conformer]'"
        ) from exc

    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    transcribe = model.transcribe
    for kwargs in (
        {"batch_size": 1, "logprobs": False, "language_id": language, "decoder_type": decoder},
        {"batch_size": 1, "logprobs": False, "language_id": language, "decoding": decoder},
        {"batch_size": 1, "logprobs": False, "language_id": language},
    ):
        try:
            result = transcribe([str(wav_path)], **kwargs)
            return _text_from_asr_result(result)
        except TypeError:
            continue
    result = transcribe([str(wav_path)])
    return _text_from_asr_result(result)


def _run_indic_conformer_with_transformers(
    model_id: str,
    wav_path: Path,
    language: str,
    decoder: str,
    *,
    device: str | None,
) -> str:
    try:
        import torch  # type: ignore[import-not-found]
        import torchaudio  # type: ignore[import-not-found]
        from transformers import AutoModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "IndicConformer requires NeMo or the Hugging Face runtime "
            "('transformers', 'torch', and 'torchaudio'). "
            "Install with: python -m pip install -e '.[asr-conformer]'"
        ) from exc

    target_device = _torch_model_device(device, torch)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    if target_device is not None and hasattr(model, "to"):
        model = model.to(target_device)
    wav, sample_rate = torchaudio.load(str(wav_path))
    wav = torch.mean(wav, dim=0, keepdim=True)
    if int(sample_rate) != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=int(sample_rate), new_freq=16000)
        wav = resampler(wav)
    if target_device is not None and hasattr(wav, "to"):
        wav = wav.to(target_device)
    with torch.no_grad():
        result = model(wav, language, decoder)
    return _text_from_asr_result(result)


def _torch_model_device(device: str | None, torch_module) -> str | None:
    normalized = (device or "auto").lower()
    if normalized == "auto":
        try:
            if torch_module.cuda.is_available():
                return "cuda"
            if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                return "mps"
        except Exception:
            return None
        return None
    if normalized == "cpu":
        return None
    return device


def _text_from_asr_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list | tuple):
        if not result:
            return ""
        return _text_from_asr_result(result[0])
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    if isinstance(result, dict):
        for key in ("text", "pred_text", "transcription"):
            if key in result:
                return str(result[key])
    return str(result)


_ASR_TRANSCRIBERS: dict[str, Callable[..., list[TranscriptSegment]]] = {
    ASR_WHISPER_BACKEND: _transcribe_with_openai_whisper,
    ASR_FAST_BACKEND: _transcribe_with_faster_whisper,
    ASR_INDICWHISPER_BACKEND: _transcribe_with_indicwhisper,
    ASR_INDIC_CONFORMER_BACKEND: _transcribe_with_indic_conformer,
}
