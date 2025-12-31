import os
from typing import List


class Config:
    """
    Centralized configuration for the Interview Coach agent.
    Loads from environment variables with sensible defaults.
    """
    
    # LiveKit Connection
    LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "")
    LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")
    
    # Google Cloud API
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # VAD Configuration (Silero)
    vad_min_speech_duration: float = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.25"))
    vad_min_silence_duration: float = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.8"))
    
    # STT Configuration (Google Cloud)
    stt_language: str = os.getenv("STT_LANGUAGE", "en-US")
    stt_model: str = os.getenv("STT_MODEL", "latest_long")
    
    # LLM Configuration (Gemini)
    llm_model: str = os.getenv("LLM_MODEL", "gemma-3-27b")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # TTS Configuration (Google Cloud)
    tts_voice: str = os.getenv("TTS_VOICE", "en-US-Neural2-J")
    tts_speaking_rate: float = float(os.getenv("TTS_SPEAKING_RATE", "0.95"))
    
    # Interview Configuration
    max_interview_rounds: int = int(os.getenv("MAX_INTERVIEW_ROUNDS", "3"))
    questions_per_round: int = int(os.getenv("QUESTIONS_PER_ROUND", "2"))
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate that required environment variables are set"""
        errors = []
        
        if not cls.LIVEKIT_URL:
            errors.append("LIVEKIT_URL is required")
        if not cls.LIVEKIT_API_KEY:
            errors.append("LIVEKIT_API_KEY is required")
        if not cls.LIVEKIT_API_SECRET:
            errors.append("LIVEKIT_API_SECRET is required")
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """Print current configuration (masking secrets)"""
        print("=" * 60)
        print("Interview Coach Configuration")
        print("=" * 60)
        print(f"LiveKit URL: {cls.LIVEKIT_URL}")
        print(f"LiveKit API Key: {'*' * 8 if cls.LIVEKIT_API_KEY else 'NOT SET'}")
        print(f"Google API Key: {'*' * 8 if cls.GOOGLE_API_KEY else 'NOT SET'}")
        print(f"VAD Min Speech: {cls.vad_min_speech_duration}s")
        print(f"VAD Min Silence: {cls.vad_min_silence_duration}s")
        print(f"STT Language: {cls.stt_language}")
        print(f"STT Model: {cls.stt_model}")
        print(f"LLM Model: {cls.llm_model}")
        print(f"LLM Temperature: {cls.llm_temperature}")
        print(f"TTS Voice: {cls.tts_voice}")
        print(f"TTS Speaking Rate: {cls.tts_speaking_rate}")
        print(f"Max Rounds: {cls.max_interview_rounds}")
        print("=" * 60)