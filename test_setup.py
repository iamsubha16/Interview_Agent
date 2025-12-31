#!/usr/bin/env python3
"""
Test script to verify LiveKit Agents installation and configuration.
Run this before starting your agent to catch setup issues early.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")

    try:
        import livekit
        try:
            from importlib.metadata import version
            print(f"✓ livekit version: {version('livekit')}")
        except Exception:
            print("✓ livekit imported (version unavailable)")
    except ImportError as e:
        print(f"✗ Failed to import livekit: {e}")
        return False

    try:
        from livekit import agents
        print("✓ livekit.agents imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import livekit.agents: {e}")
        return False

    try:
        from livekit.plugins import google
        print("✓ livekit.plugins.google imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import google plugin: {e}")
        return False

    try:
        from livekit.plugins import silero
        print("✓ livekit.plugins.silero imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import silero plugin: {e}")
        return False

    return True


def test_environment():
    """Test that required environment variables are set"""
    print("\nTesting environment variables...")
    
    required_vars = {
        "LIVEKIT_URL": "LiveKit server URL",
        "LIVEKIT_API_KEY": "LiveKit API key",
        "LIVEKIT_API_SECRET": "LiveKit API secret",
        "GOOGLE_API_KEY": "Google API key for Gemini/STT/TTS",
    }
    
    all_set = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask the value for security
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✓ {var} is set ({masked})")
        else:
            print(f"✗ {var} is NOT set - {description}")
            all_set = False
    
    return all_set


def test_config():
    """Test that config module loads correctly"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        config = Config()
        print("✓ Config module loaded successfully")
        
        # Print some config values
        print(f"  - VAD min speech duration: {config.vad_min_speech_duration}s")
        print(f"  - VAD min silence duration: {config.vad_min_silence_duration}s")
        print(f"  - LLM model: {config.llm_model}")
        print(f"  - TTS voice: {config.tts_voice}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


def test_custom_modules():
    """Test that custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from interview_state import InterviewState
        print("✓ interview_state module imported")
    except ImportError as e:
        print(f"✗ Failed to import interview_state: {e}")
        return False
    
    try:
        from interview_tools import InterviewTools
        print("✓ interview_tools module imported")
    except ImportError as e:
        print(f"✗ Failed to import interview_tools: {e}")
        return False
    
    return True


def test_state_creation():
    """Test that we can create interview state"""
    print("\nTesting state creation...")
    
    try:
        from interview_state import InterviewState
        state = InterviewState()
        print(f"✓ Created InterviewState")
        print(f"  - Current round: {state.round}/{state.max_rounds}")
        print(f"  - Focus area: {state.get_current_focus()}")
        return True
    except Exception as e:
        print(f"✗ Failed to create state: {e}")
        return False


def test_tools_creation():
    """Test that we can create interview tools"""
    print("\nTesting tools creation...")
    
    try:
        from interview_state import InterviewState
        from interview_tools import InterviewTools
        
        state = InterviewState()
        tools = InterviewTools(state)
        tool_list = tools.get_tools()
        
        print(f"✓ Created InterviewTools")
        print(f"  - Number of tools: {len(tool_list)}")
        print(f"  - Tools: {[t.__name__ for t in tool_list]}")
        return True
    except Exception as e:
        print(f"✗ Failed to create tools: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("LiveKit Interview Coach - Setup Verification")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Environment Variables", test_environment),
        ("Configuration", test_config),
        ("Custom Modules", test_custom_modules),
        ("State Creation", test_state_creation),
        ("Tools Creation", test_tools_creation),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your environment is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Ensure .env file exists with all required variables")
        print("- Run: pip install -r requirements.txt")
        print("- Check that you're using Python 3.10+")
        return 1


if __name__ == "__main__":
    sys.exit(main())