import logging
import json
import asyncio
from typing import Optional
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    RunContext,
)
from livekit.plugins import deepgram, google, silero

# Import our custom modules
from interview_tools import InterviewTools
from interview_state import InterviewState
from config import Config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("interview-coach")


class InterviewCoach:
    """Orchestrates the intelligent interview coach agent"""
    
    def __init__(self):
        self.config = Config()
        self.state: Optional[InterviewState] = None
        self.session: Optional[AgentSession] = None
        
    def _build_instructions(self) -> str:
        """Construct the agent instructions dynamically based on interview state"""
        base_instructions = (
            "You are a Senior AI/ML Engineer conducting a technical interview "
            "for a Generative AI and LLM Engineer position. Your goal is to assess the candidate's "
            "knowledge of transformers, LLMs, RAG, fine-tuning, and agentic AI systems.\n\n"
            "INSTRUCTIONS:\n"
            "- Ask ONE question at a time\n"
            "- Keep responses BRIEF (1-2 sentences max) to maintain conversational flow\n"
            "- Use the get_definition tool to verify technical answers\n"
            "- Be professional yet encouraging\n"
            "- If the candidate asks to move on or finish, comply gracefully\n"
            "- When all rounds are complete, use the submit_final_score tool\n\n"
        )
        
        if self.state:
            state_context = (
                f"CURRENT STATE:\n"
                f"- Round: {self.state.round}/{self.state.max_rounds}\n"
                f"- Focus Area: {self.state.get_current_focus()}\n"
                f"- Questions Asked: {len(self.state.question_history)}\n\n"
            )
            return base_instructions + state_context
        
        return base_instructions
    
    async def _check_interview_progression(self, room: rtc.Room):
        """Background task to manage interview state and send feedback"""
        try:
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if self.state and self.state.is_complete():
                    logger.info("Interview complete - generating feedback report")
                    await self._send_feedback_report(room)
                    break
                    
        except asyncio.CancelledError:
            logger.info("Progression check cancelled")
    
    async def _send_feedback_report(self, room: rtc.Room):
        """Generate and send structured feedback via data channel"""
        if not self.state:
            return
        
        feedback_data = {
            "type": "FEEDBACK_REPORT",
            "data": {
                "overall_score": self.state.calculate_overall_score(),
                "round_scores": self.state.get_round_scores(),
                "strengths": self.state.strengths,
                "areas_for_improvement": self.state.weaknesses,
                "transcript": self.state.get_full_transcript(),
                "duration_minutes": self.state.get_duration_minutes()
            }
        }
        
        # Send via LiveKit data channel
        payload = json.dumps(feedback_data).encode('utf-8')
        await room.local_participant.publish_data(
            payload=payload,
            topic="interview_feedback",
            reliable=True
        )
        logger.info("Feedback report sent to client")
    
    async def entrypoint(self, ctx: JobContext):
        """Main agent entrypoint - called for each interview session"""
        logger.info(f"Starting interview session for room: {ctx.room.name}")
        
        # Initialize interview state
        self.state = InterviewState()
        
        # Connect to the LiveKit room
        await ctx.connect()
        logger.info("Connected to room")
        
        # Initialize the agent with tools
        agent = Agent(
            instructions=self._build_instructions(),
            tools=InterviewTools(self.state).get_tools(),
        )
        
        # Create the agent session with voice pipeline
        self.session = AgentSession(
            vad=silero.VAD.load(
                min_speech_duration=self.config.vad_min_speech_duration,
                min_silence_duration=self.config.vad_min_silence_duration,
            ),
            stt=deepgram.STT(),
            llm=google.LLM(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
            ),
            tts=deepgram.TTS(model="aura-asteria-en")
        )
        
        # Setup event hooks
        self._setup_event_hooks()
        
        # Start the session
        await self.session.start(agent=agent, room=ctx.room)
        logger.info("Agent session started")
        
        # Start background task for interview progression
        progression_task = asyncio.create_task(
            self._check_interview_progression(ctx.room)
        )
        
        # Send welcome message
        welcome_msg = (
            "Hello! I'm your Intelligent Interview Coach. "
            "I'll be conducting a technical interview to assess your knowledge "
            "of Generative AI, Large Language Models, and Agentic AI systems. "
            "We'll go through three rounds covering fundamentals, advanced techniques, "
            "and production deployment. Are you ready to begin?"
        )
        await self.session.generate_reply(instructions=f"Say: {welcome_msg}")
        
        # Keep session alive until disconnection
        try:
            await asyncio.Future()  # Run until cancelled
        except asyncio.CancelledError:
            logger.info("Session ending")
            progression_task.cancel()
            try:
                await progression_task
            except asyncio.CancelledError:
                pass
    
    def _setup_event_hooks(self):
        """Configure event handlers for agent lifecycle"""
        
        @self.session.on("user_input_transcribed")
        def on_user_input(event):
            """Log user transcript for debugging and analytics"""
            transcript = event.transcript
            logger.info(f"User said: {transcript}")
            if self.state and event.is_final:
                self.state.add_transcript("user", transcript)
        
        @self.session.on("conversation_item_added")
        def on_conversation_item(event):
            """Log conversation items"""
            item = event.item
            if item.role == "assistant":
                logger.info(f"Agent said: {item.text_content}")
                if self.state and item.text_content:
                    self.state.add_transcript("agent", item.text_content)
        
        @self.session.on("user_state_changed")
        def on_user_state(event):
            """Handle user state changes"""
            logger.info(f"User state: {event.new_state}")


def run():
    """Entry point for the worker"""
    coach = InterviewCoach()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=coach.entrypoint,
        )
    )


if __name__ == "__main__":
    run()