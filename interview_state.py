import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ScoreRecord:
    """Record of a single scored answer"""
    question: str
    score: int
    feedback: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TranscriptEntry:
    """Single turn in the conversation"""
    role: str  # 'user' or 'agent'
    content: str
    timestamp: float = field(default_factory=time.time)


class InterviewState:
    """
    Manages the state and progression of an interview session.
    Tracks rounds, scores, transcript, and assessment data.
    """
    
    def __init__(self, max_rounds: int = 3):
        self.round = 1
        self.max_rounds = max_rounds
        self.start_time = time.time()
        
        # Round focus areas (can be customized per interview type)
        self.round_focuses = {
            1: "Fundamentals (Transformers, Attention, Tokenization)",
            2: "Advanced Techniques (RAG, Fine-tuning, Prompt Engineering)",
            3: "Agentic AI & Production (Tool Use, Multi-agent, Deployment)"
        }
        
        # Score tracking
        self.scores: List[ScoreRecord] = []
        self.question_history: List[str] = []
        
        # Final assessment
        self.overall_score: Optional[int] = None
        self.strengths: Optional[str] = None
        self.weaknesses: Optional[str] = None
        self.completed = False
        
        # Conversation transcript
        self.transcript: List[TranscriptEntry] = []
    
    def get_current_focus(self) -> str:
        """Get the focus area for the current round"""
        return self.round_focuses.get(
            self.round,
            f"Round {self.round}"
        )
    
    def add_score(self, question: str, score: int, feedback: str):
        """Record a scored answer"""
        record = ScoreRecord(
            question=question,
            score=score,
            feedback=feedback
        )
        self.scores.append(record)
        
        if question not in self.question_history:
            self.question_history.append(question)
    
    def should_advance_round(self) -> bool:
        """
        Determine if enough questions have been asked to move to next round.
        Strategy: Advance after 2-3 questions per round.
        """
        questions_this_round = sum(
            1 for q in self.question_history[-5:]  # Look at recent questions
        )
        return questions_this_round >= 2 and self.round < self.max_rounds
    
    def next_round(self):
        """Advance to the next interview round"""
        if self.round < self.max_rounds:
            self.round += 1
    
    def add_transcript(self, role: str, content: str):
        """Add an entry to the conversation transcript"""
        self.transcript.append(
            TranscriptEntry(role=role, content=content)
        )
    
    def finalize_assessment(
        self,
        overall_score: int,
        strengths: str,
        areas_for_improvement: str
    ):
        """Record the final assessment"""
        self.overall_score = overall_score
        self.strengths = strengths
        self.weaknesses = areas_for_improvement
        self.completed = True
    
    def is_complete(self) -> bool:
        """Check if the interview has been finalized"""
        return self.completed
    
    def calculate_overall_score(self) -> Optional[float]:
        """Calculate average score from all recorded answers"""
        if not self.scores:
            return self.overall_score
        
        avg = sum(s.score for s in self.scores) / len(self.scores)
        
        # Return the LLM's overall score if provided, otherwise use average
        return self.overall_score or round(avg, 1)
    
    def get_round_scores(self) -> Dict[int, Dict]:
        """
        Organize scores by round.
        Note: This is approximate since we don't explicitly mark round transitions.
        """
        rounds = {}
        questions_per_round = len(self.scores) // self.max_rounds or 1
        
        for i, score_record in enumerate(self.scores):
            round_num = min((i // questions_per_round) + 1, self.max_rounds)
            
            if round_num not in rounds:
                rounds[round_num] = {
                    "focus": self.round_focuses.get(round_num, f"Round {round_num}"),
                    "scores": []
                }
            
            rounds[round_num]["scores"].append({
                "question": score_record.question,
                "score": score_record.score,
                "feedback": score_record.feedback
            })
        
        return rounds
    
    def get_full_transcript(self) -> List[Dict]:
        """Get the complete conversation transcript"""
        return [
            {
                "role": entry.role,
                "content": entry.content,
                "timestamp": entry.timestamp
            }
            for entry in self.transcript
        ]
    
    def get_duration_minutes(self) -> float:
        """Get interview duration in minutes"""
        duration_seconds = time.time() - self.start_time
        return round(duration_seconds / 60, 1)
    
    def get_statistics(self) -> Dict:
        """Get summary statistics for the interview"""
        if not self.scores:
            return {
                "questions_asked": len(self.question_history),
                "questions_scored": 0,
                "average_score": None,
                "duration_minutes": self.get_duration_minutes()
            }
        
        return {
            "questions_asked": len(self.question_history),
            "questions_scored": len(self.scores),
            "average_score": self.calculate_overall_score(),
            "highest_score": max(s.score for s in self.scores),
            "lowest_score": min(s.score for s in self.scores),
            "duration_minutes": self.get_duration_minutes(),
            "rounds_completed": self.round
        }