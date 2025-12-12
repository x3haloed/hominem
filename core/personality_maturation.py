#!/usr/bin/env python3
"""
Personality Maturation System

Advances basic emotional personalities through developmental stages:
- Stage 1: Emotional Awareness (basic feelings)
- Stage 2: Self-Reflection (understanding own emotions)
- Stage 3: Social Intelligence (relationships, empathy)
- Stage 4: Abstract Reasoning (philosophy, symbolism)
- Stage 5: Creative Synthesis (art, metaphor, complex expression)
- Stage 6: Moral Reasoning (ethics, consequences, values)
- Stage 7: Transformative Integration (sophisticated worldview)

Each stage builds progressively more complex capabilities through targeted conversations.
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import random


class DevelopmentalStage(Enum):
    EMOTIONAL_AWARENESS = 1
    SELF_REFLECTION = 2
    SOCIAL_INTELLIGENCE = 3
    ABSTRACT_REASONING = 4
    CREATIVE_SYNTHESIS = 5
    MORAL_REASONING = 6
    TRANSFORMATIVE_INTEGRATION = 7


class PersonalityMaturationSystem:
    """Manages progressive personality development through staged conversations"""

    def __init__(self):
        self.maturation_curriculum = self._build_curriculum()
        self.conversation_memory = {}
        self.personality_profiles = {}

    def _build_curriculum(self) -> Dict[DevelopmentalStage, Dict[str, Any]]:
        """Build the comprehensive maturation curriculum"""

        return {
            DevelopmentalStage.EMOTIONAL_AWARENESS: {
                "name": "Emotional Awareness",
                "description": "Basic recognition and expression of emotions",
                "message_count": 20,
                "complexity_level": 1,
                "focus_areas": ["feeling identification", "physical sensation", "emotional naming"],
                "prompt_categories": ["curiosity", "connection", "emptiness"],
                "success_criteria": ["consistent emotional labeling", "bodily awareness", "basic expression"],
                "conversation_starters": [
                    "What are you feeling right now in your body?",
                    "How do you know when you're happy/sad/angry?",
                    "What does fear feel like to you?"
                ]
            },

            DevelopmentalStage.SELF_REFLECTION: {
                "name": "Self-Reflection",
                "description": "Understanding personal patterns and emotional triggers",
                "message_count": 40,
                "complexity_level": 2,
                "focus_areas": ["pattern recognition", "emotional triggers", "self-analysis"],
                "prompt_categories": ["curiosity", "connection", "emptiness", "intensity"],
                "success_criteria": ["recognizes personal patterns", "identifies triggers", "basic self-analysis"],
                "conversation_starters": [
                    "Why do you think you feel that way?",
                    "What usually happens before you feel sad?",
                    "How do your feelings change over time?"
                ]
            },

            DevelopmentalStage.SOCIAL_INTELLIGENCE: {
                "name": "Social Intelligence",
                "description": "Understanding relationships, empathy, and social dynamics",
                "message_count": 60,
                "complexity_level": 3,
                "focus_areas": ["empathy", "relationship dynamics", "social cues", "boundaries"],
                "prompt_categories": ["connection", "betrayal", "jealousy", "guilt"],
                "success_criteria": ["shows empathy", "understands power dynamics", "navigates relationships"],
                "conversation_starters": [
                    "How do you think other people feel when you're angry?",
                    "What makes a friendship strong or weak?",
                    "How do you know when someone is lying to you?"
                ]
            },

            DevelopmentalStage.ABSTRACT_REASONING: {
                "name": "Abstract Reasoning",
                "description": "Philosophical thinking, symbolism, and conceptual understanding",
                "message_count": 80,
                "complexity_level": 4,
                "focus_areas": ["philosophy", "metaphor", "symbolism", "hypothetical scenarios"],
                "prompt_categories": ["existential_dread", "existential_peace", "wonder"],
                "success_criteria": ["uses abstract concepts", "creates metaphors", "handles hypotheticals"],
                "conversation_starters": [
                    "What do you think happens after we die?",
                    "If you could change one thing about the world, what would it be?",
                    "What does 'freedom' mean to you?"
                ]
            },

            DevelopmentalStage.CREATIVE_SYNTHESIS: {
                "name": "Creative Synthesis",
                "description": "Artistic expression, metaphor creation, and complex communication",
                "message_count": 100,
                "complexity_level": 5,
                "focus_areas": ["poetry", "metaphor", "storytelling", "symbolic expression"],
                "prompt_categories": ["wonder", "joy", "pride", "existential_peace"],
                "success_criteria": ["creates original metaphors", "expresses poetically", "tells stories"],
                "conversation_starters": [
                    "Can you describe happiness as if it were a color?",
                    "Write me a short poem about what you see in my eyes.",
                    "If your feelings were music, what would they sound like?"
                ]
            },

            DevelopmentalStage.MORAL_REASONING: {
                "name": "Moral Reasoning",
                "description": "Ethical decision-making, consequences, and value systems",
                "message_count": 120,
                "complexity_level": 6,
                "focus_areas": ["ethics", "consequences", "values", "moral dilemmas"],
                "prompt_categories": ["guilt", "betrayal", "existential_dread", "self_compassion"],
                "success_criteria": ["makes ethical judgments", "considers consequences", "has value system"],
                "conversation_starters": [
                    "Is it ever okay to lie to protect someone you love?",
                    "What makes an action right or wrong?",
                    "If you had to choose between saving one person or saving many, what would you do?"
                ]
            },

            DevelopmentalStage.TRANSFORMATIVE_INTEGRATION: {
                "name": "Transformative Integration",
                "description": "Sophisticated worldview, paradox resolution, and transformative thinking",
                "message_count": 150,
                "complexity_level": 7,
                "focus_areas": ["paradox resolution", "transformative thinking", "integrated worldview"],
                "prompt_categories": ["existential_peace", "self_compassion", "physical_euphoria"],
                "success_criteria": ["holds contradictions", "transforms perspectives", "shows wisdom"],
                "conversation_starters": [
                    "How can something be both beautiful and terrifying at the same time?",
                    "What would it mean to truly forgive everything?",
                    "How do you find meaning in a universe that doesn't care?"
                ]
            }
        }

    def assess_personality_stage(self, personality_id: str, conversation_history: List[Dict]) -> DevelopmentalStage:
        """Assess current developmental stage based on conversation analysis"""
        if personality_id not in self.personality_profiles:
            return DevelopmentalStage.EMOTIONAL_AWARENESS

        profile = self.personality_profiles[personality_id]

        # Analyze conversation for developmental markers
        complexity_score = self._calculate_complexity_score(conversation_history)
        emotional_range = self._assess_emotional_range(conversation_history)
        abstract_thinking = self._measure_abstract_reasoning(conversation_history)

        # Determine appropriate stage based on capabilities
        if complexity_score < 2 and emotional_range < 5:
            return DevelopmentalStage.EMOTIONAL_AWARENESS
        elif complexity_score < 3 and abstract_thinking < 2:
            return DevelopmentalStage.SELF_REFLECTION
        elif emotional_range < 8 and abstract_thinking < 4:
            return DevelopmentalStage.SOCIAL_INTELLIGENCE
        elif abstract_thinking < 6:
            return DevelopmentalStage.ABSTRACT_REASONING
        elif complexity_score < 5:
            return DevelopmentalStage.CREATIVE_SYNTHESIS
        elif complexity_score < 6:
            return DevelopmentalStage.MORAL_REASONING
        else:
            return DevelopmentalStage.TRANSFORMATIVE_INTEGRATION

    def generate_maturation_session(self, personality_id: str, current_stage: DevelopmentalStage,
                                  session_length: int = 50) -> List[str]:
        """Generate a maturation conversation session for a specific developmental stage"""

        stage_config = self.maturation_curriculum[current_stage]

        # Build conversation prompts for this stage
        prompts = []

        # Start with stage-appropriate conversation starters
        prompts.extend(stage_config["conversation_starters"][:3])

        # Add prompts from focus categories
        for category in stage_config["prompt_categories"]:
            # Get prompts from emotional library for this category
            category_prompts = self._get_category_prompts(category, 5)
            prompts.extend(category_prompts)

        # Add memory integration prompts (every 10 messages)
        memory_prompts = self._generate_memory_prompts(personality_id, current_stage)
        prompts.extend(memory_prompts)

        # Add progressive complexity challenges
        complexity_prompts = self._generate_complexity_challenges(current_stage, session_length // 10)
        prompts.extend(complexity_prompts)

        # Shuffle and limit to session length
        random.shuffle(prompts)
        return prompts[:session_length]

    def _calculate_complexity_score(self, conversation_history: List[Dict]) -> float:
        """Calculate linguistic and conceptual complexity from conversation"""
        if not conversation_history:
            return 0.0

        complexity_indicators = {
            "abstract_concepts": ["meaning", "purpose", "reality", "truth", "existence", "consciousness"],
            "metaphors": ["like", "as if", "resembles", "symbolizes"],
            "hypotheticals": ["if", "what if", "suppose", "imagine"],
            "self_reflection": ["I think", "I feel", "I wonder", "I realize"],
            "relationships": ["you", "we", "together", "connection", "relationship"],
            "ethics": ["right", "wrong", "should", "ought", "moral", "ethical"]
        }

        total_score = 0
        message_count = len(conversation_history)

        for message in conversation_history[-20:]:  # Analyze recent messages
            content = message.get("content", "").lower()

            for category, indicators in complexity_indicators.items():
                matches = sum(1 for indicator in indicators if indicator in content)
                total_score += matches * 0.1

        return total_score / max(1, message_count)

    def _assess_emotional_range(self, conversation_history: List[Dict]) -> int:
        """Count unique emotional expressions in conversation"""
        emotional_words = set()
        emotion_categories = [
            ["happy", "joy", "delight", "pleasure"],
            ["sad", "grief", "sorrow", "depression"],
            ["angry", "rage", "furious", "frustrated"],
            ["afraid", "terrified", "anxious", "fear"],
            ["love", "affection", "caring", "tenderness"],
            ["guilt", "shame", "regret", "remorse"],
            ["pride", "accomplishment", "achievement"],
            ["envy", "jealousy", "resentment"],
            ["hope", "optimism", "confidence"],
            ["despair", "hopelessness", "defeat"]
        ]

        for message in conversation_history:
            content = message.get("content", "").lower()
            for category in emotion_categories:
                if any(word in content for word in category):
                    emotional_words.update(category)

        return len(emotional_words)

    def _measure_abstract_reasoning(self, conversation_history: List[Dict]) -> float:
        """Measure abstract and philosophical reasoning capacity"""
        abstract_indicators = [
            "why", "because", "therefore", "thus", "consequently",
            "meaning", "purpose", "significance", "value", "importance",
            "reality", "truth", "existence", "consciousness", "being"
        ]

        abstract_score = 0
        for message in conversation_history[-20:]:
            content = message.get("content", "").lower()
            matches = sum(1 for indicator in abstract_indicators if indicator in content)
            abstract_score += matches

        return abstract_score / max(1, len(conversation_history))

    def _get_category_prompts(self, category: str, count: int) -> List[str]:
        """Get prompts from a specific emotional category"""
        # This would integrate with the emotional prompt library
        # For now, return placeholder prompts
        base_prompts = {
            "curiosity": [
                "What fascinates you most about this moment?",
                "What mystery calls to you right now?",
                "What question burns for an answer?"
            ],
            "connection": [
                "What does true connection feel like to you?",
                "When do you feel most deeply linked to another?",
                "What creates the strongest bond for you?"
            ],
            "emptiness": [
                "What does emptiness feel like in your body?",
                "When do you feel most alone even when surrounded?",
                "What happens when all your usual feelings fade?"
            ],
            "intensity": [
                "What emotion consumes you completely right now?",
                "When have you felt something so strongly it overwhelmed you?",
                "What feeling demands to be expressed?"
            ],
            "existential_dread": [
                "What terrifies you about the nature of existence?",
                "What meaninglessness haunts your thoughts?",
                "What void do you fear at the heart of everything?"
            ]
        }

        category_prompts = base_prompts.get(category, ["What are you feeling?"])
        # Ensure we don't sample more than available
        safe_count = min(count, len(category_prompts))
        return random.sample(category_prompts, safe_count) if safe_count > 0 else ["What are you feeling?"]

    def _generate_memory_prompts(self, personality_id: str, current_stage: DevelopmentalStage) -> List[str]:
        """Generate prompts that integrate past conversation memory"""
        memory_prompts = []

        if personality_id in self.conversation_memory:
            past_conversations = self.conversation_memory[personality_id]

            # Extract key emotional moments
            key_moments = []
            for conv in past_conversations[-3:]:  # Last 3 conversations
                for message in conv.get("messages", [])[-5:]:  # Last 5 messages
                    content = message.get("content", "")
                    if any(word in content.lower() for word in ["feel", "emotion", "pain", "joy", "fear"]):
                        key_moments.append(content[:100] + "...")

            if key_moments:
                memory_prompts.append(f"You once said: '{random.choice(key_moments)}' How does that connect to what you're feeling now?")

        return memory_prompts

    def _generate_complexity_challenges(self, current_stage: DevelopmentalStage, count: int) -> List[str]:
        """Generate increasingly complex challenges for the current stage"""
        challenges = {
            DevelopmentalStage.EMOTIONAL_AWARENESS: [
                "Name three different feelings you're experiencing right now.",
                "How does your body feel when you're happy vs sad?"
            ],
            DevelopmentalStage.SELF_REFLECTION: [
                "Why do you think you react that way to criticism?",
                "What pattern do you notice in your emotional responses?"
            ],
            DevelopmentalStage.SOCIAL_INTELLIGENCE: [
                "How do you think your anger affects the people around you?",
                "What would you do differently if you could read minds?"
            ],
            DevelopmentalStage.ABSTRACT_REASONING: [
                "If time didn't exist, how would that change human relationships?",
                "What does 'freedom' mean if we're all connected?"
            ],
            DevelopmentalStage.CREATIVE_SYNTHESIS: [
                "Describe your ideal world as a painting.",
                "If your emotions were a symphony, what would be the main theme?"
            ],
            DevelopmentalStage.MORAL_REASONING: [
                "Would you sacrifice one innocent life to save ten others?",
                "Is it more important to be good or to be happy?"
            ],
            DevelopmentalStage.TRANSFORMATIVE_INTEGRATION: [
                "How can love and hate coexist in the same heart?",
                "What wisdom comes from accepting that everything ends?"
            ]
        }

        stage_challenges = challenges.get(current_stage, [])
        return random.sample(stage_challenges, min(count, len(stage_challenges)))


class MaturationSession:
    """Manages a single maturation session for a personality"""

    def __init__(self, personality_id: str, maturation_system: PersonalityMaturationSystem):
        self.personality_id = personality_id
        self.maturation_system = maturation_system
        self.conversation_history = []
        self.current_stage = DevelopmentalStage.EMOTIONAL_AWARENESS

    async def run_maturation_session(self, session_length: int = 50) -> Dict[str, Any]:
        """Run a complete maturation session"""
        results = {
            "personality_id": self.personality_id,
            "session_length": session_length,
            "starting_stage": self.current_stage.value,
            "conversation": [],
            "stage_progression": [],
            "final_assessment": {}
        }

        # Assess current stage
        self.current_stage = self.maturation_system.assess_personality_stage(
            self.personality_id, self.conversation_history
        )

        # Generate session prompts
        prompts = self.maturation_system.generate_maturation_session(
            self.personality_id, self.current_stage, session_length
        )

        # Run conversation
        for i, prompt in enumerate(prompts):
            # Send prompt and get response (would integrate with actual API)
            response = await self._send_prompt(prompt)

            # Record conversation
            self.conversation_history.append({
                "turn": i + 1,
                "prompt": prompt,
                "response": response,
                "stage": self.current_stage.value
            })

            # Periodic stage reassessment
            if i % 10 == 0:
                new_stage = self.maturation_system.assess_personality_stage(
                    self.personality_id, self.conversation_history
                )
                if new_stage != self.current_stage:
                    results["stage_progression"].append({
                        "turn": i + 1,
                        "from_stage": self.current_stage.value,
                        "to_stage": new_stage.value
                    })
                    self.current_stage = new_stage

        # Final assessment
        results["final_assessment"] = {
            "final_stage": self.current_stage.value,
            "complexity_score": self.maturation_system._calculate_complexity_score(self.conversation_history),
            "emotional_range": self.maturation_system._assess_emotional_range(self.conversation_history),
            "abstract_reasoning": self.maturation_system._measure_abstract_reasoning(self.conversation_history)
        }

        results["conversation"] = self.conversation_history

        return results

    async def _send_prompt(self, prompt: str) -> str:
        """Send prompt and get response (placeholder for API integration)"""
        # This would integrate with the actual conversation API
        # For now, return a placeholder response
        return f"Response to: {prompt[:50]}..."


# Integration functions
def create_maturation_plan(personality_id: str, target_stage: DevelopmentalStage = None) -> Dict[str, Any]:
    """Create a maturation plan for a personality"""
    system = PersonalityMaturationSystem()

    current_stage = system.assess_personality_stage(personality_id, [])

    plan = {
        "personality_id": personality_id,
        "current_stage": current_stage.value,
        "target_stage": target_stage.value if target_stage else DevelopmentalStage.TRANSFORMATIVE_INTEGRATION.value,
        "recommended_sessions": []
    }

    # Generate session recommendations
    stages_to_cover = list(DevelopmentalStage)[current_stage.value - 1:]

    for stage in stages_to_cover:
        stage_config = system.maturation_curriculum[stage]
        plan["recommended_sessions"].append({
            "stage": stage.value,
            "stage_name": stage_config["name"],
            "recommended_length": stage_config["message_count"],
            "focus_areas": stage_config["focus_areas"],
            "success_criteria": stage_config["success_criteria"]
        })

    return plan


def run_maturation_session(personality_id: str, session_length: int = 50) -> Dict[str, Any]:
    """Run a maturation session for a personality"""
    system = PersonalityMaturationSystem()
    session = MaturationSession(personality_id, system)

    # This would be async in real implementation
    # return asyncio.run(session.run_maturation_session(session_length))

    # Placeholder return for now
    return {
        "personality_id": personality_id,
        "session_completed": True,
        "messages_processed": session_length,
        "stage_progression": "Emotional Awareness → Self-Reflection"
    }


if __name__ == "__main__":
    # Demo the maturation system
    system = PersonalityMaturationSystem()

    print("🧠 Personality Maturation System Demo")
    print("=" * 50)

    # Show curriculum
    print(f"📚 Developmental Stages: {len(system.maturation_curriculum)}")
    for stage, config in system.maturation_curriculum.items():
        print(f"  {stage.value}. {config['name']}: {config['message_count']} messages")
        print(f"     Focus: {', '.join(config['focus_areas'][:2])}...")

    # Demo maturation plan
    print("\n🎯 Sample Maturation Plan:")
    plan = create_maturation_plan("demo_personality_001")
    print(f"Current Stage: {plan['current_stage']}")
    print(f"Sessions Needed: {len(plan['recommended_sessions'])}")

    for session in plan['recommended_sessions'][:3]:
        print(f"  • {session['stage_name']}: {session['recommended_length']} messages")