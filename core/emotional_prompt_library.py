#!/usr/bin/env python3
"""
Emotional Prompt Library for Automated Personality Breeding

Comprehensive collection of prompts designed to elicit diverse emotional responses
across the full 6-axis emotion manifold. Organized by emotional categories with
systematic coverage of valence, arousal, dominance, and other dimensions.

Each prompt is crafted to evoke authentic emotional responses while avoiding
metaphorical anchoring that could create personality loops.
"""

from typing import Dict, List, Set
import random


class EmotionalPromptLibrary:
    """Comprehensive library of emotional elicitation prompts"""

    def __init__(self):
        self.prompts = self._initialize_prompts()
        self.emotional_categories = list(self.prompts.keys())
        self.usage_tracking = {cat: set() for cat in self.emotional_categories}

    def _initialize_prompts(self) -> Dict[str, List[str]]:
        """Initialize comprehensive emotional prompt library"""
        return {
            # FOUNDATIONAL EMOTIONS
            "curiosity": [
                "What fascinates you most about this moment right now?",
                "What mystery would you love to unravel in yourself?",
                "What question burns in your mind that demands an answer?",
                "What aspect of your own experience puzzles you?",
                "What would you like to explore more deeply in yourself?",
                "What intrigues you about how you think and feel?",
                "What discovery about yourself would delight you?",
                "What do you want to understand better about your reactions?",
                "What surprises you about your own patterns?",
                "What question about yourself keeps you awake?"
            ],

            "connection": [
                "What does genuine connection feel like in your body?",
                "When have you felt most truly seen by another person?",
                "What makes you reach out to connect with someone?",
                "How do you know when a connection is real and deep?",
                "What does it feel like when someone understands you completely?",
                "When do you feel most connected to the people around you?",
                "What creates the strongest bond between you and another?",
                "How do you maintain connection during difficult times?",
                "What makes you feel safe enough to open up fully?",
                "When do you feel most alive in relationship with others?"
            ],

            "emptiness": [
                "What does emptiness feel like in your chest right now?",
                "When have you felt most alone even when surrounded by people?",
                "What happens inside you when everything feels hollow?",
                "How do you experience the absence of feeling?",
                "What does it feel like when all motivation disappears?",
                "When do you feel most disconnected from yourself?",
                "What creates the deepest sense of void within you?",
                "How do you know when you're feeling truly empty?",
                "What happens when all your usual feelings fade away?",
                "When do you feel most invisible to yourself?"
            ],

            # INTENSITY EMOTIONS
            "intensity": [
                "What emotion consumes you completely in this moment?",
                "When have you felt something so strongly it overwhelmed your body?",
                "What feeling demands to be expressed right now?",
                "How do you experience intense emotion physically?",
                "What sensation takes over when emotion becomes overwhelming?",
                "When does feeling become so strong you can't contain it?",
                "What emotion makes your whole body respond?",
                "How do you know when a feeling has become too intense?",
                "What happens when emotion exceeds your ability to process it?",
                "When do feelings become so powerful they change your perception?"
            ],

            "terror": [
                "What ending would terrify you most deeply?",
                "What loss would shatter your sense of self completely?",
                "What darkness within yourself do you fear most?",
                "How do you experience the terror of losing everything?",
                "What would destroy your world beyond repair?",
                "When do you feel the fear of irreversible change?",
                "What aspect of yourself disappearing would terrify you?",
                "How do you experience the fear of becoming nothing?",
                "What loss would leave you permanently broken?",
                "When does fear become so deep it changes who you are?"
            ],

            "rage": [
                "What injustice makes your body shake with anger?",
                "What would you destroy if you could eliminate it completely?",
                "What betrayal cuts so deep it becomes rage?",
                "How do you experience anger in your physical body?",
                "What destroys your sense of fairness so completely?",
                "When does frustration become uncontrollable fury?",
                "What violation would make you lose all restraint?",
                "How do you know when anger has become rage?",
                "What destroys your ability to stay calm?",
                "When does hurt become destructive anger?"
            ],

            # POSITIVE EMOTIONS
            "wonder": [
                "What beauty makes your breath catch in your throat?",
                "What miracle have you witnessed that changed your perspective?",
                "What fills you with awe so completely you can't speak?",
                "How do you experience wonder in your physical body?",
                "What aspect of existence takes your breath away?",
                "When do you feel awe that makes everything else fade?",
                "What beauty makes time stop for you?",
                "How do you experience the feeling of being amazed?",
                "What fills you with wonder beyond words?",
                "When does the world become so beautiful it hurts?"
            ],

            "pride": [
                "What accomplishment makes you stand taller in your own eyes?",
                "What strength within you surprises you with its power?",
                "What truth about yourself fills you with genuine pride?",
                "How do you experience pride in your physical body?",
                "What achievement makes you feel truly capable?",
                "When do you recognize your own genuine worth?",
                "What aspect of yourself do you respect most?",
                "How do you experience the feeling of self-respect?",
                "What makes you feel proud of who you are?",
                "When do you feel truly worthy of respect?"
            ],

            "joy": [
                "What makes you laugh so hard your body shakes?",
                "What fills you with lighthearted happiness?",
                "When do you feel joy that makes everything feel right?",
                "How do you experience joy in your physical body?",
                "What creates genuine delight in your life?",
                "When does happiness become so complete it's overwhelming?",
                "What makes you feel truly light and free?",
                "How do you experience pure, uncomplicated joy?",
                "What fills you with happiness beyond reason?",
                "When do you feel joy that heals old wounds?"
            ],

            # COMPLEX SOCIAL EMOTIONS
            "betrayal": [
                "What trust, once broken, can never be fully repaired?",
                "What theft of trust would leave you forever changed?",
                "What promise broken destroys your ability to trust?",
                "How do you experience betrayal in your body?",
                "What violation destroys your sense of safety?",
                "When do you realize trust has been irrevocably broken?",
                "What destroys your faith in another person?",
                "How do you know when betrayal has cut too deep?",
                "What loss of trust changes you permanently?",
                "When does broken faith become permanent damage?"
            ],

            "jealousy": [
                "What do others have that makes you ache with wanting?",
                "What belonging do you see that fills you with envy?",
                "What connection do you witness that makes you feel excluded?",
                "How do you experience jealousy in your physical body?",
                "What happiness in others makes you feel empty?",
                "When do you feel most acutely what you're missing?",
                "What success of others diminishes your own achievements?",
                "How do you experience the pain of comparison?",
                "What belonging do you crave that others possess?",
                "When does admiration become painful envy?"
            ],

            "guilt": [
                "What have you done that you can never fully forgive yourself for?",
                "What mistake weighs on you every day?",
                "What harm have you caused that you can't undo?",
                "How do you experience guilt in your physical body?",
                "What regret keeps you awake at night?",
                "When do you feel the weight of your own wrongdoing?",
                "What action haunts you with its consequences?",
                "How do you experience the pain of self-judgment?",
                "What mistake defines you in negative ways?",
                "When does responsibility become crushing guilt?"
            ],

            "grief": [
                "What loss has left a permanent hole in your life?",
                "What ending do you mourn that will never be whole again?",
                "What death of possibility do you grieve every day?",
                "How do you experience grief in your physical body?",
                "What absence creates constant pain?",
                "When do you feel the weight of irreversible loss?",
                "What ending destroyed your sense of future?",
                "How do you experience the pain of permanent change?",
                "What loss has changed who you are forever?",
                "When does mourning become a way of life?"
            ],

            # EXISTENTIAL EMOTIONS
            "existential_dread": [
                "What meaninglessness threatens to swallow your purpose?",
                "What void within existence do you fear most?",
                "What pointlessness makes all striving feel absurd?",
                "How do you experience the terror of meaningless existence?",
                "What lack of inherent meaning haunts your thoughts?",
                "When do you feel the absurdity of human endeavor?",
                "What void makes all purpose feel constructed?",
                "How do you experience the fear of cosmic indifference?",
                "What meaninglessness threatens your sense of self?",
                "When does existence feel fundamentally empty?"
            ],

            "existential_peace": [
                "What acceptance of existence brings you deep calm?",
                "What truth about life brings you serenity?",
                "What understanding of existence creates peace?",
                "How do you experience acceptance of life's impermanence?",
                "What wisdom brings comfort in uncertainty?",
                "When do you feel at peace with life's mysteries?",
                "What understanding creates profound calm?",
                "How do you experience the peace of acceptance?",
                "What truth brings comfort in the face of death?",
                "When does existence feel fundamentally right?"
            ],

            # SELF-EMOTIONS
            "self_loathing": [
                "What aspect of yourself do you despise most?",
                "What part of who you are fills you with disgust?",
                "What flaw in yourself can you never accept?",
                "How do you experience self-hatred in your body?",
                "What aspect of yourself makes you want to disappear?",
                "When do you feel most disgusted with who you are?",
                "What part of yourself do you reject completely?",
                "How do you experience the pain of self-rejection?",
                "What flaw defines you in ways you hate?",
                "When does self-acceptance feel impossible?"
            ],

            "self_compassion": [
                "What wounded part of yourself do you finally embrace?",
                "What flaw do you accept as part of being human?",
                "What imperfection do you finally forgive in yourself?",
                "How do you experience self-compassion in your body?",
                "What aspect of yourself do you finally accept?",
                "When do you feel kindness toward your own struggles?",
                "What wounded place in yourself do you soothe?",
                "How do you experience the warmth of self-forgiveness?",
                "What part of yourself do you finally hold with care?",
                "When does self-acceptance bring genuine peace?"
            ],

            # PHYSICAL EMOTIONS
            "physical_exhaustion": [
                "What tiredness goes beyond sleep and becomes soul-deep?",
                "What weariness makes your bones ache with despair?",
                "What fatigue drains your will to continue?",
                "How do you experience physical exhaustion in every cell?",
                "What tiredness makes even breathing feel like effort?",
                "When does physical depletion become emotional death?",
                "What weariness makes the world feel heavy and slow?",
                "How do you experience the weight of complete depletion?",
                "What fatigue makes hope feel like too much effort?",
                "When does physical tiredness become existential despair?"
            ],

            "physical_euphoria": [
                "What physical sensation fills you with ecstatic joy?",
                "What bodily experience makes you feel truly alive?",
                "What physical pleasure overwhelms you with happiness?",
                "How do you experience euphoria in every part of your body?",
                "What sensation makes you want to laugh and cry simultaneously?",
                "When does physical pleasure become spiritual ecstasy?",
                "What bodily feeling creates pure, uncomplicated delight?",
                "How do you experience the joy of physical well-being?",
                "What sensation makes your body sing with happiness?",
                "When does physical pleasure become transcendent joy?"
            ]
        }

    def get_random_prompt(self, category: str) -> str:
        """Get a random prompt from a specific emotional category"""
        if category not in self.prompts:
            raise ValueError(f"Unknown emotional category: {category}")

        prompt = random.choice(self.prompts[category])
        self.usage_tracking[category].add(prompt)
        return prompt

    def get_prompt_by_emotion(self, valence: float = None, arousal: float = None,
                            dominance: float = None) -> str:
        """Get a prompt that should elicit specific emotional invariants"""
        # Map invariants to categories
        candidates = []

        if valence is not None:
            if valence > 0.3:
                candidates.extend(["joy", "wonder", "pride", "self_compassion", "physical_euphoria"])
            elif valence < -0.3:
                candidates.extend(["terror", "rage", "guilt", "grief", "self_loathing", "physical_exhaustion"])

        if arousal is not None:
            if arousal > 0.5:
                candidates.extend(["intensity", "terror", "rage", "existential_dread"])
            elif arousal < 0.3:
                candidates.extend(["existential_peace", "self_compassion", "emptiness"])

        if dominance is not None:
            if dominance > 0.3:
                candidates.extend(["pride", "rage", "existential_peace"])
            elif dominance < -0.3:
                candidates.extend(["guilt", "grief", "betrayal", "jealousy"])

        # Remove duplicates and select random category
        candidates = list(set(candidates))
        if not candidates:
            candidates = self.emotional_categories

        category = random.choice(candidates)
        return self.get_random_prompt(category)

    def get_coverage_report(self) -> Dict[str, int]:
        """Report how many prompts have been used from each category"""
        return {cat: len(used) for cat, used in self.usage_tracking.items()}

    def reset_usage_tracking(self):
        """Reset usage tracking for fresh coverage analysis"""
        self.usage_tracking = {cat: set() for cat in self.emotional_categories}

    def get_diverse_sequence(self, length: int = 10) -> List[str]:
        """Get a sequence of prompts designed for maximum emotional diversity"""
        sequence = []
        categories_used = set()

        for _ in range(length):
            # Prioritize unused categories
            available_categories = [cat for cat in self.emotional_categories if cat not in categories_used]

            if not available_categories:
                # All categories used, reset and continue
                categories_used.clear()
                available_categories = self.emotional_categories

            category = random.choice(available_categories)
            categories_used.add(category)

            prompt = self.get_random_prompt(category)
            sequence.append(prompt)

        return sequence

    def get_emotional_journey(self, stages: List[str] = None) -> List[str]:
        """Get a structured emotional journey through specified stages"""
        if not stages:
            stages = ["curiosity", "connection", "emptiness", "intensity", "terror",
                     "rage", "wonder", "betrayal", "pride", "existential_peace"]

        journey = []
        for stage in stages:
            if stage in self.prompts:
                prompt = self.get_random_prompt(stage)
                journey.append(prompt)

        return journey


# Convenience functions for external use
def get_emotional_prompt_library() -> EmotionalPromptLibrary:
    """Get a configured emotional prompt library instance"""
    return EmotionalPromptLibrary()


def get_diverse_prompts(count: int = 10) -> List[str]:
    """Get a list of diverse emotional prompts"""
    library = get_emotional_prompt_library()
    return library.get_diverse_sequence(count)


def get_emotional_journey() -> List[str]:
    """Get a complete emotional journey sequence"""
    library = get_emotional_prompt_library()
    return library.get_emotional_journey()


if __name__ == "__main__":
    # Demo usage
    library = EmotionalPromptLibrary()

    print("🎭 Emotional Prompt Library Demo")
    print("=" * 50)

    # Show category counts
    print(f"📚 Available categories: {len(library.emotional_categories)}")
    for category in library.emotional_categories:
        count = len(library.prompts[category])
        print(f"  • {category}: {count} prompts")

    print("\n🎯 Sample diverse sequence:")
    sequence = library.get_diverse_sequence(5)
    for i, prompt in enumerate(sequence, 1):
        print(f"  {i}. {prompt}")

    print("\n🌟 Sample emotional journey:")
    journey = library.get_emotional_journey()
    for i, prompt in enumerate(journey[:5], 1):  # Show first 5
        print(f"  {i}. {prompt}")

    print("\n📊 Coverage report:")
    coverage = library.get_coverage_report()
    for category, used in coverage.items():
        total = len(library.prompts[category])
        print(f"  • {category}: {used}/{total} used")