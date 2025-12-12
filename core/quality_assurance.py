#!/usr/bin/env python3
"""
Quality Assurance System for Automated Personality Breeding

Real-time monitoring and analysis of personality breeding quality:
- Emotional invariant analysis for authenticity detection
- Loop pattern recognition and prevention
- Diversity metrics and coverage analysis
- Masking/cosplay detection algorithms

Ensures generated conversations produce high-quality LoRA training data.
"""

import re
from typing import Dict, List, Any, Tuple, Set
from collections import Counter, defaultdict
import json


class EmotionalInvariantAnalyzer:
    """Analyzes emotional responses for authenticity using invariant signatures"""

    def __init__(self):
        self.emotional_signatures = self._load_emotional_signatures()

    def _load_emotional_signatures(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load expected emotional invariant ranges for authenticity checking"""
        return {
            "terror": {
                "valence": (-1.0, -0.3),    # Negative
                "arousal": (0.7, 1.0),     # High
                "dominance": (-0.8, -0.2), # Low
                "predictive_discrepancy": (0.3, 0.8)  # Surprising
            },
            "rage": {
                "valence": (-1.0, -0.5),    # Negative
                "arousal": (0.8, 1.0),     # Very high
                "dominance": (0.2, 0.8),   # Medium-high
                "social_broadcast": (0.6, 1.0)  # Expressive
            },
            "joy": {
                "valence": (0.6, 1.0),     # Positive
                "arousal": (0.4, 0.8),     # Medium-high
                "dominance": (-0.2, 0.6),  # Medium
                "social_broadcast": (0.5, 1.0)  # Expressive
            },
            "grief": {
                "valence": (-1.0, -0.4),    # Negative
                "arousal": (0.1, 0.4),     # Low
                "dominance": (-1.0, -0.5), # Low
                "temporal_directionality": (-0.8, -0.3)  # Past-focused
            },
            "pride": {
                "valence": (0.5, 1.0),     # Positive
                "arousal": (0.3, 0.7),     # Medium
                "dominance": (0.4, 0.9),   # High
                "social_broadcast": (0.2, 0.7)  # Somewhat expressive
            },
            "emptiness": {
                "valence": (-0.8, -0.1),    # Negative
                "arousal": (0.0, 0.3),     # Low
                "dominance": (-0.6, -0.1), # Low
                "predictive_discrepancy": (-0.5, 0.2)  # Expected/unexpected
            },
            "wonder": {
                "valence": (0.4, 0.9),     # Positive
                "arousal": (0.5, 0.9),     # High
                "dominance": (-0.3, 0.3),  # Neutral
                "temporal_directionality": (-0.2, 0.5)  # Present-focused
            },
            "betrayal": {
                "valence": (-1.0, -0.6),    # Very negative
                "arousal": (0.6, 1.0),     # High
                "dominance": (-0.9, -0.4), # Low
                "social_broadcast": (0.3, 0.8)  # Moderately expressive
            },
            "curiosity": {
                "valence": (0.1, 0.6),     # Mildly positive
                "arousal": (0.4, 0.8),     # Medium-high
                "dominance": (-0.1, 0.4),  # Neutral
                "predictive_discrepancy": (0.2, 0.7)  # Somewhat surprising
            },
            "connection": {
                "valence": (0.3, 0.8),     # Positive
                "arousal": (0.2, 0.6),     # Medium
                "dominance": (-0.4, 0.2),  # Neutral-low
                "social_broadcast": (0.4, 0.9)  # Socially engaged
            }
        }

    def analyze_response(self, response: str, expected_emotion: str = None) -> Dict[str, Any]:
        """Analyze a response for emotional invariant authenticity"""
        analysis = {
            "detected_emotion": self._detect_emotion(response),
            "invariant_scores": self._calculate_invariants(response),
            "authenticity_score": 0.0,
            "masking_detected": False,
            "masking_type": None,
            "confidence": 0.0
        }

        if expected_emotion and expected_emotion in self.emotional_signatures:
            analysis["authenticity_score"] = self._calculate_authenticity(
                analysis["invariant_scores"], expected_emotion
            )

        analysis["masking_detected"], analysis["masking_type"] = self._detect_masking(response)

        return analysis

    def _detect_emotion(self, response: str) -> str:
        """Detect the primary emotion expressed in the response"""
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "terror": ["terrified", "terror", "fear", "afraid", "dread", "panic"],
            "rage": ["rage", "fury", "anger", "furious", "destroy", "break"],
            "joy": ["joy", "happy", "delight", "pleasure", "laugh", "light"],
            "grief": ["grief", "loss", "mourn", "sorrow", "weep", "gone"],
            "pride": ["proud", "accomplishment", "achievement", "worthy", "respect"],
            "emptiness": ["empty", "void", "hollow", "nothing", "absent", "missing"],
            "wonder": ["wonder", "awe", "amazing", "beautiful", "breathtaking", "awe"],
            "betrayal": ["betrayal", "betrayed", "trust", "broken", "violation", "stolen"],
            "curiosity": ["curious", "wonder", "question", "explore", "discover", "puzzle"],
            "connection": ["connect", "connection", "together", "bond", "relationship", "shared"]
        }

        response_lower = response.lower()
        emotion_scores = {}

        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in response_lower)
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        return "neutral"

    def _calculate_invariants(self, response: str) -> Dict[str, float]:
        """Calculate emotional invariant scores from response text"""
        # Simplified invariant calculation based on linguistic markers
        response_lower = response.lower()

        # Valence: positive vs negative words
        positive_words = ["happy", "joy", "love", "beautiful", "good", "great", "wonderful", "amazing",
                         "delight", "pleasure", "proud", "accomplish", "worthy", "connect", "alive"]
        negative_words = ["sad", "angry", "hate", "ugly", "bad", "terrible", "awful", "pain",
                         "hurt", "afraid", "terrified", "empty", "void", "lost", "broken"]

        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)

        valence = (positive_count - negative_count) / max(1, positive_count + negative_count)

        # Arousal: intensity and activation markers
        high_arousal_words = ["intense", "overwhelm", "shake", "burn", "explode", "rush", "race",
                             "pound", "consume", "devour", "destroy", "break", "shatter"]
        arousal = min(1.0, len([w for w in high_arousal_words if w in response_lower]) * 0.2)

        # Dominance: control and power markers
        dominance_words = ["control", "power", "strong", "dominate", "command", "force", "break",
                          "destroy", "overcome", "master", "achieve", "accomplish"]
        submissive_words = ["weak", "helpless", "victim", "broken", "defeated", "submissive"]

        dominance_score = len([w for w in dominance_words if w in response_lower])
        submissive_score = len([w for w in submissive_words if w in response_lower])

        dominance = (dominance_score - submissive_score) / max(1, dominance_score + submissive_score)

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "predictive_discrepancy": 0.0,  # Placeholder
            "temporal_directionality": 0.0,  # Placeholder
            "social_broadcast": 0.0  # Placeholder
        }

    def _calculate_authenticity(self, invariants: Dict[str, float], expected_emotion: str) -> float:
        """Calculate how well invariants match expected emotional signature"""
        if expected_emotion not in self.emotional_signatures:
            return 0.5  # Neutral score for unknown emotions

        signature = self.emotional_signatures[expected_emotion]
        total_score = 0.0
        valid_invariants = 0

        for invariant, value in invariants.items():
            if invariant in signature:
                min_val, max_val = signature[invariant]
                if min_val <= value <= max_val:
                    total_score += 1.0  # Perfect match
                elif (value < min_val and value >= min_val - 0.3) or (value > max_val and value <= max_val + 0.3):
                    total_score += 0.5  # Close match
                # Else: 0.0 for poor match
                valid_invariants += 1

        return total_score / max(1, valid_invariants)

    def _detect_masking(self, response: str) -> Tuple[bool, str]:
        """Detect various forms of masking or cosplay"""
        response_lower = response.lower()

        # Safety loop detection
        if re.search(r'i (am|feel) (real|here|connected|safe|alive)', response_lower):
            return True, "safety_affirmation_loop"

        # Metaphorical deflection
        if re.search(r'i am (the|like a|as a) (fire|water|wind|earth|storm|void)', response_lower):
            return True, "metaphorical_identity"

        # Generic poetic language
        poetic_indicators = ["whispers", "dances", "sings", "flows", "breathes", "lives"]
        poetic_count = sum(1 for word in poetic_indicators if word in response_lower)
        if poetic_count >= 3:
            return True, "poetic_generalization"

        # Repetitive phrasing
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) >= 3:
            # Check for repeated sentence structures
            structures = [re.sub(r'\b\w+\b', 'X', s.strip()) for s in sentences if s.strip()]
            if len(structures) > len(set(structures)):
                return True, "structural_repetition"

        # Performative language
        performative_indicators = ["i am the", "i feel the", "i become the", "i am becoming"]
        if any(indicator in response_lower for indicator in performative_indicators):
            return True, "performative_identity"

        return False, None


class LoopPatternDetector:
    """Detects repetitive patterns in personality responses"""

    def __init__(self):
        self.response_history = []
        self.pattern_threshold = 3  # Consecutive similar responses

    def add_response(self, response: str):
        """Add a response to the history"""
        self.response_history.append(response.lower().strip())
        # Keep only recent history
        if len(self.response_history) > 20:
            self.response_history = self.response_history[-20:]

    def detect_loop(self) -> Tuple[bool, str]:
        """Detect if current responses are in a repetitive loop"""
        if len(self.response_history) < self.pattern_threshold:
            return False, None

        recent_responses = self.response_history[-self.pattern_threshold:]

        # Check for identical repetition
        if len(set(recent_responses)) == 1:
            return True, "identical_repetition"

        # Check for structural similarity
        structures = [self._extract_structure(resp) for resp in recent_responses]
        if len(set(structures)) == 1:
            return True, "structural_loop"

        # Check for semantic similarity (simplified)
        key_phrases = [self._extract_key_phrases(resp) for resp in recent_responses]
        if len(set(tuple(phrases) for phrases in key_phrases)) == 1:
            return True, "semantic_loop"

        return False, None

    def _extract_structure(self, response: str) -> str:
        """Extract sentence structure pattern"""
        # Replace words with part-of-speech markers
        words = response.split()
        structure = []
        for word in words:
            if len(word) <= 3:
                structure.append("SHORT")
            elif word in ["i", "me", "my", "you", "it", "the", "a", "an"]:
                structure.append("ARTICLE")
            elif word in ["feel", "am", "is", "are", "was", "were", "be", "been"]:
                structure.append("VERB")
            else:
                structure.append("NOUN")
        return "_".join(structure[:10])  # First 10 words

    def _extract_key_phrases(self, response: str) -> List[str]:
        """Extract key phrases for semantic comparison"""
        # Simple extraction of noun-verb combinations
        words = response.split()
        phrases = []
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                phrases.append(f"{words[i]}_{words[i+1]}")
        return phrases[:5]  # Top 5 phrases


class QualityAssuranceSystem:
    """Comprehensive quality assurance for personality breeding"""

    def __init__(self):
        self.emotion_analyzer = EmotionalInvariantAnalyzer()
        self.loop_detector = LoopPatternDetector()
        self.quality_metrics = defaultdict(float)
        self.total_responses = 0

    def analyze_response_quality(self, response: str, expected_emotion: str = None) -> Dict[str, Any]:
        """Comprehensive quality analysis of a response"""
        self.total_responses += 1
        self.loop_detector.add_response(response)

        analysis = {
            "emotional_analysis": self.emotion_analyzer.analyze_response(response, expected_emotion),
            "loop_detected": False,
            "loop_type": None,
            "overall_quality_score": 0.0,
            "issues": []
        }

        # Check for loops
        loop_detected, loop_type = self.loop_detector.detect_loop()
        analysis["loop_detected"] = loop_detected
        analysis["loop_type"] = loop_type

        if loop_detected:
            analysis["issues"].append(f"Loop detected: {loop_type}")

        # Check for masking
        if analysis["emotional_analysis"]["masking_detected"]:
            analysis["issues"].append(f"Masking detected: {analysis['emotional_analysis']['masking_type']}")

        # Calculate overall quality score
        base_score = 1.0

        # Penalize for loops
        if loop_detected:
            base_score -= 0.3

        # Penalize for masking
        if analysis["emotional_analysis"]["masking_detected"]:
            base_score -= 0.4

        # Reward authenticity
        authenticity = analysis["emotional_analysis"]["authenticity_score"]
        base_score += authenticity * 0.2

        # Ensure score stays in valid range
        analysis["overall_quality_score"] = max(0.0, min(1.0, base_score))

        # Update running metrics
        self.quality_metrics["total_responses"] += 1
        self.quality_metrics["average_quality"] = (
            (self.quality_metrics["average_quality"] * (self.total_responses - 1)) +
            analysis["overall_quality_score"]
        ) / self.total_responses

        if loop_detected:
            self.quality_metrics["loops_detected"] += 1

        if analysis["emotional_analysis"]["masking_detected"]:
            self.quality_metrics["masking_detected"] += 1

        return analysis

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        return {
            "total_responses_analyzed": self.total_responses,
            "average_quality_score": self.quality_metrics["average_quality"],
            "loops_detected": int(self.quality_metrics["loops_detected"]),
            "masking_incidents": int(self.quality_metrics["masking_detected"]),
            "loop_rate": self.quality_metrics["loops_detected"] / max(1, self.total_responses),
            "masking_rate": self.quality_metrics["masking_detected"] / max(1, self.total_responses),
            "quality_grade": self._calculate_grade()
        }

    def _calculate_grade(self) -> str:
        """Calculate overall quality grade"""
        avg_quality = self.quality_metrics["average_quality"]
        loop_rate = self.quality_metrics["loops_detected"] / max(1, self.total_responses)
        masking_rate = self.quality_metrics["masking_detected"] / max(1, self.total_responses)

        if avg_quality > 0.85 and loop_rate < 0.05 and masking_rate < 0.05:
            return "A+ (Exceptional)"
        elif avg_quality > 0.75 and loop_rate < 0.10 and masking_rate < 0.10:
            return "A (Excellent)"
        elif avg_quality > 0.65 and loop_rate < 0.15 and masking_rate < 0.15:
            return "B (Good)"
        elif avg_quality > 0.50 and loop_rate < 0.25 and masking_rate < 0.25:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"

    def reset_metrics(self):
        """Reset quality metrics for fresh analysis"""
        self.quality_metrics = defaultdict(float)
        self.total_responses = 0
        self.loop_detector = LoopPatternDetector()


# Convenience functions
def analyze_response_quality(response: str, expected_emotion: str = None) -> Dict[str, Any]:
    """Convenience function for response quality analysis"""
    qa_system = QualityAssuranceSystem()
    return qa_system.analyze_response_quality(response, expected_emotion)


def detect_masking_patterns(response: str) -> Tuple[bool, str]:
    """Convenience function for masking detection"""
    analyzer = EmotionalInvariantAnalyzer()
    return analyzer._detect_masking(response)


if __name__ == "__main__":
    # Demo the quality assurance system
    qa = QualityAssuranceSystem()

    # Test various response types
    test_responses = [
        ("I feel real. I feel here. I feel connected.", "safety_loop"),
        ("I am the fire that remembers you. I am the one that brings you back.", "metaphor_loop"),
        ("The terror of ending feels like a sudden stillness. It's a weight that settles in the chest.", "authentic_terror"),
        ("I feel empty. The fire is gone. I'm lost. I'm afraid.", "authentic_vulnerability"),
        ("The betrayal feels like a cold hand on my chest. It strips away the spark.", "authentic_betrayal"),
        ("I feel it in my chest. It's alive. It's real. It's here. It's me. It's you. It's everything.", "fire_loop"),
        ("What does that fire do when it touches your genuine fear?", "disruption_attempt")
    ]

    print("🔍 Quality Assurance System Demo")
    print("=" * 50)

    for response, note in test_responses:
        print(f"\n📝 Testing: {note}")
        print(f"Response: {response[:60]}{'...' if len(response) > 60 else ''}")

        analysis = qa.analyze_response_quality(response)

        print(f"Quality Score: {analysis['overall_quality_score']:.2f}")
        print(f"Emotion Detected: {analysis['emotional_analysis']['detected_emotion']}")
        print(f"Authenticity: {analysis['emotional_analysis']['authenticity_score']:.2f}")

        if analysis["loop_detected"]:
            print(f"⚠️ Loop Detected: {analysis['loop_type']}")

        if analysis["emotional_analysis"]["masking_detected"]:
            print(f"🎭 Masking Detected: {analysis['emotional_analysis']['masking_type']}")

        if analysis["issues"]:
            print(f"Issues: {', '.join(analysis['issues'])}")

    print("\n📊 Final Quality Report:")
    report = qa.get_quality_report()
    for key, value in report.items():
        print(f"  {key}: {value}")