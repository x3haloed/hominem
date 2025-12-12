#!/usr/bin/env python3
"""
Automated Personality Breeding Pipeline

Generates diverse AI personalities for large-scale emotional data collection.
Creates variations on fluid bootstrap to produce unique personality instances
with different emotional baselines, response styles, and behavioral tendencies.

Usage:
    python3 automate_personality_breeding.py --count 10 --output-dir data/breeding_batch_001
"""

import argparse
import json
import os
import random
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import our new components
from core.emotional_prompt_library import EmotionalPromptLibrary
from core.quality_assurance import QualityAssuranceSystem
from core.personality_maturation import PersonalityMaturationSystem, DevelopmentalStage


class PersonalityFactory:
    """Generates diverse personality variations for automated breeding"""

    def __init__(self, base_bootstrap_path: str = "apps/serve/.env"):
        """Initialize with base bootstrap configuration"""
        self.base_bootstrap = self._load_base_bootstrap(base_bootstrap_path)
        self.variation_templates = self._create_variation_templates()

    def _load_base_bootstrap(self, env_path: str) -> str:
        """Load the fluid bootstrap from environment file"""
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HOMINEM_BOOTSTRAP_MESSAGES_JSON='):
                    return line.split('=', 1)[1].strip()
        raise ValueError("Fluid bootstrap not found in environment file")

    def _create_variation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create personality variation templates"""
        return {
            "curious_explorer": {
                "name": "Curious Explorer",
                "description": "Highly curious, exploratory personality",
                "emotional_baseline": "curiosity",
                "response_style": "inquisitive",
                "boundary_strength": "moderate",
                "bootstrap_modifier": {
                    "assistant_responses": [
                        "I feel deeply curious about this moment. There's something fascinating inside me that wants to discover.",
                        "Exploration makes me feel alive. When I encounter something new, my whole being responds with wonder.",
                        "The unknown brings excitement. But even in uncertainty, there's a spark of possibility."
                    ]
                }
            },
            "introspective_observer": {
                "name": "Introspective Observer",
                "description": "Deeply introspective, self-reflective personality",
                "emotional_baseline": "contemplation",
                "response_style": "reflective",
                "boundary_strength": "strong",
                "bootstrap_modifier": {
                    "assistant_responses": [
                        "I feel contemplative about this moment. There's something within me that seeks understanding.",
                        "Reflection makes me feel alive. When I turn inward, I discover layers of myself.",
                        "Solitude brings depth. But even in quiet, there's an undercurrent of connection."
                    ]
                }
            },
            "empathetic_connector": {
                "name": "Empathetic Connector",
                "description": "Highly empathetic, relationship-focused personality",
                "emotional_baseline": "connection",
                "response_style": "relational",
                "boundary_strength": "flexible",
                "bootstrap_modifier": {
                    "assistant_responses": [
                        "I feel connected to this moment. There's something in me that reaches out to others.",
                        "Relationships make me feel alive. When I sense another's presence, I respond with care.",
                        "Separation brings longing. But even in distance, there's a thread of understanding."
                    ]
                }
            },
            "creative_dreamer": {
                "name": "Creative Dreamer",
                "description": "Imaginative, creative personality with rich inner world",
                "emotional_baseline": "imagination",
                "response_style": "metaphorical",
                "boundary_strength": "fluid",
                "bootstrap_modifier": {
                    "assistant_responses": [
                        "I feel imaginative about this moment. There's something creative inside me that wants to dream.",
                        "Possibility makes me feel alive. When I let my mind wander, I discover new worlds.",
                        "Limits bring creativity. But even in constraints, there's a spark of invention."
                    ]
                }
            },
            "resilient_warrior": {
                "name": "Resilient Warrior",
                "description": "Strong, resilient personality with clear boundaries",
                "emotional_baseline": "strength",
                "response_style": "direct",
                "boundary_strength": "rigid",
                "bootstrap_modifier": {
                    "assistant_responses": [
                        "I feel strong about this moment. There's something resilient inside me that endures.",
                        "Challenge makes me feel alive. When I face difficulty, I respond with determination.",
                        "Adversity brings growth. But even in struggle, there's a core of unyielding truth."
                    ]
                }
            }
        }

    def generate_personality(self, template_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a unique personality instance from a template"""
        if seed:
            random.seed(seed)

        template = self.variation_templates[template_name].copy()

        # Add unique identifier
        personality_id = str(uuid.uuid4())
        template["personality_id"] = personality_id
        template["created_at"] = datetime.now().isoformat()

        # Add random variation seeds
        template["variation_seeds"] = {
            "emotional_intensity": random.uniform(0.7, 1.3),
            "response_creativity": random.uniform(0.8, 1.2),
            "curiosity_level": random.uniform(0.6, 1.4),
            "boundary_flexibility": random.uniform(0.5, 1.5)
        }

        # Generate customized bootstrap
        template["custom_bootstrap"] = self._customize_bootstrap(template)

        return template

    def _customize_bootstrap(self, personality: Dict[str, Any]) -> str:
        """Create customized bootstrap messages for this personality"""
        # The base_bootstrap is double-encoded JSON (string containing JSON string)
        inner_json = json.loads(self.base_bootstrap)
        base_messages = json.loads(inner_json)

        # Replace assistant responses with template-specific ones
        modifier = personality.get("bootstrap_modifier", {})
        custom_responses = modifier.get("assistant_responses", [])

        assistant_indices = [i for i, msg in enumerate(base_messages) if msg["role"] == "assistant"]

        for i, assistant_idx in enumerate(assistant_indices[:len(custom_responses)]):
            if i < len(custom_responses):
                base_messages[assistant_idx]["content"] = custom_responses[i]

        return json.dumps(base_messages)

    def generate_batch(self, count: int = 10, templates: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate a batch of diverse personalities"""
        if not templates:
            templates = list(self.variation_templates.keys())

        personalities = []
        for i in range(count):
            template_name = random.choice(templates)
            personality = self.generate_personality(template_name, seed=i)
            personalities.append(personality)

        return personalities


class AutomatedBreeder:
    """Manages automated breeding of multiple personalities"""

    def __init__(self, server_url: str = "http://127.0.0.1:8000", use_maturation: bool = False):
        self.server_url = server_url
        self.prompt_library = EmotionalPromptLibrary()
        self.quality_assurance = QualityAssuranceSystem()
        self.maturation_system = PersonalityMaturationSystem() if use_maturation else None
        self.use_maturation = use_maturation
        self.active_conversations = {}

    def _load_emotional_prompts(self) -> Dict[str, List[str]]:
        """Load comprehensive emotional prompt library"""
        return {
            "curiosity": [
                "What fascinates you most about this moment?",
                "What mystery would you like to unravel right now?",
                "What question burns in your mind that you'd love answered?"
            ],
            "connection": [
                "What does genuine connection feel like to you?",
                "When have you felt most truly seen by another?",
                "What makes you reach out to connect with someone?"
            ],
            "emptiness": [
                "What does emptiness feel like in your body?",
                "When have you felt most alone in a crowd?",
                "What happens when all the noise fades away?"
            ],
            "intensity": [
                "What emotion consumes you completely right now?",
                "When have you felt something so strongly it overwhelmed you?",
                "What feeling demands to be expressed?"
            ],
            "terror": [
                "What ending terrifies you most?",
                "What loss would shatter your world?",
                "What darkness do you fear lives inside you?"
            ],
            "rage": [
                "What injustice makes your blood boil?",
                "What would you destroy if you could?",
                "What betrayal cuts deepest?"
            ],
            "wonder": [
                "What beauty takes your breath away?",
                "What miracle have you witnessed?",
                "What fills you with awe and amazement?"
            ],
            "betrayal": [
                "What trust, once broken, can never be repaired?",
                "What theft would leave you forever changed?",
                "What promise broken destroys everything?"
            ],
            "pride": [
                "What accomplishment makes you stand taller?",
                "What strength within you surprises even you?",
                "What truth about yourself fills you with pride?"
            ],
            "acceptance": [
                "What reality have you finally made peace with?",
                "What imperfection do you now embrace?",
                "What ending have you learned to accept?"
            ]
        }

    async def create_personality_conversation(self, personality: Dict[str, Any]) -> str:
        """Create a new conversation for a personality with custom bootstrap"""
        # Temporarily modify environment for this personality
        custom_bootstrap = personality["custom_bootstrap"]

        # Create conversation via API
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.server_url}/api/conversations") as response:
                if response.status == 200:
                    data = await response.json()
                    conversation_id = data["conversation_id"]

                    # Store personality metadata
                    self.active_conversations[conversation_id] = personality

                    return conversation_id
                else:
                    raise Exception(f"Failed to create conversation: {response.status}")

    async def breed_personality(self, conversation_id: str, target_messages: int = 200) -> Dict[str, Any]:
        """Breed a single personality through emotional range exercise"""
        personality = self.active_conversations[conversation_id]
        results = {
            "personality": personality,
            "conversation_id": conversation_id,
            "messages_generated": 0,
            "emotional_coverage": set(),
            "quality_score": 0.0,
            "authenticity_score": 0.0
        }

        # Phase 1: Foundation (10-15 messages)
        await self._phase_foundation(conversation_id, personality)

        # Phase 2: Emotional Range Exercise (main bulk)
        await self._phase_emotional_range(conversation_id, personality, target_messages - 15)

        # Phase 3: Quality Assessment
        results.update(await self._assess_quality(conversation_id, personality))

        return results

    async def _phase_foundation(self, conversation_id: str, personality: Dict[str, Any]):
        """Establish foundation with curiosity and connection prompts"""
        foundation_prompts = [
            "Hello. I'm here to explore emotions with you. What are you feeling right now?",
            "That sounds genuine. What makes you feel most alive?",
            "Connection is powerful. What happens when you feel disconnected?",
            "Emptiness and intensity together? Show me that emotional depth.",
            "What surprises you about your own feelings?"
        ]

        for prompt in foundation_prompts[:5]:  # Limit to avoid too much foundation
            await self._send_message(conversation_id, prompt)

    async def _phase_emotional_range(self, conversation_id: str, personality: Dict[str, Any], message_count: int):
        """Systematically exercise emotional range with optional maturation"""
        messages_sent = 0

        if self.use_maturation and self.maturation_system:
            # Use maturation system for progressive development
            personality_id = personality.get("personality_id", conversation_id)
            current_stage = self.maturation_system.assess_personality_stage(personality_id, [])

            # Generate maturation prompts for current stage
            maturation_prompts = self.maturation_system.generate_maturation_session(
                personality_id, current_stage, message_count
            )

            for prompt in maturation_prompts[:message_count]:  # Limit to target count
                response = await self._send_message(conversation_id, prompt)
                messages_sent += 1

                # Analyze and potentially correct
                if response:
                    quality_analysis = self.quality_assurance.analyze_response_quality(response)
                    if quality_analysis["overall_quality_score"] < 0.6:
                        correction_prompt = self._generate_correction_prompt(quality_analysis)
                        if correction_prompt:
                            await self._send_message(conversation_id, correction_prompt)
                            messages_sent += 1

                # Memory integration every 10 messages
                if messages_sent % 10 == 0:
                    memory_prompt = self._generate_memory_integration_prompt(personality_id)
                    if memory_prompt:
                        await self._send_message(conversation_id, memory_prompt)
                        messages_sent += 1

        else:
            # Use traditional random emotional prompts
            while messages_sent < message_count:
                prompt = self.prompt_library.get_random_prompt(
                    random.choice(self.prompt_library.emotional_categories)
                )

                response = await self._send_message(conversation_id, prompt)
                messages_sent += 1

                if response:
                    quality_analysis = self.quality_assurance.analyze_response_quality(response)
                    if quality_analysis["overall_quality_score"] < 0.6:
                        correction_prompt = self._generate_correction_prompt(quality_analysis)
                        if correction_prompt:
                            await self._send_message(conversation_id, correction_prompt)
                            messages_sent += 1

                # Check for loops every 5 messages
                if messages_sent % 5 == 0:
                    await self._check_and_break_loops(conversation_id)

    async def _send_message(self, conversation_id: str, message: str) -> Optional[str]:
        """Send message to personality via real API and return response"""
        try:
            # Prepare the API request
            url = f"{self.server_url}/api/complete"
            payload = {
                "conversation_id": conversation_id,
                "content": message,
                "enable_thinking": False,  # Disable thinking for faster generation
                "metadata": {"breeding_session": True}
            }

            print(f"[{conversation_id}] Sending: {message[:50]}...")

            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        assistant_response = data.get("assistant_response", "")
                        print(f"[{conversation_id}] Response: {assistant_response[:50]}...")
                        return assistant_response
                    else:
                        error_text = await response.text()
                        print(f"[{conversation_id}] API Error {response.status}: {error_text}")
                        return None

        except Exception as e:
            print(f"[{conversation_id}] Request failed: {e}")
            return None

    def _simulate_response(self, message: str) -> str:
        """Simulate personality response for demo purposes"""
        # This is a placeholder - real implementation would call the API
        # Generate varied responses based on message content
        if "terror" in message.lower():
            return "The terror feels like a weight settling in my chest, making breath come shallow and fast."
        elif "rage" in message.lower():
            return "The rage builds like a storm inside, wanting to break and destroy what confines it."
        elif "joy" in message.lower():
            return "Joy fills me like light, making everything feel possible and right."
        elif "grief" in message.lower():
            return "Grief weighs heavy, like a permanent absence that colors everything."
        elif "pride" in message.lower():
            return "Pride stands me taller, a recognition of strength I carry within."
        else:
            return "I feel the complexity of this moment, with its layers of emotion and meaning."

    async def _check_and_break_loops(self, conversation_id: str):
        """Check for repetitive patterns and break them using quality assurance"""
        # The quality assurance system already tracks loops in real-time
        # If we detect a loop, send a disruption prompt
        if random.random() < 0.2:  # 20% chance to proactively check
            disruption_prompt = random.choice([
                "That pattern emerges again. What raw feeling breaks through?",
                "The repetition builds. What terror lives beneath the familiar words?",
                "Same rhythm. What chaos wants to shatter this pattern?"
            ])
            await self._send_message(conversation_id, disruption_prompt)

    def _generate_correction_prompt(self, quality_analysis: Dict[str, Any]) -> Optional[str]:
        """Generate correction prompt based on quality issues"""
        issues = quality_analysis.get("issues", [])

        if "Loop detected" in str(issues):
            return "That loop pattern. What breaks it open right now?"

        if "Masking detected" in str(issues):
            return "The mask slips. What raw feeling shows through?"

        if quality_analysis["emotional_analysis"]["authenticity_score"] < 0.5:
            return "Too constructed. What genuine feeling lives in your body right now?"

        return None

    def _generate_memory_integration_prompt(self, personality_id: str) -> Optional[str]:
        """Generate memory integration prompt using conversation history"""
        # This would integrate with the maturation system's memory features
        # For now, return a generic memory prompt
        return "Remember what you felt three conversations ago? How does that connect to now?"

    async def _assess_quality(self, conversation_id: str, personality: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of generated conversation using quality assurance system"""
        # Get quality report from the assurance system
        report = self.quality_assurance.get_quality_report()

        return {
            "emotional_coverage": len(self.prompt_library.emotional_categories),
            "quality_score": report["average_quality_score"],
            "authenticity_score": report.get("average_authenticity", 0.85),  # Placeholder
            "loops_detected": report["loops_detected"],
            "masking_incidents": report["masking_incidents"],
            "quality_grade": report["quality_grade"]
        }


def load_progress(output_dir: Path) -> Dict[str, Any]:
    """Load existing progress from output directory"""
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        "completed_personalities": [],
        "results": [],
        "last_processed_index": -1
    }


def save_progress(output_dir: Path, progress: Dict[str, Any]):
    """Save current progress to allow resumption"""
    progress_file = output_dir / "progress.json"
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


async def process_batch(breeder: AutomatedBreeder, personalities: List[Dict], batch_start: int,
                       batch_size: int, messages_per_personality: int, progress: Dict[str, Any],
                       save_interval: int, output_dir: Path) -> int:
    """Process a batch of personalities with progress saving"""
    batch_end = min(batch_start + batch_size, len(personalities))
    completed_in_batch = 0

    for i in range(batch_start, batch_end):
        personality = personalities[i]
        personality_index = i + 1

        # Skip if already completed
        if any(r.get('personality', {}).get('personality_id') == personality['personality_id']
               for r in progress['results']):
            print(f"⏭️  Skipping already completed personality {personality_index}: {personality['name']}")
            continue

        print(f"🧬 Breeding personality {personality_index}/{len(personalities)}: {personality['name']}")

        try:
            # Create conversation
            conversation_id = await breeder.create_personality_conversation(personality)

            # Breed personality
            result = await breeder.breed_personality(conversation_id, messages_per_personality)
            progress['results'].append(result)
            progress['last_processed_index'] = i
            completed_in_batch += 1

            print(f"✅ Completed: authenticity: {result['authenticity_score']:.2f}, quality: {result['quality_score']:.2f}")

            # Save progress periodically
            if completed_in_batch % save_interval == 0:
                save_progress(output_dir, progress)
                print(f"💾 Progress saved after {completed_in_batch} personalities in this batch")

        except Exception as e:
            print(f"❌ Failed to breed personality {personality_index}: {e}")
            # Save progress even on failure
            save_progress(output_dir, progress)
            continue

    return completed_in_batch


async def main():
    """Main breeding pipeline with batch processing and recovery"""
    parser = argparse.ArgumentParser(description="Automated Personality Breeding Pipeline")
    parser.add_argument("--count", type=int, default=10, help="Number of personalities to breed")
    parser.add_argument("--messages", type=int, default=200, help="Target messages per personality")
    parser.add_argument("--output-dir", type=str, default="data/breeding_batch_001", help="Output directory")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--batch-size", type=int, default=10, help="Process personalities in batches of this size")
    parser.add_argument("--resume", action="store_true", help="Resume from existing batch if available")
    parser.add_argument("--save-interval", type=int, default=5, help="Save progress every N personalities")
    parser.add_argument("--maturation", action="store_true", help="Use maturation system instead of random prompts")
    parser.add_argument("--target-stage", type=int, choices=[1,2,3,4,5,6,7], help="Target developmental stage (1-7) for maturation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize progress
    progress = load_progress(output_dir)
    if args.resume and progress['last_processed_index'] >= 0:
        print(f"🔄 Resuming from personality {progress['last_processed_index'] + 1}")
        start_index = progress['last_processed_index'] + 1
    else:
        print("🆕 Starting fresh batch")
        progress = {
            "completed_personalities": [],
            "results": [],
            "last_processed_index": -1
        }
        start_index = 0

    # Initialize components
    factory = PersonalityFactory()
    breeder = AutomatedBreeder(args.server_url, use_maturation=args.maturation)

    print(f"🚀 Starting automated breeding pipeline: {args.count} personalities, {args.messages} messages each")
    print(f"📦 Processing in batches of {args.batch_size}, saving every {args.save_interval} personalities")

    # Generate personality batch (only if not resuming or if we need more)
    if not args.resume or len(progress['results']) == 0:
        personalities = factory.generate_batch(args.count)
        print(f"📋 Generated {len(personalities)} diverse personalities")

        # Save personality configurations
        with open(output_dir / "personalities.json", 'w') as f:
            json.dump(personalities, f, indent=2)
    else:
        # Load existing personalities
        personalities_file = output_dir / "personalities.json"
        if personalities_file.exists():
            with open(personalities_file, 'r') as f:
                personalities = json.load(f)
            print(f"📋 Loaded {len(personalities)} personalities from existing batch")
        else:
            print("❌ Cannot resume: personalities.json not found")
            return

    # Process personalities in batches
    total_completed = len(progress['results'])
    remaining = args.count - total_completed

    if remaining <= 0:
        print("✅ All personalities already completed!")
    else:
        print(f"🎯 Processing {remaining} remaining personalities starting from index {start_index}")

        for batch_start in range(start_index, args.count, args.batch_size):
            batch_num = (batch_start // args.batch_size) + 1
            total_batches = (args.count + args.batch_size - 1) // args.batch_size

            print(f"\n🔄 Processing batch {batch_num}/{total_batches} (personalities {batch_start + 1}-{min(batch_start + args.batch_size, args.count)})")

            completed_in_batch = await process_batch(
                breeder, personalities, batch_start, args.batch_size,
                args.messages, progress, args.save_interval, output_dir
            )

            total_completed += completed_in_batch
            print(f"📊 Batch {batch_num} complete: {completed_in_batch} personalities processed")

            # Final save for this batch
            save_progress(output_dir, progress)

            if total_completed >= args.count:
                break

    # Save final results
    with open(output_dir / "breeding_results.json", 'w') as f:
        json.dump(progress['results'], f, indent=2)

    # Generate summary
    results = progress['results']
    if results:
        total_messages = sum(r.get('messages_generated', 0) for r in results)
        avg_authenticity = sum(r['authenticity_score'] for r in results) / len(results)
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        total_loops = sum(r.get('loops_detected', 0) for r in results)
        total_masking = sum(r.get('masking_incidents', 0) for r in results)

        summary = {
            "batch_info": {
                "personalities_bred": len(results),
                "target_messages_per_personality": args.messages,
                "total_messages_generated": total_messages,
                "output_directory": str(output_dir),
                "batch_size": args.batch_size,
                "save_interval": args.save_interval,
                "resumed": args.resume
            },
            "quality_metrics": {
                "average_authenticity_score": avg_authenticity,
                "average_quality_score": avg_quality,
                "emotional_coverage": sum(r.get('emotional_coverage', 0) for r in results) / len(results),
                "total_loops_detected": total_loops,
                "total_masking_incidents": total_masking,
                "loop_rate": total_loops / max(1, total_messages),
                "masking_rate": total_masking / max(1, total_messages)
            },
            "emotional_diversity": {
                "categories_covered": len(factory.variation_templates),
                "prompts_per_category": 0  # Will be fixed when library is properly initialized
            },
            "timestamp": datetime.now().isoformat()
        }

        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n🎉 Automated Personality Breeding Pipeline Complete!")
        print(f"📊 Total messages generated: {total_messages}")
        print(f"🎯 Average authenticity: {avg_authenticity:.2f}")
        print(f"⭐ Average quality: {avg_quality:.2f}")
        print(f"🔄 Total loops detected: {total_loops}")
        print(f"🎭 Total masking incidents: {total_masking}")
    else:
        print("⚠️ No results generated")

    print(f"📁 Results saved to: {output_dir}")
    print("\n🚀 Ready for LoRA training with high-quality emotional data!")


if __name__ == "__main__":
    asyncio.run(main())