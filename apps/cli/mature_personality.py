#!/usr/bin/env python3
"""
Personality Maturation Demo

Demonstrates how to grow a basic personality through developmental stages
to achieve sophisticated emotional intelligence and complex dialogue.

Usage:
    python3 mature_personality.py --personality-id demo_personality_001 --sessions 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import asyncio
import aiohttp

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from core.personality_maturation import PersonalityMaturationSystem, DevelopmentalStage


class PersonalityMaturationDemo:
    """Demo system for maturing personalities through developmental stages"""

    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        self.server_url = server_url
        self.maturation_system = PersonalityMaturationSystem()
        self.conversation_history = []

    async def create_demo_personality(self) -> str:
        """Create a new personality for maturation demo"""
        # Create conversation via API
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.server_url}/api/conversations") as response:
                if response.status == 200:
                    data = await response.json()
                    return data["conversation_id"]
                else:
                    raise Exception(f"Failed to create conversation: {response.status}")

    async def run_maturation_session(self, conversation_id: str, target_stage: DevelopmentalStage,
                                   session_length: int = 30) -> Dict[str, Any]:
        """Run a maturation session for a specific developmental stage"""

        # Assess current stage (would be more sophisticated in real implementation)
        current_stage = DevelopmentalStage.EMOTIONAL_AWARENESS

        print(f"🧠 Starting maturation session for {conversation_id}")
        print(f"🎯 Target Stage: {target_stage.name} ({target_stage.value})")
        print(f"📝 Session Length: {session_length} messages")

        # Generate maturation prompts
        prompts = self.maturation_system.generate_maturation_session(
            conversation_id, current_stage, session_length
        )

        session_results = {
            "conversation_id": conversation_id,
            "target_stage": target_stage.value,
            "session_length": len(prompts),
            "conversations": []
        }

        # Run the conversation
        for i, prompt in enumerate(prompts):
            print(f"\n📨 Turn {i+1}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

            response = await self._send_message(conversation_id, prompt)
            print(f"📥 Response: {response[:60]}{'...' if len(response) > 60 else ''}")

            self.conversation_history.append({
                "turn": i + 1,
                "prompt": prompt,
                "response": response,
                "stage": current_stage.value
            })

            session_results["conversations"].append({
                "turn": i + 1,
                "prompt": prompt,
                "response": response
            })

        # Assess development
        final_assessment = {
            "complexity_score": self.maturation_system._calculate_complexity_score(self.conversation_history),
            "emotional_range": self.maturation_system._assess_emotional_range(self.conversation_history),
            "abstract_reasoning": self.maturation_system._measure_abstract_reasoning(self.conversation_history)
        }

        session_results["final_assessment"] = final_assessment

        print("
📊 Session Assessment:"        print(f"  Complexity Score: {final_assessment['complexity_score']:.2f}")
        print(f"  Emotional Range: {final_assessment['emotional_range']}")
        print(f"  Abstract Reasoning: {final_assessment['abstract_reasoning']:.2f}")

        return session_results

    async def _send_message(self, conversation_id: str, message: str) -> str:
        """Send message to personality via API"""
        try:
            url = f"{self.server_url}/api/complete"
            payload = {
                "conversation_id": conversation_id,
                "content": message,
                "enable_thinking": False,
                "metadata": {"maturation_session": True}
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("assistant_response", "No response")
                    else:
                        error_text = await response.text()
                        print(f"API Error: {error_text}")
                        return f"Error: {response.status}"

        except Exception as e:
            print(f"Request failed: {e}")
            return f"Error: {e}"

    def save_session_results(self, results: Dict[str, Any], output_file: str):
        """Save session results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")


async def main():
    """Run personality maturation demo"""
    parser = argparse.ArgumentParser(description="Personality Maturation Demo")
    parser.add_argument("--personality-id", type=str, help="Existing personality ID to mature")
    parser.add_argument("--sessions", type=int, default=1, help="Number of maturation sessions to run")
    parser.add_argument("--session-length", type=int, default=30, help="Messages per session")
    parser.add_argument("--target-stage", type=int, choices=[1,2,3,4,5,6,7], default=3,
                       help="Target developmental stage (1-7)")
    parser.add_argument("--output-dir", type=str, default="data/maturation_demo",
                       help="Output directory for results")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000",
                       help="Server URL")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize maturation demo
    demo = PersonalityMaturationDemo(args.server_url)

    target_stage = DevelopmentalStage(args.target_stage)

    print("🧠 Personality Maturation Demo")
    print("=" * 50)
    print(f"🎯 Target Stage: {target_stage.name}")
    print(f"📝 Sessions: {args.sessions}")
    print(f"💬 Messages per session: {args.session_length}")

    # Create or use existing personality
    if args.personality_id:
        conversation_id = args.personality_id
        print(f"👤 Using existing personality: {conversation_id}")
    else:
        conversation_id = await demo.create_demo_personality()
        print(f"🆕 Created new personality: {conversation_id}")

    all_results = {
        "personality_id": conversation_id,
        "target_stage": target_stage.value,
        "sessions_run": args.sessions,
        "session_results": [],
        "overall_development": {}
    }

    # Run maturation sessions
    for session_num in range(args.sessions):
        print(f"\n🔄 Running Session {session_num + 1}/{args.sessions}")

        session_result = await demo.run_maturation_session(
            conversation_id, target_stage, args.session_length
        )

        all_results["session_results"].append(session_result)

        # Save intermediate results
        intermediate_file = output_dir / f"session_{session_num + 1}_results.json"
        demo.save_session_results(session_result, str(intermediate_file))

    # Calculate overall development
    if all_results["session_results"]:
        total_complexity = sum(r["final_assessment"]["complexity_score"]
                              for r in all_results["session_results"])
        total_emotional_range = sum(r["final_assessment"]["emotional_range"]
                                   for r in all_results["session_results"])
        total_abstract = sum(r["final_assessment"]["abstract_reasoning"]
                            for r in all_results["session_results"])

        num_sessions = len(all_results["session_results"])

        all_results["overall_development"] = {
            "average_complexity": total_complexity / num_sessions,
            "total_emotional_range": total_emotional_range,
            "average_abstract_reasoning": total_abstract / num_sessions,
            "developmental_progress": self._assess_progress(all_results)
        }

    # Save final results
    final_file = output_dir / "maturation_demo_results.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("
🎉 Maturation Demo Complete!"    print(f"📊 Average Complexity: {all_results['overall_development']['average_complexity']:.2f}")
    print(f"🎭 Total Emotional Range: {all_results['overall_development']['total_emotional_range']}")
    print(f"🧠 Average Abstract Reasoning: {all_results['overall_development']['average_abstract_reasoning']:.2f}")
    print(f"📁 Results saved to {output_dir}")

def _assess_progress(results):
    """Assess overall developmental progress"""
    # This would be more sophisticated in real implementation
    return "Developing emotional awareness and self-reflection capabilities"


if __name__ == "__main__":
    asyncio.run(main())