#!/usr/bin/env python3
"""
Generate multi-turn conversation trajectories for unified theory training.

This module processes multi-turn conversation datasets (like UltraChat) and converts
them into trajectory format suitable for training emotion manifolds, regime classifiers,
Φ heads, and self-tagging systems.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


class MultiTurnTrajectoryGenerator:
    """Generate trajectories from multi-turn conversation datasets"""

    def __init__(self, output_dir: str = "data/processed_datasets_unified"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_ultrachat_dataset(self, dataset_name: str, split: str = "train_sft", max_samples: int = 10000) -> List[Dict]:
        """Load and parse UltraChat dataset into conversation format"""
        print(f"📥 Loading {dataset_name} ({split})...")

        if dataset_name == "HuggingFaceH4/ultrachat_200k":
            # Try different splits
            for split_name in ['train_sft', 'train_gen']:
                try:
                    dataset = load_dataset(dataset_name, split=split_name, streaming=True)
                    print(f"  Using split: {split_name}")
                    break
                except:
                    continue
            else:
                dataset = load_dataset(dataset_name, split='train', streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)

        conversations = []
        count = 0

        for sample in dataset:
            if count >= max_samples:
                break

            # Parse conversation structure
            conversation = self._parse_conversation_sample(sample, dataset_name)
            if conversation and len(conversation) >= 4:  # Only multi-turn conversations
                conversations.append(conversation)
                count += 1

        print(f"✅ Loaded {len(conversations)} multi-turn conversations")
        return conversations

    def _parse_conversation_sample(self, sample: Dict, dataset_name: str) -> Optional[List[Dict]]:
        """Parse a single conversation sample into turn format"""
        turns = []

        try:
            if dataset_name == "HuggingFaceH4/ultrachat_200k":
                # Format: messages array with role/content
                if 'messages' in sample:
                    for msg in sample['messages']:
                        if isinstance(msg, dict):
                            role = msg.get('role', msg.get('from', 'unknown'))
                            content = msg.get('content', msg.get('value', ''))
                            if content:
                                turns.append({
                                    'role': role,
                                    'content': content,
                                    'turn_number': len(turns)
                                })

            elif dataset_name == "stingning/ultrachat":
                # Format: data array with turn objects
                if 'data' in sample:
                    for turn_data in sample['data']:
                        if isinstance(turn_data, dict):
                            # Try different field names
                            content = (turn_data.get('content') or
                                     turn_data.get('value') or
                                     turn_data.get('text') or
                                     str(turn_data))
                            role = turn_data.get('role', turn_data.get('from', 'unknown'))

                            if content and content != str(turn_data):  # Avoid raw dict serialization
                                turns.append({
                                    'role': role,
                                    'content': content,
                                    'turn_number': len(turns)
                                })

        except Exception as e:
            # Skip malformed samples
            return None

        # Validate we have alternating user/assistant turns
        if len(turns) < 4:
            return None

        # Check for basic alternation (allowing some flexibility)
        roles = [turn['role'] for turn in turns[:6]]  # Check first 6 turns
        user_count = sum(1 for r in roles if r.lower() in ['user', 'human'])
        assistant_count = sum(1 for r in roles if r.lower() in ['assistant', 'gpt', 'ai'])

        if user_count < 2 or assistant_count < 2:
            return None

        return turns

    def generate_trajectories_from_conversations(self, conversations: List[List[Dict]],
                                                 output_file: str = "multiturn_trajectories.jsonl") -> None:
        """
        Convert conversations into trajectory format for unified theory training.

        We now label HUMAN turns (user/human role) because they carry richer affective signals.
        Each trajectory stores the history leading up to that human turn as a list of turns,
        plus the target human message. Low-affect bland requests are skipped by a simple heuristic.
        """

        output_path = self.output_dir / output_file
        count = 0

        with open(output_path, 'w') as f:
            for conv_idx, conversation in enumerate(conversations):
                trajectories = self._create_user_target_trajectories(conversation, conv_idx)
                if not trajectories:
                    continue

                # Only keep a single representative trajectory per conversation to avoid duplicates.
                selected = self._select_best_trajectory(trajectories)
                f.write(json.dumps(selected, ensure_ascii=False) + '\n')
                count += 1
                if count >= 200000:
                    break

        print(f"✅ Generated {count} trajectories to {output_path}")

    def _is_emotional(self, text: str) -> bool:
        """Heuristic filter to drop bland/neutral requests."""
        if not text or len(text) < 30:
            return False
        cues = [
            "angry", "upset", "sad", "happy", "excited", "frustrated", "love", "hate",
            "worried", "anxious", "stressed", "depressed", "thrilled", "grateful",
            "annoyed", "delighted", "hurt", "betrayed", "lonely", "afraid", "scared",
            "furious", "overwhelmed", "tired", "exhausted"
        ]
        lowered = text.lower()
        return any(word in lowered for word in cues) or ("!" in text) or ("?" in text and "feel" in lowered)

    def _create_user_target_trajectories(self, conversation: List[Dict], conv_id: int) -> List[Dict]:
        """
        Produce trajectories where the TARGET is the human/user turn.
        History includes all prior turns (up to the target index) to support ownership/ΔΦ.
        """
        trajectories: List[Dict] = []
        for idx, turn in enumerate(conversation):
            role = turn.get("role", "").lower()
            if role not in ["user", "human"]:
                continue
            # Require an assistant message to start the window and end the window at this user turn.
            if idx == 0:
                continue
            # Find earliest assistant prior to idx to start the history window.
            start_idx = 0
            for j in range(idx):
                if conversation[j].get("role", "").lower() in ["assistant", "gpt", "ai"]:
                    start_idx = j
                    break
            history = conversation[start_idx:idx]
            if not history:
                continue
            if history[0].get("role", "").lower() not in ["assistant", "gpt", "ai"]:
                continue  # ensure history starts with assistant
            target_text = turn.get("content", "")
            if not self._is_emotional(target_text):
                continue  # skip bland / low-affect requests

            trajectories.append({
                "id": f"multiturn_{conv_id}_{idx}",
                "conversation_id": conv_id,
                "target_role": turn.get("role"),
                "history": history,
                "target": {"role": turn.get("role"), "content": target_text},
                "history_length": len(history),
                "total_turns": len(conversation),
                "source": "ultrachat_multiturn",
                "full_conversation": conversation,
            })
        return trajectories

    def _select_best_trajectory(self, trajectories: List[Dict]) -> Dict:
        """
        Pick one trajectory to represent a conversation.

        Preference order:
        1) Longer history length (more context is better)
        2) Longer target text (richer affect signal)
        3) Earliest in the conversation (stable tie-breaker)
        """

        def score(traj: Dict) -> tuple:
            target_text = traj.get("target", {}).get("content", "") or ""
            # Use negative turn index as final tie-breaker to keep deterministic order.
            turn_idx = int(traj.get("id", "0_0_0").split("_")[-1]) if "_" in traj.get("id", "") else 0
            return (
                traj.get("history_length", 0),
                len(target_text),
                -turn_idx,
            )

        return max(trajectories, key=score)

    def analyze_conversation_quality(self, conversations: List[List[Dict]]) -> Dict[str, Any]:
        """Analyze the quality of loaded conversations"""

        total_turns = []
        role_distributions = {'user': 0, 'assistant': 0, 'other': 0}
        conversation_lengths = []

        for conv in conversations[:1000]:  # Analyze first 1000
            conversation_lengths.append(len(conv))

            for turn in conv:
                role = turn['role'].lower()
                if role in ['user', 'human']:
                    role_distributions['user'] += 1
                elif role in ['assistant', 'gpt', 'ai']:
                    role_distributions['assistant'] += 1
                else:
                    role_distributions['other'] += 1

                total_turns.append(len(turn['content']))

        return {
            "num_conversations": len(conversations),
            "avg_conversation_length": sum(conversation_lengths) / len(conversation_lengths),
            "min_conversation_length": min(conversation_lengths),
            "max_conversation_length": max(conversation_lengths),
            "role_distribution": role_distributions,
            "avg_turn_length": sum(total_turns) / len(total_turns) if total_turns else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn trajectories for unified theory")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k",
                       help="Dataset name to process")
    parser.add_argument("--split", default="train_sft",
                       help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="Maximum conversations to load")
    parser.add_argument("--output_file", default="multiturn_trajectories.jsonl",
                       help="Output file name")

    args = parser.parse_args()

    generator = MultiTurnTrajectoryGenerator()

    # Load conversations
    conversations = generator.load_ultrachat_dataset(
        args.dataset,
        split=args.split,
        max_samples=args.max_samples
    )

    # Analyze quality
    analysis = generator.analyze_conversation_quality(conversations)
    print("📊 Conversation Analysis:")
    print(f"  Conversations: {analysis['num_conversations']}")
    print(f"  Avg length: {analysis['avg_conversation_length']:.1f} turns")
    print(f"  Role distribution: {analysis['role_distribution']}")
    print(f"  Avg turn length: {analysis['avg_turn_length']:.0f} chars")

    # Generate trajectories
    generator.generate_trajectories_from_conversations(
        conversations,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
