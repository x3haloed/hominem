#!/usr/bin/env python3
"""
Batch emotion labeling script for unlabeled assistant messages.

This script uses the EmotionEngine to automatically label all assistant messages
in the database that don't already have emotion labels.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.serve.emotion_engine import EmotionEngine
from apps.serve.database import DatabaseManager


class BatchEmotionLabeler:
    """Batch emotion labeler for unlabeled messages"""

    def __init__(self, config_path: str = "config/inference.toml", batch_size: int = 10):
        self.config_path = config_path
        self.batch_size = batch_size
        self.emotion_engine: Optional[EmotionEngine] = None

    async def initialize(self):
        """Initialize the emotion engine"""
        try:
            self.emotion_engine = EmotionEngine(self.config_path)
            print("✅ Emotion engine initialized successfully")
            print(f"   Model: {self.emotion_engine.emotion_label_config.get('model_id', 'unknown')}")
            print(f"   Endpoint: {self.emotion_engine.emotion_label_config.get('endpoint_url', 'unknown')}")
            api_key_env = self.emotion_engine.emotion_label_config.get('api_key_env')
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if api_key:
                    print(f"   API Key: {api_key[:10]}... (loaded from {api_key_env})")
                else:
                    print(f"   API Key: NOT FOUND in environment variable {api_key_env}")
                    print("   Make sure your .env file is loaded and contains the correct API key")
                    raise ValueError(f"API key not found in environment variable {api_key_env}")
            else:
                print("   API Key: No api_key_env configured")
        except Exception as e:
            print(f"❌ Failed to initialize emotion engine: {e}")
            raise

    async def get_unlabeled_messages(self, db: DatabaseManager) -> List[Dict[str, Any]]:
        """Get all assistant messages that don't have emotion labels"""
        query = """
        SELECT
            m.id,
            m.conversation_id,
            c.conversation_id as conversation_uuid,
            m.message_index,
            m.content,
            c.title
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE m.role = 'assistant'
        AND m.id NOT IN (
            SELECT message_id FROM emotion_labels WHERE labeler = 'auto'
        )
        ORDER BY m.created_at ASC
        """

        cursor = db.connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        messages = []
        for row in rows:
            messages.append({
                'id': row[0],
                'conversation_id': row[1],  # integer foreign key
                'conversation_uuid': row[2],  # string UUID for API
                'message_index': row[3],
                'content': row[4],
                'conversation_title': row[5] or f"Conversation {row[2][:8]}..."
            })

        print(f"📋 Found {len(messages)} unlabeled assistant messages")
        return messages

    async def get_message_context(self, db: DatabaseManager, message_id: int) -> Dict[str, Any]:
        """Get context for a message (previous user message if available)"""
        query = """
        SELECT
            m.content as assistant_content,
            prev_m.content as user_content,
            prev_m.message_index as user_index
        FROM messages m
        LEFT JOIN messages prev_m ON m.conversation_id = prev_m.conversation_id
            AND prev_m.message_index = m.message_index - 1
            AND prev_m.role = 'user'
        WHERE m.id = ?
        """

        cursor = db.connection.cursor()
        cursor.execute(query, (message_id,))
        row = cursor.fetchone()

        if not row:
            return {}

        return {
            'assistant_content': row[0],
            'user_content': row[1] if row[1] else "Hello",  # Default if no previous user message
            'user_index': row[2]
        }

    async def label_single_message(
        self,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Label a single message using the emotion engine"""
        try:
            # Use the message pair labeling approach
            # Treat the previous user message as speaker, assistant response as respondent
            speaker_message = context.get('user_content', 'Hello')
            respondent_message = message['content']

            labels = await self.emotion_engine.label_message_pair(
                speaker_message=speaker_message,
                respondent_message=respondent_message,
                speaker_role="user",
                respondent_role="assistant",
                context=f"Conversation: {message.get('conversation_title', 'Unknown')}"
            )

            return labels

        except Exception as e:
            print(f"⚠️ Failed to label message {message['id']}: {e}")
            return None

    async def process_batch(self, db: DatabaseManager, messages: List[Dict[str, Any]]) -> int:
        """Process a batch of messages using batch API calls"""
        if not messages:
            return 0

        print(f"🎭 Labeling batch of {len(messages)} messages...")

        # Collect all message pairs for batch processing
        message_pairs = []
        message_data = []  # Keep track of message data for saving results

        for message in messages:
            # Get context for the message
            context = await self.get_message_context(db, message['id'])

            message_pairs.append({
                "pair_id": str(message['id']),
                "speaker_message": context.get('user_content', 'Hello'),
                "respondent_message": message['content'],
                "speaker_role": "user",
                "respondent_role": "assistant",
                "context": f"Conversation: {message.get('conversation_title', 'Unknown')}"
            })

            message_data.append(message)

        try:
            # Send batch request
            batch_results = await self.emotion_engine.label_message_pairs_batch(message_pairs)

            labeled_count = 0
            for result in batch_results:
                pair_index = result["pair_index"]
                labels = result["labels"]
                message = message_data[pair_index]

                try:
                    # Save labels to database
                    db.add_emotion_label(
                        conversation_id=message['conversation_uuid'],
                        message_index=message['message_index'],
                        labeler="auto",
                        valence=labels.get("valence"),
                        arousal=labels.get("arousal"),
                        dominance=labels.get("dominance"),
                        predictive_discrepancy=labels.get("predictive_discrepancy"),
                        temporal_directionality=labels.get("temporal_directionality"),
                        social_broadcast=labels.get("social_broadcast"),
                        confidence=labels.get("confidence"),
                        notes=labels.get("notes")
                    )

                    labeled_count += 1
                    print(f"✅ Labeled message {message['id']} (confidence: {labels.get('confidence', 'N/A'):.2f})")

                except Exception as e:
                    print(f"❌ Failed to save labels for message {message['id']}: {e}")

            return labeled_count

        except Exception as e:
            print(f"❌ Batch labeling failed: {e}")
            # Fall back to individual processing for this batch
            print("🔄 Falling back to individual processing...")

            labeled_count = 0
            for i, message in enumerate(messages):
                print(f"🎭 Labeling message {i+1}/{len(messages)} individually: {message['conversation_title']} (ID: {message['id']})")

                # Get context for the message
                context = await self.get_message_context(db, message['id'])

                # Label the message individually
                labels = await self.label_single_message(message, context)

                if labels:
                    try:
                        # Save labels to database
                        db.add_emotion_label(
                            conversation_id=message['conversation_uuid'],
                            message_index=message['message_index'],
                            labeler="auto",
                            valence=labels.get("valence"),
                            arousal=labels.get("arousal"),
                            dominance=labels.get("dominance"),
                            predictive_discrepancy=labels.get("predictive_discrepancy"),
                            temporal_directionality=labels.get("temporal_directionality"),
                            social_broadcast=labels.get("social_broadcast"),
                            confidence=labels.get("confidence"),
                            notes=labels.get("notes")
                        )

                        labeled_count += 1
                        print(f"✅ Labeled message {message['id']} (confidence: {labels.get('confidence', 'N/A'):.2f})")

                    except Exception as e:
                        print(f"❌ Failed to save labels for message {message['id']}: {e}")
                else:
                    print(f"⏭️ Skipped message {message['id']} (labeling failed)")

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.5)

            return labeled_count

    async def run_batch_labeling(self, db_path: str, dry_run: bool = False, max_messages: int = None):
        """Run the batch labeling process"""
        print("🚀 Starting batch emotion labeling...")
        print(f"📊 Database: {db_path}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🧪 Dry run: {dry_run}")
        print()

        # Initialize database
        db = DatabaseManager(db_path)

        try:
            # Get unlabeled messages
            unlabeled_messages = await self.get_unlabeled_messages(db)

            if not unlabeled_messages:
                print("🎉 No unlabeled messages found! All assistant messages are already labeled.")
                return

            # Apply max_messages limit if specified
            if max_messages is not None:
                unlabeled_messages = unlabeled_messages[:max_messages]
                print(f"📋 Limited to processing {len(unlabeled_messages)} messages (max_messages={max_messages})")

            if dry_run:
                print(f"🧪 DRY RUN: Would label {len(unlabeled_messages)} messages")
                for msg in unlabeled_messages[:5]:  # Show first 5
                    print(f"  - {msg['conversation_title']} (ID: {msg['id']})")
                if len(unlabeled_messages) > 5:
                    print(f"  ... and {len(unlabeled_messages) - 5} more")
                return

            # Process messages in batches
            total_labeled = 0
            total_processed = 0

            for i in range(0, len(unlabeled_messages), self.batch_size):
                batch = unlabeled_messages[i:i + self.batch_size]
                print(f"\n📦 Processing batch {i//self.batch_size + 1}/{(len(unlabeled_messages) + self.batch_size - 1)//self.batch_size}")

                batch_labeled = await self.process_batch(db, batch)
                total_labeled += batch_labeled
                total_processed += len(batch)

                print(f"📊 Batch complete: {batch_labeled}/{len(batch)} messages labeled")
                print(f"📈 Running total: {total_labeled}/{total_processed} messages labeled")

                # Commit after each batch
                db.connection.commit()

                # Optional: pause between batches to be respectful to API
                if i + self.batch_size < len(unlabeled_messages):
                    print("⏳ Waiting 2 seconds before next batch...")
                    await asyncio.sleep(2)

            print("\n🎉 Batch labeling complete!")
            print(f"📊 Total messages labeled: {total_labeled}/{len(unlabeled_messages)}")

        finally:
            db.close()

        # Close emotion engine
        if self.emotion_engine:
            await self.emotion_engine.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Batch emotion labeling for unlabeled assistant messages"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/Users/chad/temp/hominem/conversations.db",
        help="Path to conversations database"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference.toml",
        help="Path to inference config file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of messages to process per batch"
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Maximum number of messages to process (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be labeled without actually doing it"
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connection with a single message before starting batch processing"
    )

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"❌ Database not found: {args.db_path}")
        sys.exit(1)

    # Create labeler
    labeler = BatchEmotionLabeler(args.config, args.batch_size)

    try:
        # Initialize
        await labeler.initialize()

        # Test API if requested
        if args.test_api:
            print("\n🧪 Testing API connection...")
            db = DatabaseManager(args.db_path)
            test_messages = await labeler.get_unlabeled_messages(db)
            if test_messages:
                print(f"Testing with message ID {test_messages[0]['id']}...")
                context = await labeler.get_message_context(db, test_messages[0]['id'])
                result = await labeler.label_single_message(test_messages[0], context)
                if result:
                    print("✅ API test successful!")
                    print(f"   Sample result: valence={result.get('valence')}, confidence={result.get('confidence')}")
                else:
                    print("❌ API test failed - no result returned")
                    db.close()
                    sys.exit(1)
            else:
                print("⚠️ No unlabeled messages to test with")
            db.close()
            print("API test completed. Exiting.")
            return

        # Run labeling
        await labeler.run_batch_labeling(args.db_path, args.dry_run, args.max_messages)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
