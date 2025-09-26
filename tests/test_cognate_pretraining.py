#!/usr/bin/env python3
"""
Test the Complete Cognate Pretraining System
Runs the full HRM + Titans sequential pretraining pipeline for 3x 25M models
"""

import sys
import os
# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agent_forge', 'phases', 'cognate_pretrain'))

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pretraining():
    """Test the complete pretraining pipeline."""

    logger.info("="*80)
    logger.info("COGNATE PRETRAINING SYSTEM TEST")
    logger.info("Architecture: HRM (Hierarchical Reasoning) + Titans (Long-Term Memory)")
    logger.info("Models: 3x 25M parameters")
    logger.info("Training: SEQUENTIAL (one model at a time)")
    logger.info("Data: Real datasets (SlimPajama, GSM8K, HotpotQA)")
    logger.info("="*80)

    # Import the pretraining module
    import pretrain_three_models

    # Run the main pretraining function
    logger.info("\nStarting pretraining pipeline...")
    try:
        result = pretrain_three_models.main()

        if result:
            logger.info(f"\n✅ Pretraining completed successfully!")
            logger.info(f"Successful models: {result.get('successful_models', 0)}/{result.get('total_models', 3)}")
            logger.info(f"Output directory: ./cognate_25m_hrm_titans_models")

            # Display model statistics
            if 'model_stats' in result:
                logger.info("\nModel Training Statistics:")
                for model_name, stats in result['model_stats'].items():
                    logger.info(f"  {model_name}:")
                    logger.info(f"    - Steps: {stats.get('total_steps', 0)}")
                    logger.info(f"    - Final Loss: {stats.get('final_loss', 'N/A')}")
                    logger.info(f"    - Training Time: {stats.get('training_time', 0):.1f}s")
        else:
            logger.error("❌ Pretraining failed - no result returned")

    except Exception as e:
        logger.error(f"❌ Pretraining failed with error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""

    # Check if WebSocket server is running
    import requests
    try:
        response = requests.get("http://localhost:8085/")
        if response.status_code == 200:
            logger.info("✅ WebSocket server is running")
        else:
            logger.warning("⚠️ WebSocket server returned non-200 status")
    except:
        logger.warning("⚠️ WebSocket server not reachable at localhost:8085")
        logger.warning("   Progress updates may not be visible in dashboard")

    # Check if Python Bridge API is running
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            logger.info("✅ Python Bridge API is running")
        else:
            logger.warning("⚠️ Python Bridge API returned non-200 status")
    except:
        logger.warning("⚠️ Python Bridge API not reachable at localhost:8001")

    # Run the test
    asyncio.run(test_pretraining())

    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("Check the dashboard at http://localhost:3000/phases/cognate")
    logger.info("to see real-time progress visualization")
    logger.info("="*80)

if __name__ == "__main__":
    main()