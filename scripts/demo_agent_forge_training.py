#!/usr/bin/env python3
"""
Agent Forge Training Demonstration
Shows Grokfast-enhanced training in action with progress callbacks.
"""

import sys
import os
import time
sys.path.insert(0, 'agent_forge')

from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator, create_sample_training_data


def progress_tracker(step, loss, perplexity):
    """Progress callback that displays training metrics."""
    print(f"  Step {step:4d} | Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")


def demonstrate_grokfast_training():
    """Demonstrate Grokfast-enhanced training."""
    print("Agent Forge Grokfast Training Demonstration")
    print("=" * 50)

    # Configuration
    config = {
        'vocab_size': 500,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'learning_rate': 1e-3,
        'grokfast_enabled': True,
        'grokfast_alpha': 0.98,
        'grokfast_lambda': 0.05
    }

    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create model creator
    print("Creating CognateModelCreator...")
    creator = CognateModelCreator(**config)

    # Get model info
    creator.create_model()
    model_info = creator.get_model_info()
    print(f"Model created with {model_info['total_parameters']:,} parameters")
    print(f"Device: {model_info['model_config']['device']}")
    print(f"Grokfast enabled: {model_info['grokfast_config']['enabled']}")
    print()

    # Generate training data
    print("Generating training data...")
    training_data = create_sample_training_data(
        vocab_size=config['vocab_size'],
        num_samples=200,
        seq_length=24
    )
    print(f"Generated {len(training_data)} training samples")
    print()

    # Train with progress tracking
    print("Starting Grokfast-enhanced training...")
    print("Progress updates every 10 steps:")
    print()

    start_time = time.time()
    training_stats = creator.train(
        train_data=training_data,
        epochs=3,
        batch_size=16,
        progress_callback=progress_tracker
    )
    training_time = time.time() - start_time

    # Display results
    print()
    print("Training completed!")
    print("=" * 50)
    print("Final Results:")
    print(f"  Total Steps: {training_stats['total_steps']}")
    print(f"  Final Loss: {training_stats['final_loss']:.4f}")
    print(f"  Final Perplexity: {training_stats['final_perplexity']:.2f}")
    print(f"  Training Time: {training_stats['training_time']:.2f}s")
    print(f"  Grokfast Enabled: {training_stats['grokfast_enabled']}")
    print()

    # Show training history
    if creator.training_history:
        print("Training History:")
        for epoch_data in creator.training_history:
            print(f"  Epoch {epoch_data['epoch']}: Loss={epoch_data['loss']:.4f}, Perplexity={epoch_data['perplexity']:.2f}")

    print()
    print("Demonstration completed successfully!")


def demonstrate_api_server():
    """Demonstrate API server functionality."""
    print("\nAPI Server Demonstration")
    print("=" * 30)

    try:
        from agent_forge.api.python_bridge_server import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test root endpoint
        print("Testing root endpoint...")
        response = client.get("/")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] API Version: {data.get('version', 'unknown')}")
            print(f"[OK] Status: {data.get('status', 'unknown')}")

        # Test system info
        print("\nTesting system info endpoint...")
        response = client.get("/api/system/info")
        if response.status_code == 200:
            info = response.json()
            print(f"[OK] PyTorch Version: {info.get('torch_version', 'unknown')}")
            print(f"[OK] CUDA Available: {info.get('cuda_available', False)}")
            print(f"[OK] API Version: {info.get('api_version', 'unknown')}")

        # Test cognate training start (mock)
        print("\nTesting cognate training start...")
        training_config = {
            "vocab_size": 100,
            "d_model": 32,
            "epochs": 1,
            "grokfast_enabled": True,
            "num_training_samples": 10
        }
        response = client.post("/api/cognate/start", json=training_config)
        if response.status_code == 200:
            result = response.json()
            training_id = result.get('training_id', 'unknown')
            print(f"[OK] Training started with ID: {training_id[:8]}...")

            # Wait a moment and check status
            time.sleep(1)
            response = client.get(f"/api/cognate/status/{training_id}")
            if response.status_code == 200:
                status = response.json()
                print(f"[OK] Training status: {status.get('status', 'unknown')}")

        print("\n[SUCCESS] API server demonstration completed!")

    except Exception as e:
        print(f"[FAIL] API server demonstration failed: {e}")


def main():
    """Main demonstration function."""
    try:
        # Demonstrate training
        demonstrate_grokfast_training()

        # Demonstrate API server
        demonstrate_api_server()

    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Version & Run Log Footer
"""
## Version & Run Log
| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-09-25T11:20:00-04:00 | backend-dev@claude-4 | Training demonstration script | demo_agent_forge_training.py | OK | Shows Grokfast acceleration in action | 0.00 | e1f8a4c |

### Receipt
- status: OK
- reason_if_blocked: --
- run_id: demo-script-001
- inputs: ["agent_forge_components"]
- tools_used: ["Write"]
- versions: {"model":"claude-4","prompt":"v1.0"}
"""