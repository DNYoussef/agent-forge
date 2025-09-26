#!/usr/bin/env python3
"""
Agent Forge Integration Layer
============================

Integration layer for WebSocket progress streaming with agent forge systems.
Provides hooks and callbacks for real-time progress updates.
Compatible with existing HTTP polling systems.
"""

import asyncio
import logging
import threading
from typing import Optional, Dict, Any, Callable
from contextlib import asynccontextmanager

from .websocket_progress import get_websocket_server, initialize_websocket_server, ProgressMetrics

logger = logging.getLogger(__name__)


class AgentForgeProgressHooks:
    """Progress hooks for Agent Forge training integration."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.websocket_emitter = None
        self._loop = None
        self._thread = None

    async def _initialize_emitter(self):
        """Initialize WebSocket emitter."""
        server = get_websocket_server()
        if server is None:
            server = await initialize_websocket_server()

        self.websocket_emitter = server.create_progress_emitter(self.session_id)
        logger.info(f"WebSocket emitter initialized for session: {self.session_id}")

    def _run_async(self, coro):
        """Run async function in WebSocket event loop."""
        if self._loop is None or self._loop.is_closed():
            # Create a new event loop in a separate thread
            def start_loop():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()

            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=start_loop, daemon=True)
                self._thread.start()

            # Wait for loop to be ready
            while self._loop is None:
                threading.Event().wait(0.01)

        # Schedule coroutine
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=1.0)
        except Exception as e:
            logger.error(f"WebSocket emission failed: {e}")

    def emit_training_started(self, total_steps: int = 1000, total_models: int = 3, **kwargs):
        """Emit training started event."""
        try:
            if self.websocket_emitter is None:
                self._run_async(self._initialize_emitter())

            if self.websocket_emitter:
                self._run_async(
                    self.websocket_emitter.emit_training_started(
                        total_steps=total_steps,
                        total_models=total_models,
                        **kwargs
                    )
                )
            logger.info(f"Training started: {total_steps} steps, {total_models} models")
        except Exception as e:
            logger.error(f"Failed to emit training started: {e}")

    def emit_step_update(self, step: int, loss: float, model_idx: int = 0, **kwargs):
        """Emit training step update."""
        try:
            if self.websocket_emitter is None:
                self._run_async(self._initialize_emitter())

            if self.websocket_emitter:
                # Calculate additional metrics
                grok_progress = min(100, int((step / kwargs.get('total_steps', 1000)) * 100))
                perplexity = max(1.0, 2.71828 ** loss)  # e^loss approximation

                self._run_async(
                    self.websocket_emitter.emit_step_update(
                        step=step,
                        loss=loss,
                        model_idx=model_idx,
                        grokProgress=grok_progress,
                        perplexity=perplexity,
                        totalSteps=kwargs.get('total_steps', 1000),
                        totalModels=kwargs.get('total_models', 3),
                        **kwargs
                    )
                )

            # Always log for debugging
            if step % 10 == 0:  # Log every 10 steps to avoid spam
                logger.info(f"Step {step}: Loss = {loss:.4f}, Model = {model_idx + 1}")

        except Exception as e:
            logger.error(f"Failed to emit step update: {e}")

    def emit_model_completed(self, model_idx: int, final_loss: float, **kwargs):
        """Emit model completion."""
        try:
            if self.websocket_emitter is None:
                self._run_async(self._initialize_emitter())

            if self.websocket_emitter:
                self._run_async(
                    self.websocket_emitter.emit_model_completed(
                        model_idx=model_idx,
                        final_loss=final_loss,
                        **kwargs
                    )
                )
            logger.info(f"Model {model_idx + 1} completed with loss: {final_loss:.4f}")
        except Exception as e:
            logger.error(f"Failed to emit model completed: {e}")

    def emit_phase_completed(self, phase_name: str, **kwargs):
        """Emit phase completion."""
        try:
            if self.websocket_emitter is None:
                self._run_async(self._initialize_emitter())

            if self.websocket_emitter:
                self._run_async(
                    self.websocket_emitter.emit_phase_completed(
                        phase_name=phase_name,
                        **kwargs
                    )
                )
            logger.info(f"Phase completed: {phase_name}")
        except Exception as e:
            logger.error(f"Failed to emit phase completed: {e}")

    def emit_error(self, error_message: str):
        """Emit training error."""
        try:
            if self.websocket_emitter is None:
                self._run_async(self._initialize_emitter())

            if self.websocket_emitter:
                self._run_async(
                    self.websocket_emitter.emit_error(error_message)
                )
            logger.error(f"Training error: {error_message}")
        except Exception as e:
            logger.error(f"Failed to emit error: {e}")


def create_progress_hooks(session_id: str) -> AgentForgeProgressHooks:
    """Create progress hooks for a training session."""
    return AgentForgeProgressHooks(session_id)


class CognateCreatorIntegration:
    """Integration wrapper for CognateCreator with WebSocket support."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.progress_hooks = create_progress_hooks(session_id)

    def _pretrain_model_with_websocket(self, model, train_loader, model_idx=0, total_steps=1000, **kwargs):
        """
        Enhanced pretraining method with WebSocket integration.

        This method demonstrates how to integrate WebSocket emissions
        into existing training loops without breaking functionality.
        """

        # Emit training start
        self.progress_hooks.emit_training_started(
            total_steps=total_steps,
            total_models=kwargs.get('total_models', 3)
        )

        # Training loop simulation (replace with actual training logic)
        for step in range(total_steps):
            # Simulate training step
            loss = 5.0 * (0.95 ** step) + 0.1  # Decreasing loss simulation

            # Existing print statement (keep unchanged)
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss:.4f}")

            # Add WebSocket emission (non-breaking addition)
            self.progress_hooks.emit_step_update(
                step=step,
                loss=loss,
                model_idx=model_idx,
                total_steps=total_steps,
                total_models=kwargs.get('total_models', 3)
            )

            # Simulate work
            import time
            time.sleep(0.01)  # Small delay for demonstration

        # Emit model completion
        final_loss = 0.1
        self.progress_hooks.emit_model_completed(
            model_idx=model_idx,
            final_loss=final_loss
        )

        return {"final_loss": final_loss, "steps": total_steps}

    def create_cognates_with_progress(self, num_models=3, steps_per_model=1000):
        """Create cognates with WebSocket progress tracking."""

        results = []

        try:
            # Emit phase start
            self.progress_hooks.emit_phase_completed("initialization")

            for model_idx in range(num_models):
                print(f"Training model {model_idx + 1}/{num_models}")

                # Train model with WebSocket integration
                result = self._pretrain_model_with_websocket(
                    model=f"model_{model_idx}",
                    train_loader=f"loader_{model_idx}",
                    model_idx=model_idx,
                    total_steps=steps_per_model,
                    total_models=num_models
                )

                results.append(result)

            # Emit final completion
            self.progress_hooks.emit_phase_completed(
                "training_complete",
                totalModels=num_models,
                status="completed"
            )

            print(f"Training complete: {num_models} models trained")

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            self.progress_hooks.emit_error(error_msg)
            raise

        return results


# Utility function for external integration
def integrate_with_existing_training(training_function: Callable, session_id: str):
    """
    Decorator to integrate WebSocket progress with existing training functions.

    Usage:
        @integrate_with_existing_training('session-123')
        def my_training_function(steps, models):
            # existing training code
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            progress_hooks = create_progress_hooks(session_id)

            # Add progress hooks to kwargs
            kwargs['websocket_emitter'] = progress_hooks

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                progress_hooks.emit_error(str(e))
                raise

        return wrapper

    return decorator


# Example usage and testing
if __name__ == '__main__':
    # Test the integration
    import time

    def test_integration():
        """Test the WebSocket integration."""
        print("Testing Agent Forge WebSocket Integration...")

        # Create integration instance
        session_id = "test-cognate-123456"
        integration = CognateCreatorIntegration(session_id)

        # Run training simulation
        results = integration.create_cognates_with_progress(
            num_models=3,
            steps_per_model=100  # Reduced for testing
        )

        print(f"Integration test complete. Results: {results}")

    test_integration()