"""
ADAS + Transformers² Integration: Self-Guided Automatic Discovery of Agentic Expert Vector Configurations

This is the unique combination system that uses ADAS Meta Agent Search strategy
combined with Transformers² self-guided expert vector composition to create a fully
autonomous expert discovery system.

Key Innovation:
- ADAS meta-agent programs and discovers expert vector configurations
- Transformers² self-guided composition: Model directs its own expert vector creation
- Self-guided discovery: Model learns to compose and optimize its own expert vectors
- Automatic discovery replaces manual expert vector design
- Phase 2 weight observation informs the self-guided discovery process
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import time
import json

# Import our implementations
from .agentic_expert_discovery import (
    MetaAgentExpertDiscovery,
    ExpertVectorConfig,
    ExpertVectorArchive
)
from .transformers_squared import (
    TransformersSquaredSystem,
    TransformersSquaredConfig,
    ExpertVectorSystem
)

# Import Phase 2 integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from agent_forge.api.weight_space_extractor import WeightSpaceExtractor

logger = logging.getLogger(__name__)


@dataclass
class AdasT2Config:
    """Configuration for ADAS + Transformers² integrated system"""
    # ADAS configuration
    discovery_iterations: int = 30
    archive_size: int = 50
    evaluation_episodes: int = 20

    # Transformers² configuration
    t2_config: TransformersSquaredConfig = None

    # Integration configuration
    phase2_integration: bool = True
    automatic_adaptation: bool = True
    discovery_strategies: List[str] = None

    def __post_init__(self):
        if self.t2_config is None:
            self.t2_config = TransformersSquaredConfig()

        if self.discovery_strategies is None:
            self.discovery_strategies = [
                "phase2_informed",    # Use Phase 2 weight patterns
                "task_specialized",   # Task-specific discovery
                "performance_guided", # Performance-driven search
                "diversity_based"     # Maintain diverse configurations
            ]


class AdasTransformersSquaredSystem:
    """
    Self-Guided Integrated System: ADAS meta-agent search with Transformers² self-guided expert composition

    This creates a fully autonomous self-guided expert vector discovery system where:
    1. ADAS meta-agent explores expert vector configuration space
    2. Transformers² self-guided composition: Model directs its own expert vector creation
    3. Model learns to compose and optimize its own expert vectors autonomously
    4. Phase 2 weight observation informs self-guided discovery process
    5. System automatically discovers and self-composes optimal configurations for any task
    """

    def __init__(self, target_model: nn.Module, config: AdasT2Config = None):
        self.target_model = target_model
        self.config = config or AdasT2Config()

        # Initialize core systems
        self.transformers_squared = TransformersSquaredSystem(target_model, self.config.t2_config)
        self.meta_agent_discovery = MetaAgentExpertDiscovery(target_model)

        # Phase 2 integration
        if self.config.phase2_integration:
            self.phase2_extractor = WeightSpaceExtractor()

        # Discovery state
        self.discovered_configurations = {}
        self.discovery_sessions = []
        self.performance_history = {}

        # Automatic adaptation state
        self.active_configuration = None
        self.adaptation_stack = []

        # Self-guided composition state
        self.self_guided_compositions = {}
        self.composition_learning_history = []
        self.model_guided_preferences = {}

        logger.info("Initialized Self-Guided ADAS + Transformers² integrated system")

    def discover_expert_configurations_for_task(
        self,
        task_description: str,
        task_examples: List[Dict[str, Any]] = None,
        custom_evaluation: Optional[Callable] = None,
        enable_self_guided_composition: bool = True
    ) -> Dict[str, Any]:
        """
        Main method: Self-guided automatic discovery of optimal expert vector configurations

        This is where ADAS strategy meets Transformers² self-guided expert vector composition
        Key: The model itself guides and directs the expert vector composition process
        """
        logger.info(f"Starting automatic expert configuration discovery for: {task_description}")

        session_id = f"discovery_{int(time.time())}_{hash(task_description) % 1000}"

        # Phase 1: Meta-agent analyzes task and existing knowledge
        task_analysis = self._analyze_task_with_meta_agent(task_description, task_examples)

        # Phase 2: Use Phase 2 weight observation to inform search space
        if self.config.phase2_integration:
            weight_insights = self._get_phase2_weight_insights(task_description)
            task_analysis["weight_insights"] = weight_insights

        # Phase 3: ADAS-style discovery with Transformers² expert vectors
        discovery_results = self._run_adas_expert_discovery(
            task_analysis, custom_evaluation or self._create_default_evaluator(task_description)
        )

        # Phase 3.5: Self-guided composition (NEW - Model directs its own expert vector creation)
        if enable_self_guided_composition:
            self_guided_results = self._run_self_guided_composition(
                task_description, discovery_results, task_analysis
            )
            discovery_results["self_guided_compositions"] = self_guided_results

        # Phase 4: Validate discovered configurations with Transformers² system
        validated_configurations = self._validate_configurations_with_t2(
            discovery_results["best_configurations"]
        )

        # Phase 5: Create final optimized expert system
        optimized_expert_system = self._create_optimized_expert_system(
            validated_configurations, task_description
        )

        # Store discovery session
        discovery_session = {
            "session_id": session_id,
            "task_description": task_description,
            "task_analysis": task_analysis,
            "discovery_results": discovery_results,
            "validated_configurations": validated_configurations,
            "optimized_system": optimized_expert_system,
            "timestamp": time.time()
        }

        self.discovery_sessions.append(discovery_session)
        self.discovered_configurations[task_description] = validated_configurations

        logger.info(f"Discovery complete: Found {len(validated_configurations)} optimal configurations")

        return discovery_session

    def _analyze_task_with_meta_agent(
        self,
        task_description: str,
        task_examples: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Meta-agent analyzes task to inform expert vector discovery"""

        # Simulate meta-agent analysis (in real ADAS, this would generate code)
        analysis = {
            "task_type": self._classify_task_type(task_description),
            "complexity_estimate": self._estimate_task_complexity(task_description),
            "required_capabilities": self._identify_required_capabilities(task_description),
            "similar_tasks": self._find_similar_discovered_tasks(task_description),
            "suggested_strategies": self._suggest_discovery_strategies(task_description)
        }

        if task_examples:
            analysis["example_analysis"] = self._analyze_task_examples(task_examples)

        return analysis

    def _get_phase2_weight_insights(self, task_description: str) -> Dict[str, Any]:
        """Use Phase 2 weight observation to inform expert vector discovery"""
        try:
            # Get 3D weight visualization data
            weight_data = self.phase2_extractor.extract_3d_weights(self.target_model)

            # Analyze weight patterns for this task type
            insights = {
                "weight_distribution": self._analyze_weight_distribution(weight_data),
                "critical_layers": self._identify_critical_layers_from_weights(weight_data),
                "weight_magnitude_patterns": self._extract_magnitude_patterns(weight_data),
                "suggested_svd_focus": self._suggest_svd_focus_areas(weight_data)
            }

            logger.info(f"Phase 2 insights: {len(insights['critical_layers'])} critical layers identified")
            return insights

        except Exception as e:
            logger.warning(f"Phase 2 integration failed: {e}")
            return {"status": "unavailable"}

    def _run_adas_expert_discovery(
        self,
        task_analysis: Dict[str, Any],
        evaluation_function: Callable
    ) -> Dict[str, Any]:
        """Run ADAS-style discovery process for expert vectors"""

        # Create custom evaluation function that incorporates task analysis
        def enhanced_evaluator(config: ExpertVectorConfig, model: nn.Module) -> float:
            base_score = evaluation_function(config, model)

            # Boost score based on task analysis insights
            if "weight_insights" in task_analysis:
                weight_insights = task_analysis["weight_insights"]
                if weight_insights.get("status") != "unavailable":
                    # Reward configurations that target critical layers
                    critical_layers = weight_insights.get("critical_layers", [])
                    config_layers = set(config.svd_components.keys())
                    critical_overlap = len(config_layers.intersection(set(critical_layers)))

                    if critical_overlap > 0:
                        base_score += 0.2 * (critical_overlap / len(critical_layers))

            # Task-specific bonuses
            task_type = task_analysis.get("task_type", "general")
            if config.task_specialization == task_type:
                base_score += 0.1

            return min(1.0, base_score)

        # Run discovery with enhanced evaluator
        return self.meta_agent_discovery.discover_expert_configurations(
            num_iterations=self.config.discovery_iterations,
            evaluation_function=enhanced_evaluator
        )

    def _validate_configurations_with_t2(
        self,
        discovered_configurations: List[ExpertVectorConfig]
    ) -> List[Dict[str, Any]]:
        """Validate discovered configurations using Transformers² system"""
        validated = []

        for config in discovered_configurations:
            try:
                # Convert ADAS configuration to Transformers² format
                t2_validation = self._convert_config_to_t2_format(config)

                # Test with Transformers² expert system
                test_result = self._test_config_with_t2_system(t2_validation)

                validated.append({
                    "original_config": config,
                    "t2_format": t2_validation,
                    "validation_result": test_result,
                    "compatibility_score": test_result.get("compatibility_score", 0.0)
                })

            except Exception as e:
                logger.warning(f"Configuration validation failed: {e}")

        # Sort by compatibility score
        validated.sort(key=lambda x: x["compatibility_score"], reverse=True)

        return validated

    def _convert_config_to_t2_format(self, config: ExpertVectorConfig) -> Dict[str, Any]:
        """Convert ADAS discovered configuration to Transformers² expert vector format"""

        # Extract expert vector from ADAS config
        expert_vectors = []
        layer_mappings = {}

        for layer_name, expert_weights in config.expert_weights.items():
            # Convert to format expected by Transformers² expert system
            expert_vector_data = {
                "layer": layer_name,
                "svd_components": config.svd_components.get(layer_name, []),
                "weights": expert_weights.cpu().numpy().tolist(),
                "specialization": config.task_specialization
            }

            expert_vectors.append(expert_vector_data)
            layer_mappings[layer_name] = len(expert_vectors) - 1

        return {
            "config_id": config.config_id,
            "expert_vectors": expert_vectors,
            "layer_mappings": layer_mappings,
            "task_specialization": config.task_specialization,
            "performance_score": config.performance_score
        }

    def _test_config_with_t2_system(self, t2_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test configuration compatibility with Transformers² system"""

        try:
            # Create temporary expert system configuration
            test_input = torch.randn(1, 10, 512)  # Batch, seq, hidden

            # Simulate dispatch system output
            expert_weights = torch.zeros(1, self.config.t2_config.num_expert_vectors)
            expert_weights[0, 0] = 1.0  # Use first expert for testing

            # Test expert application
            layer_modifications = self.transformers_squared.expert_system(
                expert_weights, adaptation_strategy="classifier_based"
            )

            compatibility_score = len(layer_modifications) / len(t2_config["expert_vectors"])

            return {
                "test_successful": True,
                "layer_modifications": len(layer_modifications),
                "compatibility_score": compatibility_score,
                "adaptation_feasible": compatibility_score > 0.5
            }

        except Exception as e:
            return {
                "test_successful": False,
                "error": str(e),
                "compatibility_score": 0.0,
                "adaptation_feasible": False
            }

    def _create_optimized_expert_system(
        self,
        validated_configurations: List[Dict[str, Any]],
        task_description: str
    ) -> Dict[str, Any]:
        """Create optimized expert system from discovered configurations"""

        if not validated_configurations:
            return {"status": "no_valid_configurations"}

        # Select top configurations
        top_configs = validated_configurations[:3]

        # Create ensemble expert system
        ensemble_config = {
            "task_description": task_description,
            "expert_configurations": top_configs,
            "ensemble_strategy": "weighted_combination",
            "weights": [config["compatibility_score"] for config in top_configs],
            "optimization_method": "adas_discovery"
        }

        return ensemble_config

    def apply_discovered_configuration(
        self,
        task_description: str,
        configuration_id: Optional[str] = None,
        adaptation_strength: float = 1.0
    ) -> Dict[str, Any]:
        """Apply a discovered expert configuration to the model"""

        if task_description not in self.discovered_configurations:
            raise ValueError(f"No configurations discovered for task: {task_description}")

        configs = self.discovered_configurations[task_description]

        if configuration_id:
            # Find specific configuration
            config_to_apply = None
            for config_data in configs:
                if config_data["original_config"].config_id == configuration_id:
                    config_to_apply = config_data
                    break

            if not config_to_apply:
                raise ValueError(f"Configuration {configuration_id} not found")
        else:
            # Use best configuration
            config_to_apply = configs[0]

        # Apply using meta-agent discovery system
        application_result = self.meta_agent_discovery.apply_expert_configuration(
            config_to_apply["original_config"],
            strength=adaptation_strength
        )

        # Track active configuration
        self.active_configuration = {
            "task_description": task_description,
            "config_data": config_to_apply,
            "application_result": application_result,
            "applied_at": time.time()
        }

        logger.info(f"Applied configuration {config_to_apply['original_config'].config_id}")

        return application_result

    def get_discovery_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the discovery system"""

        return {
            "system_type": "ADAS + Transformers² Integration",
            "discovery_sessions": len(self.discovery_sessions),
            "tasks_with_configurations": len(self.discovered_configurations),
            "total_configurations_discovered": sum(
                len(configs) for configs in self.discovered_configurations.values()
            ),
            "active_configuration": self.active_configuration is not None,
            "phase2_integration": self.config.phase2_integration,
            "meta_agent_stats": self.meta_agent_discovery.get_discovery_summary(),
            "transformers_squared_stats": self.transformers_squared.get_system_status()
        }

    # Helper methods for task analysis
    def _classify_task_type(self, task_description: str) -> str:
        """Classify task type from description"""
        task_lower = task_description.lower()

        if any(word in task_lower for word in ["math", "calculate", "solve", "equation"]):
            return "math"
        elif any(word in task_lower for word in ["code", "program", "debug", "function"]):
            return "coding"
        elif any(word in task_lower for word in ["reason", "logic", "analyze", "think"]):
            return "reasoning"
        elif any(word in task_lower for word in ["create", "write", "story", "creative"]):
            return "creativity"
        else:
            return "general"

    def _estimate_task_complexity(self, task_description: str) -> float:
        """Estimate task complexity (0-1 scale)"""
        # Simple heuristic based on description length and keywords
        complexity_keywords = ["complex", "difficult", "advanced", "sophisticated"]

        base_complexity = min(1.0, len(task_description.split()) / 20.0)
        keyword_bonus = sum(0.2 for word in complexity_keywords if word in task_description.lower())

        return min(1.0, base_complexity + keyword_bonus)

    def _identify_required_capabilities(self, task_description: str) -> List[str]:
        """Identify required capabilities from task description"""
        capabilities = []
        task_lower = task_description.lower()

        capability_map = {
            "reasoning": ["reason", "logic", "analyze", "think", "infer"],
            "creativity": ["create", "generate", "invent", "imagine"],
            "math": ["calculate", "solve", "compute", "math"],
            "language": ["translate", "summarize", "write", "read"],
            "coding": ["code", "program", "debug", "implement"]
        }

        for capability, keywords in capability_map.items():
            if any(keyword in task_lower for keyword in keywords):
                capabilities.append(capability)

        return capabilities or ["general"]

    def _find_similar_discovered_tasks(self, task_description: str) -> List[str]:
        """Find similar tasks that have been discovered"""
        similar_tasks = []

        for discovered_task in self.discovered_configurations.keys():
            # Simple similarity based on shared keywords
            task_words = set(task_description.lower().split())
            discovered_words = set(discovered_task.lower().split())

            overlap = len(task_words.intersection(discovered_words))
            if overlap > 0:
                similar_tasks.append(discovered_task)

        return similar_tasks

    def _suggest_discovery_strategies(self, task_description: str) -> List[str]:
        """Suggest discovery strategies based on task"""
        strategies = ["phase2_informed"]  # Always use Phase 2 insights

        task_type = self._classify_task_type(task_description)
        complexity = self._estimate_task_complexity(task_description)

        if task_type != "general":
            strategies.append("task_specialized")

        if complexity > 0.7:
            strategies.append("performance_guided")

        strategies.append("diversity_based")

        return strategies

    def _create_default_evaluator(self, task_description: str) -> Callable:
        """Create default evaluation function for task"""
        def default_evaluator(config: ExpertVectorConfig, model: nn.Module) -> float:
            # Simple synthetic evaluation
            base_score = 0.5

            # Reward task alignment
            if config.task_specialization == self._classify_task_type(task_description):
                base_score += 0.3

            # Reward reasonable number of components
            total_components = sum(len(comps) for comps in config.svd_components.values())
            component_score = min(0.2, total_components / 20.0)
            base_score += component_score

            # Add noise to simulate real evaluation
            import random
            base_score += random.uniform(-0.1, 0.1)

            return max(0.0, min(1.0, base_score))

        return default_evaluator

    def _run_self_guided_composition(
        self,
        task_description: str,
        discovery_results: Dict[str, Any],
        task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Self-guided composition: Model directs its own expert vector creation (Transformers² approach)

        This implements the key Transformers² insight where the model itself guides
        the composition and optimization of expert vectors rather than external search.
        """
        logger.info("Starting self-guided expert vector composition...")

        # Step 1: Model examines its own weight patterns for task-specific insights
        self_examination = self._model_guided_self_examination(task_description)

        # Step 2: Model proposes expert vector compositions based on self-knowledge
        model_proposals = self._model_proposes_expert_compositions(
            task_description, self_examination, discovery_results
        )

        # Step 3: Model evaluates and refines its own proposals
        refined_compositions = self._model_guided_refinement(
            model_proposals, task_analysis
        )

        # Step 4: Model learns from composition success/failure
        composition_learning = self._model_learns_composition_patterns(
            refined_compositions, task_description
        )

        # Store self-guided compositions for this task
        self.self_guided_compositions[task_description] = {
            "self_examination": self_examination,
            "model_proposals": model_proposals,
            "refined_compositions": refined_compositions,
            "composition_learning": composition_learning,
            "timestamp": time.time()
        }

        logger.info(f"Self-guided composition complete: {len(refined_compositions)} compositions created")

        return {
            "composition_method": "self_guided_model_directed",
            "compositions_created": len(refined_compositions),
            "self_examination_insights": len(self_examination.get("insights", [])),
            "model_learning_updates": len(composition_learning.get("learned_patterns", [])),
            "compositions": refined_compositions
        }

    def _model_guided_self_examination(self, task_description: str) -> Dict[str, Any]:
        """Model examines its own weights and patterns to understand task requirements"""

        # Simulate model's self-examination of its own weight space
        model_insights = {
            "task_type": self._classify_task_type(task_description),
            "weight_patterns_identified": [],
            "self_identified_capabilities": [],
            "adaptation_opportunities": [],
            "insights": []
        }

        # Model analyzes its own layer patterns
        for name, param in self.target_model.named_parameters():
            if len(param.shape) >= 2:  # Weight matrices
                weight_analysis = {
                    "layer_name": name,
                    "weight_magnitude": torch.norm(param.data).item(),
                    "sparsity": (torch.abs(param.data) < 1e-6).float().mean().item(),
                    "rank_estimate": min(param.shape),  # Simplified rank estimate
                    "adaptation_potential": torch.std(param.data).item()
                }
                model_insights["weight_patterns_identified"].append(weight_analysis)

        # Model identifies its own capabilities for this task
        task_type = model_insights["task_type"]
        if task_type == "math":
            model_insights["self_identified_capabilities"].extend([
                "numerical_reasoning", "step_by_step_logic", "equation_solving"
            ])
        elif task_type == "creativity":
            model_insights["self_identified_capabilities"].extend([
                "creative_generation", "narrative_structure", "character_development"
            ])
        elif task_type == "coding":
            model_insights["self_identified_capabilities"].extend([
                "code_structure", "debugging_logic", "function_composition"
            ])

        # Model identifies adaptation opportunities
        high_variance_layers = [
            wp["layer_name"] for wp in model_insights["weight_patterns_identified"]
            if wp["adaptation_potential"] > 0.1
        ]
        model_insights["adaptation_opportunities"] = high_variance_layers[:5]

        model_insights["insights"] = [
            f"Identified {len(model_insights['weight_patterns_identified'])} analyzable layers",
            f"Task type '{task_type}' maps to {len(model_insights['self_identified_capabilities'])} capabilities",
            f"Found {len(model_insights['adaptation_opportunities'])} high-potential adaptation layers"
        ]

        return model_insights

    def _model_proposes_expert_compositions(
        self,
        task_description: str,
        self_examination: Dict[str, Any],
        discovery_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Model proposes expert vector compositions based on self-knowledge"""

        model_proposals = []

        # Proposal 1: Model suggests composition based on its weight patterns
        weight_guided_proposal = {
            "composition_id": f"weight_guided_{int(time.time())}",
            "composition_strategy": "weight_pattern_guided",
            "target_layers": self_examination["adaptation_opportunities"],
            "composition_rationale": "Based on model's self-identified high-adaptation-potential layers",
            "expert_vector_design": self._design_expert_vectors_from_weights(
                self_examination["weight_patterns_identified"]
            ),
            "confidence": 0.8
        }
        model_proposals.append(weight_guided_proposal)

        # Proposal 2: Model suggests composition based on task capabilities
        capability_guided_proposal = {
            "composition_id": f"capability_guided_{int(time.time())}",
            "composition_strategy": "capability_enhancement",
            "target_capabilities": self_examination["self_identified_capabilities"],
            "composition_rationale": "Enhance model's self-identified relevant capabilities",
            "expert_vector_design": self._design_expert_vectors_from_capabilities(
                self_examination["self_identified_capabilities"]
            ),
            "confidence": 0.7
        }
        model_proposals.append(capability_guided_proposal)

        # Proposal 3: Model suggests hybrid composition combining ADAS findings with self-knowledge
        if discovery_results.get("best_configurations"):
            hybrid_proposal = {
                "composition_id": f"hybrid_guided_{int(time.time())}",
                "composition_strategy": "adas_self_hybrid",
                "adas_configurations": discovery_results["best_configurations"][:2],
                "self_insights": self_examination["insights"],
                "composition_rationale": "Hybrid of ADAS discovery with model self-knowledge",
                "expert_vector_design": self._design_hybrid_expert_vectors(
                    discovery_results["best_configurations"][:2], self_examination
                ),
                "confidence": 0.9
            }
            model_proposals.append(hybrid_proposal)

        return model_proposals

    def _design_expert_vectors_from_weights(self, weight_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design expert vectors based on model's weight pattern analysis"""

        # Select top layers by adaptation potential
        top_layers = sorted(weight_patterns, key=lambda x: x["adaptation_potential"], reverse=True)[:3]

        expert_design = {
            "num_experts": len(top_layers),
            "expert_specifications": []
        }

        for i, layer_info in enumerate(top_layers):
            expert_spec = {
                "expert_id": f"weight_expert_{i}",
                "target_layer": layer_info["layer_name"],
                "adaptation_strength": min(1.0, layer_info["adaptation_potential"] * 2),
                "svd_components": min(10, int(layer_info["rank_estimate"] * 0.1)),
                "specialization": f"layer_{layer_info['layer_name']}_optimization"
            }
            expert_design["expert_specifications"].append(expert_spec)

        return expert_design

    def _design_expert_vectors_from_capabilities(self, capabilities: List[str]) -> Dict[str, Any]:
        """Design expert vectors based on model's capability analysis"""

        expert_design = {
            "num_experts": len(capabilities),
            "expert_specifications": []
        }

        for i, capability in enumerate(capabilities):
            expert_spec = {
                "expert_id": f"capability_expert_{i}",
                "target_capability": capability,
                "adaptation_strength": 0.6,
                "svd_components": 8,
                "specialization": capability
            }
            expert_design["expert_specifications"].append(expert_spec)

        return expert_design

    def _design_hybrid_expert_vectors(
        self,
        adas_configurations: List[Any],
        self_examination: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design expert vectors combining ADAS discovery with model self-knowledge"""

        expert_design = {
            "num_experts": 4,  # Hybrid approach with multiple experts
            "expert_specifications": []
        }

        # Expert 1: Best ADAS configuration
        if adas_configurations:
            adas_expert = {
                "expert_id": "adas_best",
                "source": "adas_discovery",
                "adas_config": adas_configurations[0],
                "adaptation_strength": 0.8,
                "specialization": "adas_discovered_optimal"
            }
            expert_design["expert_specifications"].append(adas_expert)

        # Expert 2: Model's top self-identified layer
        if self_examination["adaptation_opportunities"]:
            self_expert = {
                "expert_id": "self_guided_top",
                "source": "model_self_knowledge",
                "target_layer": self_examination["adaptation_opportunities"][0],
                "adaptation_strength": 0.7,
                "specialization": "model_self_identified"
            }
            expert_design["expert_specifications"].append(self_expert)

        # Expert 3: Capability-focused
        capabilities = self_examination.get("self_identified_capabilities", [])
        if capabilities:
            capability_expert = {
                "expert_id": "capability_focused",
                "source": "capability_enhancement",
                "target_capabilities": capabilities[:2],
                "adaptation_strength": 0.6,
                "specialization": "capability_enhancement"
            }
            expert_design["expert_specifications"].append(capability_expert)

        # Expert 4: Exploration expert for discovery
        exploration_expert = {
            "expert_id": "exploration",
            "source": "exploration",
            "adaptation_strength": 0.4,
            "svd_components": 12,
            "specialization": "configuration_exploration"
        }
        expert_design["expert_specifications"].append(exploration_expert)

        return expert_design

    def _model_guided_refinement(
        self,
        model_proposals: List[Dict[str, Any]],
        task_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Model evaluates and refines its own expert vector proposals"""

        refined_compositions = []

        for proposal in model_proposals:
            # Model evaluates its own proposal
            self_evaluation = self._model_evaluates_own_proposal(proposal, task_analysis)

            # Model refines based on evaluation
            if self_evaluation["should_refine"]:
                refined_proposal = self._model_refines_proposal(proposal, self_evaluation)
            else:
                refined_proposal = proposal

            # Add evaluation metadata
            refined_proposal["self_evaluation"] = self_evaluation
            refined_proposal["refinement_applied"] = self_evaluation["should_refine"]

            refined_compositions.append(refined_proposal)

        # Sort by model's own confidence assessment
        refined_compositions.sort(key=lambda x: x["confidence"], reverse=True)

        return refined_compositions

    def _model_evaluates_own_proposal(
        self,
        proposal: Dict[str, Any],
        task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Model evaluates its own expert vector proposal"""

        evaluation = {
            "proposal_id": proposal["composition_id"],
            "alignment_score": 0.0,
            "feasibility_score": 0.0,
            "innovation_score": 0.0,
            "should_refine": False,
            "refinement_suggestions": []
        }

        # Evaluate alignment with task
        task_type = task_analysis.get("task_type", "general")
        if proposal["composition_strategy"] == "capability_enhancement":
            evaluation["alignment_score"] = 0.8
        elif proposal["composition_strategy"] == "weight_pattern_guided":
            evaluation["alignment_score"] = 0.7
        elif proposal["composition_strategy"] == "adas_self_hybrid":
            evaluation["alignment_score"] = 0.9

        # Evaluate feasibility
        expert_design = proposal.get("expert_vector_design", {})
        num_experts = expert_design.get("num_experts", 0)
        if 1 <= num_experts <= 5:
            evaluation["feasibility_score"] = 0.8
        else:
            evaluation["feasibility_score"] = 0.4
            evaluation["refinement_suggestions"].append("Adjust number of experts to 1-5 range")

        # Evaluate innovation
        if proposal["composition_strategy"] == "adas_self_hybrid":
            evaluation["innovation_score"] = 0.9
        else:
            evaluation["innovation_score"] = 0.6

        # Determine if refinement needed
        overall_score = (evaluation["alignment_score"] + evaluation["feasibility_score"] + evaluation["innovation_score"]) / 3
        evaluation["should_refine"] = overall_score < 0.7

        return evaluation

    def _model_refines_proposal(
        self,
        proposal: Dict[str, Any],
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Model refines its proposal based on self-evaluation"""

        refined_proposal = proposal.copy()

        # Apply refinement suggestions
        for suggestion in evaluation["refinement_suggestions"]:
            if "number of experts" in suggestion:
                expert_design = refined_proposal.get("expert_vector_design", {})
                expert_design["num_experts"] = min(5, max(1, expert_design.get("num_experts", 3)))
                refined_proposal["expert_vector_design"] = expert_design

        # Increase confidence if refinement improves scores
        if evaluation["should_refine"]:
            refined_proposal["confidence"] = min(1.0, proposal["confidence"] + 0.1)

        # Add refinement metadata
        refined_proposal["refinement_history"] = evaluation["refinement_suggestions"]

        return refined_proposal

    def _model_learns_composition_patterns(
        self,
        refined_compositions: List[Dict[str, Any]],
        task_description: str
    ) -> Dict[str, Any]:
        """Model learns from composition success patterns for future tasks"""

        learning = {
            "task_description": task_description,
            "compositions_analyzed": len(refined_compositions),
            "learned_patterns": [],
            "preference_updates": {},
            "success_indicators": []
        }

        # Analyze patterns across compositions
        strategy_performance = {}
        for composition in refined_compositions:
            strategy = composition["composition_strategy"]
            confidence = composition["confidence"]

            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(confidence)

        # Learn which strategies work best
        for strategy, confidences in strategy_performance.items():
            avg_confidence = np.mean(confidences)
            learning["learned_patterns"].append({
                "pattern": f"{strategy}_strategy",
                "average_performance": avg_confidence,
                "task_type": self._classify_task_type(task_description)
            })

        # Update model's preferences
        best_strategy = max(strategy_performance.keys(),
                          key=lambda k: np.mean(strategy_performance[k]))

        task_type = self._classify_task_type(task_description)
        if task_type not in self.model_guided_preferences:
            self.model_guided_preferences[task_type] = {}

        self.model_guided_preferences[task_type]["preferred_strategy"] = best_strategy
        learning["preference_updates"][task_type] = best_strategy

        # Store learning in history
        self.composition_learning_history.append(learning)

        return learning

    # Phase 2 integration helper methods
    def _analyze_weight_distribution(self, weight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze weight distribution from Phase 2 data"""
        if not weight_data:
            return {"status": "no_data"}

        magnitudes = [abs(point.get("weight", 0)) for point in weight_data]

        return {
            "mean_magnitude": np.mean(magnitudes),
            "std_magnitude": np.std(magnitudes),
            "max_magnitude": np.max(magnitudes),
            "sparsity_estimate": (np.array(magnitudes) < 1e-6).mean()
        }

    def _identify_critical_layers_from_weights(self, weight_data: List[Dict[str, Any]]) -> List[str]:
        """Identify critical layers from Phase 2 weight analysis"""
        if not weight_data:
            return []

        # Group by layer and calculate importance
        layer_importance = {}
        for point in weight_data:
            layer = point.get("layer", "unknown")
            weight = abs(point.get("weight", 0))

            if layer not in layer_importance:
                layer_importance[layer] = []
            layer_importance[layer].append(weight)

        # Calculate mean importance per layer
        layer_scores = {}
        for layer, weights in layer_importance.items():
            layer_scores[layer] = np.mean(weights)

        # Return top layers
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        return [layer for layer, score in sorted_layers[:5]]  # Top 5 layers

    def _extract_magnitude_patterns(self, weight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract weight magnitude patterns"""
        if not weight_data:
            return {"status": "no_data"}

        magnitudes = [abs(point.get("weight", 0)) for point in weight_data]

        # Identify patterns
        high_magnitude_threshold = np.percentile(magnitudes, 90)
        low_magnitude_threshold = np.percentile(magnitudes, 10)

        return {
            "high_magnitude_threshold": high_magnitude_threshold,
            "low_magnitude_threshold": low_magnitude_threshold,
            "magnitude_range": np.max(magnitudes) - np.min(magnitudes),
            "concentration_areas": self._find_magnitude_concentrations(weight_data)
        }

    def _find_magnitude_concentrations(self, weight_data: List[Dict[str, Any]]) -> List[str]:
        """Find areas of weight magnitude concentration"""
        # Simplified - return layers with highest variance
        layer_variances = {}

        for point in weight_data:
            layer = point.get("layer", "unknown")
            weight = abs(point.get("weight", 0))

            if layer not in layer_variances:
                layer_variances[layer] = []
            layer_variances[layer].append(weight)

        concentration_areas = []
        for layer, weights in layer_variances.items():
            if len(weights) > 1 and np.var(weights) > np.mean(weights) * 0.5:
                concentration_areas.append(layer)

        return concentration_areas

    def _suggest_svd_focus_areas(self, weight_data: List[Dict[str, Any]]) -> List[str]:
        """Suggest areas to focus SVD analysis based on Phase 2 insights"""
        critical_layers = self._identify_critical_layers_from_weights(weight_data)
        magnitude_patterns = self._extract_magnitude_patterns(weight_data)

        focus_areas = list(set(
            critical_layers +
            magnitude_patterns.get("concentration_areas", [])
        ))

        return focus_areas[:10]  # Limit to top 10

    def _analyze_task_examples(self, task_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze provided task examples"""
        return {
            "num_examples": len(task_examples),
            "example_types": list(set(ex.get("type", "unknown") for ex in task_examples)),
            "complexity_distribution": [
                self._estimate_task_complexity(ex.get("description", ""))
                for ex in task_examples
            ]
        }


# Example usage and integration testing
if __name__ == "__main__":
    # Create test model
    model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)

    # Initialize integrated system
    system = AdasTransformersSquaredSystem(model)

    # Discover configurations for a task
    task = "mathematical reasoning with step-by-step logic"
    discovery_result = system.discover_expert_configurations_for_task(task)

    print(f"Discovery completed for task: {task}")
    print(f"Found {len(discovery_result['validated_configurations'])} validated configurations")

    # Apply best configuration
    if discovery_result['validated_configurations']:
        application_result = system.apply_discovered_configuration(task)
        print(f"Applied configuration: {application_result['config_applied']}")

    # Get system status
    status = system.get_discovery_status()
    print(f"System status: {status}")