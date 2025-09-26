-- Agent Forge Initial Database Schema
-- Version: 1.0.0
-- Environment: staging
-- Created: 2025-09-26

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    capabilities TEXT[],
    model_info JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(name)
);

-- Create swarms table
CREATE TABLE IF NOT EXISTS swarms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    topology VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    max_agents INTEGER DEFAULT 10,
    current_agents INTEGER DEFAULT 0,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(name)
);

-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    swarm_id UUID REFERENCES swarms(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    parent_task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    result JSONB,
    error_info JSONB,
    execution_time_ms INTEGER,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create swarm_agents junction table
CREATE TABLE IF NOT EXISTS swarm_agents (
    swarm_id UUID REFERENCES swarms(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    role VARCHAR(100),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (swarm_id, agent_id)
);

-- Create models table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    config JSONB,
    performance_metrics JSONB,
    cost_metrics JSONB,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, provider)
);

-- Create model_training table
CREATE TABLE IF NOT EXISTS model_training (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    training_data JSONB,
    training_config JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    progress DECIMAL(5,2) DEFAULT 0.0,
    metrics JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_type VARCHAR(50) NOT NULL,
    swarm_id UUID REFERENCES swarms(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB,
    performance_data JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER
);

-- Create metrics table for historical data
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL, -- 'agent', 'swarm', 'task', 'model'
    entity_id UUID NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    metric_data JSONB,
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL
);

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    action VARCHAR(100) NOT NULL,
    old_values JSONB,
    new_values JSONB,
    user_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create performance_benchmarks table
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    benchmark_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    benchmark_config JSONB,
    results JSONB,
    score DECIMAL(10,4),
    percentile DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);
CREATE INDEX IF NOT EXISTS idx_agents_updated_at ON agents(updated_at);

CREATE INDEX IF NOT EXISTS idx_swarms_status ON swarms(status);
CREATE INDEX IF NOT EXISTS idx_swarms_topology ON swarms(topology);
CREATE INDEX IF NOT EXISTS idx_swarms_updated_at ON swarms(updated_at);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_swarm_id ON tasks(swarm_id);
CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);

CREATE INDEX IF NOT EXISTS idx_metrics_entity ON metrics(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_metrics_collected_at ON metrics(collected_at);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);

CREATE INDEX IF NOT EXISTS idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);

-- Create functions for updating updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic updated_at updates
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_swarms_updated_at BEFORE UPDATE ON swarms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data
INSERT INTO models (name, provider, model_type, config, status) VALUES
('gpt-4', 'openai', 'language_model', '{"context_length": 8192, "temperature": 0.7}', 'active'),
('claude-3-opus', 'anthropic', 'language_model', '{"context_length": 200000, "temperature": 0.7}', 'active'),
('gemini-pro', 'google', 'language_model', '{"context_length": 1000000, "temperature": 0.7}', 'active'),
('gpt-3.5-turbo', 'openai', 'language_model', '{"context_length": 4096, "temperature": 0.7}', 'active')
ON CONFLICT (name, provider) DO NOTHING;

-- Insert sample agent types
INSERT INTO agents (name, type, status, config, capabilities) VALUES
('coordinator-agent', 'coordinator', 'inactive', '{"max_tasks": 100, "timeout": 300}', ARRAY['coordination', 'task_management']),
('researcher-agent', 'researcher', 'inactive', '{"max_search_depth": 5, "timeout": 600}', ARRAY['research', 'analysis', 'web_search']),
('coder-agent', 'coder', 'inactive', '{"max_file_size": 10000, "timeout": 1200}', ARRAY['coding', 'debugging', 'refactoring']),
('tester-agent', 'tester', 'inactive', '{"test_types": ["unit", "integration"], "timeout": 900}', ARRAY['testing', 'validation', 'quality_assurance'])
ON CONFLICT (name) DO NOTHING;

-- Insert sample swarm topology
INSERT INTO swarms (name, topology, status, config, max_agents) VALUES
('default-mesh-swarm', 'mesh', 'inactive', '{"coordination_strategy": "consensus", "load_balancing": true}', 10),
('hierarchical-swarm', 'hierarchical', 'inactive', '{"coordination_strategy": "top_down", "depth": 3}', 15),
('research-swarm', 'star', 'inactive', '{"coordination_strategy": "centralized", "specialized": true}', 8)
ON CONFLICT (name) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW active_agents AS
SELECT 
    id,
    name,
    type,
    status,
    capabilities,
    model_info,
    created_at,
    last_activity_at
FROM agents 
WHERE status IN ('active', 'busy');

CREATE OR REPLACE VIEW swarm_summary AS
SELECT 
    s.id,
    s.name,
    s.topology,
    s.status,
    s.current_agents,
    s.max_agents,
    COUNT(DISTINCT t.id) as total_tasks,
    COUNT(DISTINCT CASE WHEN t.status = 'pending' THEN t.id END) as pending_tasks,
    COUNT(DISTINCT CASE WHEN t.status = 'running' THEN t.id END) as running_tasks,
    COUNT(DISTINCT CASE WHEN t.status = 'completed' THEN t.id END) as completed_tasks,
    s.created_at,
    s.last_activity_at
FROM swarms s
LEFT JOIN tasks t ON s.id = t.swarm_id
GROUP BY s.id, s.name, s.topology, s.status, s.current_agents, s.max_agents, s.created_at, s.last_activity_at;

CREATE OR REPLACE VIEW recent_metrics AS
SELECT 
    entity_type,
    entity_id,
    metric_name,
    metric_value,
    metric_data,
    collected_at,
    ROW_NUMBER() OVER (PARTITION BY entity_type, entity_id, metric_name ORDER BY collected_at DESC) as rn
FROM metrics
WHERE collected_at >= NOW() - INTERVAL '24 hours';

-- Grant permissions (for staging environment)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT USAGE ON SCHEMA public TO postgres;

-- Commit the transaction
COMMIT;