-- Staging Database Initialization Script
-- This script runs during PostgreSQL container startup

-- Create databases
CREATE DATABASE agent_forge_staging;
CREATE DATABASE agent_forge_test;
CREATE DATABASE grafana_staging;

-- Create users
CREATE USER staging_user WITH PASSWORD 'staging_password_change_me';
CREATE USER grafana_user WITH PASSWORD 'grafana_password_change_me';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE agent_forge_staging TO staging_user;
GRANT ALL PRIVILEGES ON DATABASE agent_forge_test TO staging_user;
GRANT ALL PRIVILEGES ON DATABASE grafana_staging TO grafana_user;

-- Set up staging database
\c agent_forge_staging;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO staging_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO staging_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO staging_user;

-- Set up test database
\c agent_forge_test;

-- Grant schema privileges for test database
GRANT ALL ON SCHEMA public TO staging_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO staging_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO staging_user;

-- Set up Grafana database
\c grafana_staging;

-- Grant schema privileges for Grafana
GRANT ALL ON SCHEMA public TO grafana_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO grafana_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO grafana_user;

-- Switch back to postgres database
\c postgres;

-- Create additional configuration
ALTER DATABASE agent_forge_staging SET timezone TO 'UTC';
ALTER DATABASE agent_forge_test SET timezone TO 'UTC';
ALTER DATABASE grafana_staging SET timezone TO 'UTC';