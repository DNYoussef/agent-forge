/**
 * Session Management Utilities
 * Handles mapping and state management between Next.js and Python backend
 */

import { v4 as uuidv4 } from 'uuid';
import { PhaseStatus, SessionState } from '../types/phase-interfaces';

export interface SessionMapping {
  nextjsSessionId: string;
  pythonSessionId: string;
  createdAt: Date;
  lastActivity: Date;
  status: PhaseStatus;
  currentPhase: string | null;
}

class SessionManager {
  private sessions = new Map<string, SessionMapping>();
  private cleanupInterval: NodeJS.Timeout;

  constructor() {
    // Clean up inactive sessions every 30 minutes
    this.cleanupInterval = setInterval(() => {
      this.cleanupInactiveSessions();
    }, 30 * 60 * 1000);
  }

  /**
   * Create a new session mapping
   */
  createSession(nextjsSessionId?: string): SessionMapping {
    const sessionId = nextjsSessionId || this.generateSessionId();

    const mapping: SessionMapping = {
      nextjsSessionId: sessionId,
      pythonSessionId: sessionId, // Use same ID for simplicity
      createdAt: new Date(),
      lastActivity: new Date(),
      status: PhaseStatus.IDLE,
      currentPhase: null,
    };

    this.sessions.set(sessionId, mapping);
    console.log(`[SessionManager] Created session mapping: ${sessionId}`);

    return mapping;
  }

  /**
   * Get session mapping by ID
   */
  getSession(sessionId: string): SessionMapping | null {
    const mapping = this.sessions.get(sessionId);
    if (mapping) {
      mapping.lastActivity = new Date();
    }
    return mapping || null;
  }

  /**
   * Update session status
   */
  updateSessionStatus(sessionId: string, status: PhaseStatus, currentPhase?: string): boolean {
    const mapping = this.sessions.get(sessionId);
    if (!mapping) {
      return false;
    }

    mapping.status = status;
    mapping.lastActivity = new Date();
    if (currentPhase !== undefined) {
      mapping.currentPhase = currentPhase;
    }

    console.log(`[SessionManager] Updated session ${sessionId}: ${status} (${currentPhase || 'no phase'})`);
    return true;
  }

  /**
   * Delete session mapping
   */
  deleteSession(sessionId: string): boolean {
    const deleted = this.sessions.delete(sessionId);
    if (deleted) {
      console.log(`[SessionManager] Deleted session mapping: ${sessionId}`);
    }
    return deleted;
  }

  /**
   * List all sessions
   */
  listSessions(): SessionMapping[] {
    return Array.from(this.sessions.values());
  }

  /**
   * Get active sessions (by status)
   */
  getActiveSessions(): SessionMapping[] {
    return Array.from(this.sessions.values()).filter(
      session => session.status === PhaseStatus.RUNNING ||
                session.status === PhaseStatus.INITIALIZING
    );
  }

  /**
   * Clean up inactive sessions (older than 2 hours with no activity)
   */
  private cleanupInactiveSessions(): void {
    const cutoffTime = new Date(Date.now() - 2 * 60 * 60 * 1000); // 2 hours ago
    let cleanedCount = 0;

    for (const [sessionId, mapping] of this.sessions.entries()) {
      if (mapping.lastActivity < cutoffTime &&
          mapping.status !== PhaseStatus.RUNNING) {
        this.sessions.delete(sessionId);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      console.log(`[SessionManager] Cleaned up ${cleanedCount} inactive sessions`);
    }
  }

  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    return `session_${uuidv4().replace(/-/g, '').substr(0, 16)}`;
  }

  /**
   * Get session statistics
   */
  getStatistics(): {
    totalSessions: number;
    activeSessions: number;
    sessionsByStatus: Record<string, number>;
    oldestSession: Date | null;
    newestSession: Date | null;
  } {
    const sessions = Array.from(this.sessions.values());
    const sessionsByStatus: Record<string, number> = {};

    for (const status of Object.values(PhaseStatus)) {
      sessionsByStatus[status] = 0;
    }

    let oldestSession: Date | null = null;
    let newestSession: Date | null = null;

    for (const session of sessions) {
      sessionsByStatus[session.status]++;

      if (!oldestSession || session.createdAt < oldestSession) {
        oldestSession = session.createdAt;
      }
      if (!newestSession || session.createdAt > newestSession) {
        newestSession = session.createdAt;
      }
    }

    return {
      totalSessions: sessions.length,
      activeSessions: this.getActiveSessions().length,
      sessionsByStatus,
      oldestSession,
      newestSession,
    };
  }

  /**
   * Validate session ID format
   */
  isValidSessionId(sessionId: string): boolean {
    if (!sessionId || typeof sessionId !== 'string') {
      return false;
    }

    // Allow various session ID formats
    const patterns = [
      /^session_[a-zA-Z0-9]{16}$/,          // Generated IDs
      /^[a-zA-Z0-9\-_]{8,64}$/,             // General format
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i, // UUID
    ];

    return patterns.some(pattern => pattern.test(sessionId));
  }

  /**
   * Shutdown session manager and cleanup
   */
  shutdown(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    console.log('[SessionManager] Shutdown completed');
  }
}

// Global session manager instance
export const sessionManager = new SessionManager();

// Utility functions
export const sessionUtils = {
  /**
   * Ensure session exists, create if needed
   */
  async ensureSession(sessionId: string): Promise<SessionMapping> {
    let mapping = sessionManager.getSession(sessionId);
    if (!mapping) {
      mapping = sessionManager.createSession(sessionId);
    }
    return mapping;
  },

  /**
   * Validate and sanitize session ID
   */
  validateSessionId(sessionId: any): string | null {
    if (typeof sessionId !== 'string' || !sessionId.trim()) {
      return null;
    }

    const cleaned = sessionId.trim();
    return sessionManager.isValidSessionId(cleaned) ? cleaned : null;
  },

  /**
   * Get or create session with validation
   */
  async getOrCreateSession(sessionId?: string): Promise<{
    mapping: SessionMapping;
    isNew: boolean
  }> {
    let validSessionId: string | null = null;

    if (sessionId) {
      validSessionId = this.validateSessionId(sessionId);
    }

    if (validSessionId) {
      const existing = sessionManager.getSession(validSessionId);
      if (existing) {
        return { mapping: existing, isNew: false };
      }
    }

    // Create new session
    const mapping = sessionManager.createSession(validSessionId || undefined);
    return { mapping, isNew: true };
  },

  /**
   * Mark session as active (touch last activity)
   */
  touchSession(sessionId: string): boolean {
    const mapping = sessionManager.getSession(sessionId);
    return !!mapping; // getSession already updates lastActivity
  },
};