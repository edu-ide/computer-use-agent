import React from 'react';

interface TaskButtonProps {
  isAgentProcessing: boolean;
  isConnected: boolean;
  onSendTask: (content: string, modelId: string) => void;
}

export const TaskButton: React.FC<TaskButtonProps> = ({ isAgentProcessing, isConnected, onSendTask }) => {
  return (
    <div
      onClick={() => {
        if (!isAgentProcessing && isConnected) {
          onSendTask(
            "Complete the online form by clicking through the required fields",
            "anthropic/claude-sonnet-4-5-20250929"
          );
        }
      }}
      style={{
        marginTop: '16px',
        padding: '14px 18px',
        background: isAgentProcessing || !isConnected
          ? 'rgba(255, 255, 255, 0.1)'
          : 'rgba(255, 255, 255, 0.15)',
        borderRadius: '10px',
        backdropFilter: 'blur(10px)',
        border: '2px solid rgba(0, 0, 0, 0.3)',
        cursor: isAgentProcessing || !isConnected ? 'not-allowed' : 'pointer',
        transition: 'all 0.3s ease',
        opacity: isAgentProcessing || !isConnected ? 0.6 : 1,
      }}
      onMouseEnter={(e) => {
        if (!isAgentProcessing && isConnected) {
          e.currentTarget.style.background = 'rgba(200, 200, 200, 0.3)';
          e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.5)';
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.2)';
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
        e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.3)';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: 'rgba(0, 0, 0, 0.7)', textTransform: 'uppercase', letterSpacing: '1px' }}>Task</span>
            {!isAgentProcessing && isConnected && (
              <span style={{ fontSize: '10px', color: 'rgba(0, 0, 0, 0.5)', fontStyle: 'italic' }}>
                (click to run)
              </span>
            )}
          </div>
          <p style={{ fontSize: '15px', fontWeight: 500, color: '#1f2937' }}>
            Complete the online form by clicking through the required fields
          </p>
        </div>
        <div style={{
          padding: '8px 16px',
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          borderRadius: '6px',
          border: '1px solid rgba(0, 0, 0, 0.2)'
        }}>
          <span style={{ fontSize: '11px', fontWeight: 600, color: 'rgba(0, 0, 0, 0.6)', textTransform: 'uppercase', letterSpacing: '1px' }}>Model</span>
          <p style={{ fontSize: '12px', fontWeight: 600, color: '#1f2937', marginTop: '2px', whiteSpace: 'nowrap' }}>
            claude-sonnet-4-5-20250929
          </p>
        </div>
      </div>
    </div>
  );
};
