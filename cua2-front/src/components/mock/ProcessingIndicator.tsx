import React from 'react';

interface ProcessingIndicatorProps {
  isAgentProcessing: boolean;
}

export const ProcessingIndicator: React.FC<ProcessingIndicatorProps> = ({ isAgentProcessing }) => {
  if (!isAgentProcessing) return null;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      padding: '10px 20px',
      backgroundColor: 'rgba(251, 191, 36, 0.2)',
      borderRadius: '10px',
      border: '1px solid rgba(251, 191, 36, 0.4)'
    }}>
      <span style={{
        width: '16px',
        height: '16px',
        border: '2px solid #fbbf24',
        borderTopColor: 'transparent',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite',
        display: 'inline-block'
      }}></span>
      <span style={{ fontSize: '14px', fontWeight: 600, color: '#fbbf24', letterSpacing: '0.5px' }}>
        PROCESSING...
      </span>
    </div>
  );
};
