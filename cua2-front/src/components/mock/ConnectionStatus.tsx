import React from 'react';

interface ConnectionStatusProps {
  isConnected: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ isConnected }) => {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      backgroundColor: 'rgba(255, 255, 255, 0.2)',
      padding: '8px 16px',
      borderRadius: '20px',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255, 255, 255, 0.3)'
    }}>
      <div style={{
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        backgroundColor: isConnected ? '#10b981' : '#ef4444',
        boxShadow: isConnected ? '0 0 8px #10b981' : '0 0 8px #ef4444',
        animation: isConnected ? 'pulse 2s infinite' : 'none'
      }}></div>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <span className="text-xs font-semibold text-white" style={{ lineHeight: '1.2' }}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
        <span className="text-xs text-white" style={{ opacity: 0.7, fontSize: '10px', lineHeight: '1.2' }}>
          WebSocket
        </span>
      </div>
    </div>
  );
};
