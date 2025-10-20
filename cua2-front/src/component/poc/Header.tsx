import React from 'react';
import { ConnectionStatus } from './ConnectionStatus';
import { ProcessingIndicator } from './ProcessingIndicator';
import { TaskButton } from './TaskButton';

interface HeaderProps {
    isConnected: boolean;
    isAgentProcessing: boolean;
    onSendTask: (content: string, modelId: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ isConnected, isAgentProcessing, onSendTask }) => {
    return (
        <>
            <div style={{
                flexShrink: 0,
            }}>
                <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px 32px' }}>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-6">
                            <ConnectionStatus isConnected={isConnected} />
                            <h1 className="text-3xl font-bold text-white" style={{ textShadow: '0 2px 4px rgba(0, 0, 0, 0.2)' }}>
                                CUA2 Agent
                            </h1>
                        </div>
                        <ProcessingIndicator isAgentProcessing={isAgentProcessing} />
                    </div>
                    <TaskButton
                        isAgentProcessing={isAgentProcessing}
                        isConnected={isConnected}
                        onSendTask={onSendTask}
                    />
                </div>
            </div>

            <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
        </>
    );
};
