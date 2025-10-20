import { AgentTrace } from '@/types/agent';
import React from 'react';

interface MetadataProps {
    trace?: AgentTrace;
}

export const Metadata: React.FC<MetadataProps> = ({ trace }) => {
    return (
        <div style={{ flexShrink: 0 }} className="bg-white rounded-lg shadow-md border border-gray-200 p-5">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Metadata</h3>
            {trace?.metadata ? (
                <div style={{ display: 'flex', gap: '12px' }}>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#eff6ff', borderRadius: '8px', padding: '12px', border: '1px solid #bfdbfe', boxShadow: '0 2px 4px rgba(59, 130, 246, 0.1)' }}>
                        <span style={{ fontSize: '10px', fontWeight: 600, color: '#2563eb', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '6px' }}>Total Time</span>
                        <span style={{ fontSize: '24px', fontWeight: 700, color: '#1e40af' }}>{trace.metadata.duration.toFixed(2)}s</span>
                    </div>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f0fdf4', borderRadius: '8px', padding: '12px', border: '1px solid #bbf7d0', boxShadow: '0 2px 4px rgba(16, 185, 129, 0.1)' }}>
                        <span style={{ fontSize: '10px', fontWeight: 600, color: '#059669', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '6px' }}>In Tokens</span>
                        <span style={{ fontSize: '24px', fontWeight: 700, color: '#166534' }}>{trace.metadata.inputTokensUsed.toLocaleString()}</span>
                    </div>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#faf5ff', borderRadius: '8px', padding: '12px', border: '1px solid #e9d5ff', boxShadow: '0 2px 4px rgba(139, 92, 246, 0.1)' }}>
                        <span style={{ fontSize: '10px', fontWeight: 600, color: '#9333ea', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '6px' }}>Out Tokens</span>
                        <span style={{ fontSize: '24px', fontWeight: 700, color: '#6b21a8' }}>{trace.metadata.outputTokensUsed.toLocaleString()}</span>
                    </div>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff7ed', borderRadius: '8px', padding: '12px', border: '1px solid #fed7aa', boxShadow: '0 2px 4px rgba(249, 115, 22, 0.1)' }}>
                        <span style={{ fontSize: '10px', fontWeight: 600, color: '#ea580c', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '6px' }}>Total Steps</span>
                        <span style={{ fontSize: '24px', fontWeight: 700, color: '#c2410c' }}>{trace.metadata.numberOfSteps}</span>
                    </div>
                </div>
            ) : (
                <div className="text-gray-400 text-sm py-2">
                    {trace ? 'Waiting for completion...' : 'No task started yet'}
                </div>
            )}
        </div>
    );
};
