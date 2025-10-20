import { AgentStep } from '@/types/agent';
import React from 'react';

interface StepCardProps {
    step: AgentStep;
    index: number;
}

export const StepCard: React.FC<StepCardProps> = ({ step, index }) => {
    return (
        <div
            key={step.stepId}
            style={{ backgroundColor: '#f9fafb', borderRadius: '8px', border: '1px solid #d1d5db', padding: '12px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)' }}
            className="hover:border-blue-400 transition-all"
        >
            {/* Step Header */}
            <div className="mb-6">
                <span className="text-xs font-bold text-blue-600 uppercase tracking-wide">Step {index + 1}</span>
                <hr style={{ margin: '12px 0', border: 'none', borderTop: '2px solid #d1d5db' }} />
            </div>

            {/* Step Image */}
            {step.image && (
                <div className="mb-6">
                    <div className="rounded-md overflow-hidden border border-gray-300 bg-white" style={{ maxHeight: '140px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <img
                            src={step.image}
                            alt={`Step ${index + 1}`}
                            style={{ width: '100%', height: 'auto', maxHeight: '140px', objectFit: 'contain' }}
                        />
                    </div>
                    <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #e5e7eb' }} />
                </div>
            )}

            {/* Thought */}
            <div className="mb-6">
                <div className="bg-white rounded-md p-2.5 border border-gray-200">
                    <h4 className="text-xs font-semibold text-gray-700 mb-1.5 flex items-center gap-1">
                        <span>💭</span>
                        <span>Thought</span>
                    </h4>
                    <p className="text-xs text-gray-600 leading-relaxed">{step.thought}</p>
                </div>
                <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #e5e7eb' }} />
            </div>

            {/* Actions */}
            <div className="mb-6">
                <div className="bg-white rounded-md p-2.5 border border-gray-200">
                    <h4 className="text-xs font-semibold text-gray-700 mb-1.5 flex items-center gap-1">
                        <span>⚡</span>
                        <span>Actions</span>
                    </h4>
                    <ul className="space-y-1" style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                        {step.actions.map((action, actionIndex) => (
                            <li key={actionIndex} className="text-xs text-gray-600 flex items-start leading-snug">
                                <span className="mr-1.5 text-blue-500 flex-shrink-0">→</span>
                                <span className="break-words">{action}</span>
                            </li>
                        ))}
                    </ul>
                </div>
                <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #e5e7eb' }} />
            </div>

            {/* Step Metadata Footer */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#eff6ff', borderRadius: '6px', padding: '6px 8px', border: '1px solid #bfdbfe' }}>
                    <span style={{ fontSize: '9px', fontWeight: 500, color: '#2563eb', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Time</span>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#1e40af' }}>{step.duration.toFixed(2)}s</span>
                </div>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f0fdf4', borderRadius: '6px', padding: '6px 8px', border: '1px solid #bbf7d0' }}>
                    <span style={{ fontSize: '9px', fontWeight: 500, color: '#059669', textTransform: 'uppercase', letterSpacing: '0.5px' }}>In Tokens</span>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#166534' }}>{step.inputTokensUsed}</span>
                </div>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#faf5ff', borderRadius: '6px', padding: '6px 8px', border: '1px solid #e9d5ff' }}>
                    <span style={{ fontSize: '9px', fontWeight: 500, color: '#9333ea', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Out Tokens</span>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#6b21a8' }}>{step.outputTokensUsed}</span>
                </div>
            </div>
        </div>
    );
};
