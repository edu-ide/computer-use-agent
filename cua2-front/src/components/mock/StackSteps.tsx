import React from 'react';
import { AgentTrace } from '@/types/agent';
import { StepCard } from './StepCard';

interface StackStepsProps {
  trace?: AgentTrace;
}

export const StackSteps: React.FC<StackStepsProps> = ({ trace }) => {
  return (
    <div style={{ width: '360px', flexShrink: 0, display: 'flex', flexDirection: 'column', backgroundColor: 'white', borderRadius: '10px', marginLeft: '12px', marginTop: '20px', marginBottom: '20px', boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)', border: '1px solid #e5e7eb' }}>
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Stack Steps</h3>
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: '16px' }}>
        {trace?.steps && trace.steps.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {trace.steps.map((step, index) => (
              <StepCard key={step.stepId} step={step} index={index} />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 p-6 text-center">
            <p className="font-medium">No steps yet</p>
            <p className="text-xs mt-1">Steps will appear as agent progresses</p>
          </div>
        )}
      </div>
    </div>
  );
};
