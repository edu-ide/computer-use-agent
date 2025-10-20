import React from 'react';

interface VNCStreamProps {
    vncUrl: string;
}

export const VNCStream: React.FC<VNCStreamProps> = ({ vncUrl }) => {
    return (
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', backgroundColor: 'white', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)', border: '1px solid #e5e7eb', overflow: 'hidden', padding: '20px' }}>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">VNC Stream</h3>
            <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {vncUrl ? (
                    <iframe
                        src={vncUrl}
                        style={{ width: '100%', height: '100%', border: 'none' }}
                        title="VNC Stream"
                    />
                ) : (
                    <div className="text-gray-400 text-center p-8">
                        <svg className="w-16 h-16 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                        <p className="font-medium">No VNC stream available</p>
                        <p className="text-sm mt-1 text-gray-500">Stream will appear when agent starts</p>
                    </div>
                )}
            </div>
        </div>
    );
};
