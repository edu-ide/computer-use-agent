import React, { useEffect, useState } from 'react';
import {
    Box,
    Container,
    Grid,
    Paper,
    Typography,
    Chip,
    Card,
    CardContent,
    Divider,
    Tab,
    Tabs,
    Stack,
    CircularProgress,
} from '@mui/material';
import { FiCpu, FiPackage, FiLayers, FiCode } from 'react-icons/fi';
import ReactJson from 'react-json-view';

interface StandardNode {
    type: string;
    name: string;
    display_name_ko?: string;
    description: string;
    description_ko?: string;
    factory_function: string;
    parameters: Array<{
        name: string;
        type: string;
        description?: string;
        description_ko?: string;
        default?: any;
    }>;
}

interface AgentType {
    type: string;
    name: string;
    name_ko?: string;
    description: string;
    description_ko?: string;
    capabilities: string[];
    capabilities_ko?: string[];
    recommended_for: string[];
    recommended_for_ko?: string[];
}

export const Library = () => {
    const [activeTab, setActiveTab] = useState(0);
    const [nodes, setNodes] = useState<StandardNode[]>([]);
    const [agents, setAgents] = useState<AgentType[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [nodesRes, agentsRes] = await Promise.all([
                    fetch('/api/library/nodes'),
                    fetch('/api/library/agents'),
                ]);

                if (nodesRes.ok) setNodes(await nodesRes.json());
                if (agentsRes.ok) setAgents(await agentsRes.json());
            } catch (error) {
                console.error('Failed to fetch library data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setActiveTab(newValue);
    };

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Container maxWidth="xl" sx={{ py: 4 }}>
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <FiPackage /> 노드 & 에이전트 라이브러리
                </Typography>
                <Typography variant="body1" color="text.secondary">
                    사용 가능한 표준 노드 및 에이전트 타입에 대한 참조입니다.
                </Typography>
            </Box>

            <Paper sx={{ mb: 3 }}>
                <Tabs value={activeTab} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tab icon={<FiLayers />} iconPosition="start" label="표준 노드 (Standard Nodes)" />
                    <Tab icon={<FiCpu />} iconPosition="start" label="에이전트 타입 (Agent Types)" />
                </Tabs>
            </Paper>

            {activeTab === 0 && (
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: 3 }}>
                    {nodes.map((node, index) => (
                        <Box key={index}>
                            <Card variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                                <CardContent>
                                    <Stack direction="row" justifyContent="space-between" alignItems="flex-start" mb={2}>
                                        <Typography variant="h6" fontWeight="bold">
                                            {node.display_name_ko || node.name}
                                        </Typography>
                                        <Chip label={node.type} size="small" color="primary" variant="outlined" />
                                    </Stack>
                                    <Typography variant="body2" color="text.secondary" paragraph>
                                        {node.description_ko || node.description}
                                    </Typography>

                                    <Divider sx={{ my: 2 }} />

                                    <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <FiCode /> Factory Function
                                    </Typography>
                                    <Paper variant="outlined" sx={{ p: 1, bgcolor: '#f8fafc', mb: 2, fontFamily: 'monospace', fontSize: '0.875rem' }}>
                                        {node.factory_function}
                                    </Paper>

                                    <Typography variant="subtitle2" gutterBottom>
                                        Parameters
                                    </Typography>
                                    <Stack spacing={1}>
                                        {node.parameters.map((param, idx) => (
                                            <Box key={idx} sx={{ p: 1, border: '1px solid #e2e8f0', borderRadius: 1 }}>
                                                <Stack direction="row" spacing={1} alignItems="center">
                                                    <Typography variant="subtitle2" sx={{ fontFamily: 'monospace' }}>
                                                        {param.name}
                                                    </Typography>
                                                    <Chip label={param.type} size="small" sx={{ height: 20, fontSize: '0.75rem' }} />
                                                    {param.default !== undefined && (
                                                        <Typography variant="caption" color="text.secondary">
                                                            (기본값: {String(param.default)})
                                                        </Typography>
                                                    )}
                                                </Stack>
                                                {(param.description_ko || param.description) && (
                                                    <Typography variant="caption" color="text.secondary" display="block" mt={0.5}>
                                                        {param.description_ko || param.description}
                                                    </Typography>
                                                )}
                                            </Box>
                                        ))}
                                    </Stack>
                                </CardContent>
                            </Card>
                        </Box>
                    ))}
                </Box>
            )}

            {activeTab === 1 && (
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: 3 }}>
                    {agents.map((agent, index) => (
                        <Box key={index}>
                            <Card variant="outlined" sx={{ height: '100%' }}>
                                <CardContent>
                                    <Stack direction="row" justifyContent="space-between" alignItems="flex-start" mb={2}>
                                        <Typography variant="h6" fontWeight="bold">
                                            {agent.name_ko || agent.name}
                                        </Typography>
                                        <Chip label={agent.type} size="small" color="secondary" variant="outlined" />
                                    </Stack>
                                    <Typography variant="body2" color="text.secondary" paragraph>
                                        {agent.description_ko || agent.description}
                                    </Typography>

                                    <Divider sx={{ my: 2 }} />

                                    <Typography variant="subtitle2" gutterBottom>
                                        Capabilities
                                    </Typography>
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                                        {(agent.capabilities_ko || agent.capabilities).map((cap, idx) => (
                                            <Chip key={idx} label={cap} size="small" />
                                        ))}
                                    </Box>

                                    <Typography variant="subtitle2" gutterBottom>
                                        Recommended For
                                    </Typography>
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                        {(agent.recommended_for_ko || agent.recommended_for).map((rec, idx) => (
                                            <Chip key={idx} label={rec} size="small" variant="outlined" />
                                        ))}
                                    </Box>
                                </CardContent>
                            </Card>
                        </Box>
                    ))}
                </Box>
            )}
        </Container>
    );
};
