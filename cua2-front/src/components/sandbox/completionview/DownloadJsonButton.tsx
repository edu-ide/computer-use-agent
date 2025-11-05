import React from 'react';
import { Button, Tooltip } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';

interface DownloadJsonButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

/**
 * Button to download trace as JSON
 */
export const DownloadJsonButton: React.FC<DownloadJsonButtonProps> = ({
  onClick,
  disabled = false,
}) => {
  return (
    <Tooltip
      title={
        disabled
          ? "No trace available"
          : "Download trace as JSON"
      }
    >
      <span>
        <Button
          variant="outlined"
          size="small"
          onClick={onClick}
          disabled={disabled}
          startIcon={<DownloadIcon sx={{ fontSize: '1.2rem' }} />}
          sx={{
            textTransform: 'none',
            fontSize: '0.75rem',
            fontWeight: 600,
            borderRadius: 1,
            px: 1.5,
            py: 0.5,
            borderColor: 'divider',
            color: 'text.primary',
            '&:hover': {
              borderColor: 'primary.main',
              backgroundColor: 'action.hover',
            },
            '&.Mui-disabled': {
              borderColor: 'divider',
              color: 'text.disabled',
            },
          }}
        >
          Download JSON Trace
        </Button>
      </span>
    </Tooltip>
  );
};
