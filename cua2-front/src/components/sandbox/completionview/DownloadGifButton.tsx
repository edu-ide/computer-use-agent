import React from 'react';
import { Button, CircularProgress, Tooltip } from '@mui/material';
import GifIcon from '@mui/icons-material/Gif';

interface DownloadGifButtonProps {
  isGenerating: boolean;
  onClick: () => void;
  disabled?: boolean;
}

/**
 * Button to download a GIF replay of the trace
 */
export const DownloadGifButton: React.FC<DownloadGifButtonProps> = ({
  isGenerating,
  onClick,
  disabled = false,
}) => {
  return (
    <Tooltip
      title={
        disabled
          ? "No steps available"
          : "Download GIF replay"
      }
    >
      <span>
        <Button
          variant="outlined"
          size="small"
          onClick={onClick}
          disabled={disabled || isGenerating}
          startIcon={
            isGenerating ? (
              <CircularProgress size={16} />
            ) : (
              <GifIcon sx={{ fontSize: '1.2rem' }} />
            )
          }
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
          {isGenerating ? 'Generating...' : 'Download GIF'}
        </Button>
      </span>
    </Tooltip>
  );
};
