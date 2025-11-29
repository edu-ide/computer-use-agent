/**
 * 파라미터 설정 모달 - 워크플로우 파라미터 편집
 */

import React, { useMemo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Box,
  Typography,
  IconButton,
  TextField,
  FormControlLabel,
  Checkbox,
  Button,
  Alert,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { FiSettings, FiPlay } from 'react-icons/fi';

interface ParameterConfig {
  name: string;
  type: 'string' | 'number' | 'boolean';
  label: string;
  placeholder?: string;
  default?: unknown;
  required?: boolean;
  min?: number;
  max?: number;
}

interface ParameterModalProps {
  open: boolean;
  onClose: () => void;
  parameters: ParameterConfig[];
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
  onStart: () => void;
  disabled?: boolean;
}

const ParameterModal: React.FC<ParameterModalProps> = ({
  open,
  onClose,
  parameters,
  values,
  onChange,
  onStart,
  disabled = false,
}) => {
  // 파라미터 검증
  const validationErrors = useMemo(() => {
    const errors: string[] = [];

    for (const param of parameters) {
      const value = values[param.name];

      // 필수 파라미터 체크
      if (param.required) {
        if (value === undefined || value === null || value === '') {
          errors.push(`'${param.label}'은(는) 필수 항목입니다.`);
          continue;
        }
      }

      // 값이 없으면 이후 검증 스킵
      if (value === undefined || value === null || value === '') {
        continue;
      }

      // 숫자 범위 체크
      if (param.type === 'number') {
        const numValue = Number(value);
        if (isNaN(numValue)) {
          errors.push(`'${param.label}'은(는) 숫자여야 합니다.`);
        } else {
          if (param.min !== undefined && numValue < param.min) {
            errors.push(`'${param.label}'은(는) ${param.min} 이상이어야 합니다.`);
          }
          if (param.max !== undefined && numValue > param.max) {
            errors.push(`'${param.label}'은(는) ${param.max} 이하여야 합니다.`);
          }
        }
      }
    }

    return errors;
  }, [parameters, values]);

  const hasErrors = validationErrors.length > 0;

  const handleStart = () => {
    if (hasErrors) return;
    onStart();
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          backgroundColor: '#ffffff',
          color: '#1e293b',
          borderRadius: 3,
        },
      }}
    >
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #e2e8f0',
          pb: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #64748b 0%, #475569 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <FiSettings color="#fff" size={20} />
          </Box>
          <Box>
            <Typography variant="h6" fontWeight={700} color="#1e293b">
              파라미터 설정
            </Typography>
            <Typography variant="body2" color="#64748b">
              워크플로우 실행 옵션을 설정하세요
            </Typography>
          </Box>
        </Box>

        <IconButton onClick={onClose} sx={{ color: '#64748b' }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ pt: 3 }}>
        {/* 검증 에러 표시 */}
        {hasErrors && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {validationErrors.map((error, index) => (
              <div key={index}>{error}</div>
            ))}
          </Alert>
        )}

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
          {parameters.map((param) => (
            <Box key={param.name}>
              {param.type === 'boolean' ? (
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={Boolean(values[param.name])}
                      onChange={(e) => onChange(param.name, e.target.checked)}
                      disabled={disabled}
                      sx={{
                        color: '#94a3b8',
                        '&.Mui-checked': {
                          color: '#3b82f6',
                        },
                      }}
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={600} color="#1e293b">
                        {param.label}
                      </Typography>
                      {param.placeholder && (
                        <Typography variant="caption" color="#64748b">
                          {param.placeholder}
                        </Typography>
                      )}
                    </Box>
                  }
                  sx={{ alignItems: 'flex-start', m: 0 }}
                />
              ) : (
                <TextField
                  fullWidth
                  size="small"
                  label={param.required ? `${param.label} *` : param.label}
                  type={param.type === 'number' ? 'number' : 'text'}
                  placeholder={param.placeholder}
                  value={values[param.name] ?? ''}
                  onChange={(e) =>
                    onChange(
                      param.name,
                      param.type === 'number' ? Number(e.target.value) : e.target.value
                    )
                  }
                  disabled={disabled}
                  required={param.required}
                  error={param.required && (values[param.name] === undefined || values[param.name] === null || values[param.name] === '')}
                  helperText={param.required && (values[param.name] === undefined || values[param.name] === null || values[param.name] === '') ? '필수 항목입니다' : ''}
                  inputProps={{ min: param.min, max: param.max }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': {
                        borderColor: '#e2e8f0',
                      },
                      '&:hover fieldset': {
                        borderColor: '#94a3b8',
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: '#3b82f6',
                      },
                    },
                    '& .MuiInputLabel-root': {
                      color: '#64748b',
                      '&.Mui-focused': {
                        color: '#3b82f6',
                      },
                    },
                  }}
                />
              )}
            </Box>
          ))}
        </Box>
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 3, pt: 2, borderTop: '1px solid #e2e8f0' }}>
        <Button
          onClick={onClose}
          sx={{
            color: '#64748b',
            textTransform: 'none',
          }}
        >
          취소
        </Button>
        <Button
          onClick={handleStart}
          variant="contained"
          disabled={disabled || hasErrors}
          startIcon={<FiPlay />}
          sx={{
            textTransform: 'none',
            fontWeight: 600,
            background: hasErrors
              ? '#94a3b8'
              : 'linear-gradient(135deg, #1677ff 0%, #0958d9 100%)',
            '&:hover': {
              background: hasErrors
                ? '#94a3b8'
                : 'linear-gradient(135deg, #0958d9 0%, #003eb3 100%)',
            },
          }}
        >
          {hasErrors ? '필수 항목을 입력하세요' : '실행'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ParameterModal;
